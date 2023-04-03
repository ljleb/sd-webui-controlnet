import dataclasses
from typing import List, Optional, Iterable, TypeVar, Generic

import numpy
import torch
from PIL import Image

from modules import scripts, processing, shared, devices
import gradio as gr
import torchvision

from modules import img2img


img2img_process_batch_tab_callbacks = []
img2img_process_batch_tab_each_callbacks = []
img2img_postprocess_batch_tab_each_callbacks = []
img2img_postprocess_batch_tab_callbacks = []


def img2img_process_batch_hijack(p, *args, **kwargs):
    for callback in img2img_process_batch_tab_callbacks:
        callback(p)

    res_was_none = False
    def img2img_scripts_run_hijack(p, *args):
        nonlocal res_was_none
        if res_was_none:
            for callback in img2img_postprocess_batch_tab_each_callbacks:
                callback(p)

        for callback in img2img_process_batch_tab_each_callbacks:
            callback(p)

        res = original_img2img_scripts_run(p, *args)
        if res is not None:
            for callback in img2img_postprocess_batch_tab_each_callbacks:
                callback(p)
        else:
            res_was_none = True

        return res

    original_img2img_scripts_run = scripts.scripts_img2img.run
    scripts.scripts_img2img.run = img2img_scripts_run_hijack

    try:
        return getattr(img2img, '__batch_loopback_original_process_batch')(p, *args, **kwargs)
    finally:
        if res_was_none:
            for callback in img2img_postprocess_batch_tab_each_callbacks:
                callback(p)

        scripts.scripts_img2img.run = original_img2img_scripts_run
        for callback in img2img_postprocess_batch_tab_callbacks:
            callback(p)


if hasattr(img2img, '__batch_loopback_original_process_batch'):
    # reset in case extension was updated
    img2img.process_batch = getattr(img2img, '__batch_loopback_original_process_batch')

setattr(img2img, '__batch_loopback_original_process_batch', img2img.process_batch)
img2img.process_batch = img2img_process_batch_hijack


T = TypeVar('T')


@dataclasses.dataclass
class GrowingCircularBuffer(Generic[T]):
    buffer: List[T] = dataclasses.field(default_factory=list)
    current_index: int = 0
    size_locked: bool = False

    def lock_size(self, lock: bool=True):
        self.size_locked = lock

    def append(self, value: T):
        if self.size_locked:
            self.buffer[self.current_index % len(self.buffer)] = value
        else:
            self.buffer.append(value)

        self.current_index += 1

    def clear(self):
        for k, v in GrowingCircularBuffer().__dict__.items():
            setattr(self, k, v)

    def get_current(self) -> T:
        return self.buffer[(self.current_index - 1) % len(self.buffer)]

    def __bool__(self):
        return bool(self.buffer)


class BatchLoopbackScript(scripts.Script):
    def __init__(self):
        self.is_img2img_batch = False
        self.output_images = GrowingCircularBuffer()
        self.source_latent = None
        self.seed = -1
        self.subseed = -1

        global img2img_process_batch_tab_callbacks, img2img_process_batch_tab_each_callbacks, img2img_postprocess_batch_tab_each_callbacks, img2img_postprocess_batch_tab_callbacks
        img2img_process_batch_tab_callbacks.append(self.img2img_process_batch_tab)
        img2img_process_batch_tab_each_callbacks.append(self.img2img_process_batch_tab_each)
        img2img_postprocess_batch_tab_each_callbacks.append(self.img2img_postprocess_batch_tab_each)
        img2img_postprocess_batch_tab_callbacks.append(self.img2img_postprocess_batch_tab)

    def title(self):
        return 'Batch Loopback'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        if not is_img2img:
            return [
                gr.State(False), gr.State(False),
                gr.State(0.), gr.State(1.), gr.State(False)
            ]

        extension_name = 'batch_loopback'
        def format_elem_id(name):
            return f'{extension_name}_{name}'

        def label_space_fix(label):
            # extra space fixes broken maximum and default values for some reason -_-
            # also fixes checkboxes default values
            return f'{label} '

        with gr.Accordion(label='Batch Loopback', open=False, elem_id=extension_name):
            enable = gr.Checkbox(
                label='Enable',
                value=False,
                elem_id=f'{extension_name}_enable',
            )

            with gr.Row():
                with gr.Column(scale=3):
                    loopback_mix = gr.Slider(
                        label=label_space_fix('Loopback mix'),
                        value=1.,
                        minimum=0.,
                        maximum=1.,
                        step=.01,
                        elem_id=format_elem_id('loopback_mix'),
                    )

                with gr.Column(min_width=160):
                    wet_mix = gr.Slider(
                        label=label_space_fix('Wet mix'),
                        value=1.,
                        minimum=0.,
                        maximum=1.,
                        step=.01,
                        interactive=True,
                        elem_id=format_elem_id('wet_mix'),
                    )

                    follow_loopback_mix = gr.Checkbox(
                        label=label_space_fix('Follow loopback mix'),
                        value=not wet_mix.interactive,
                        elem_id=format_elem_id('follow_loopback_mix'),
                    )
                    follow_loopback_mix.change(
                        fn=lambda a: gr.Slider.update(interactive=not a),
                        inputs=[follow_loopback_mix],
                        outputs=[wet_mix],
                    )

            increment_seed = gr.Checkbox(
                label='Increment seed in img2img batch tab',
                value=True,
                elem_id=format_elem_id('increment_seed'),
            )

        return [enable, increment_seed, loopback_mix, wet_mix, follow_loopback_mix]

    def process(self, p, enable, increment_seed, loopback_mix, wet_mix, follow_loopback_mix):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not enable: return

        if not self.is_img2img_batch:
            self.output_images.clear()

    def process_batch(self, p, enable, increment_seed, loopback_mix, wet_mix, follow_loopback_mix, **kwargs):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not enable: return

        if self.is_img2img_batch:
            if not self.output_images.size_locked: return

        else:
            if not self.output_images: return

            self.source_latent = p.init_latent

        if follow_loopback_mix:
            wet_mix = loopback_mix

        last_latent = self.__to_latent(p, self.output_images.get_current())
        feedback_latent = self.source_latent * (1. - wet_mix) + last_latent * wet_mix
        p.init_latent = p.init_latent * (1. - loopback_mix) + feedback_latent * loopback_mix

    def postprocess_batch(self, p, enable, increment_seed, loopback_mix, wet_mix, follow_loopback_mix, **kwargs):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not enable: return

        self.output_images.append(kwargs['images'])
        if not self.is_img2img_batch:
            self.output_images.lock_size()

    def postprocess(self, p, processed, enable, increment_seed, loopback_mix, wet_mix, follow_loopback_mix):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not enable: return

        if not self.is_img2img_batch:
            self.output_images.clear()

    def __to_latent(self, p, images):
        images = 2. * images.to(device=shared.device) - 1.
        with devices.autocast():
            return p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(images))

    def img2img_process_batch_tab(self, p):
        if not self.find_script_args(p)[0]: return
        self.is_img2img_batch = True
        self.output_images.clear()
        self.source_latent = None
        self.seed = p.seed
        self.subseed = p.subseed

    def img2img_process_batch_tab_each(self, p):
        script_args = self.find_script_args(p)
        if not script_args[0]: return

        self.source_latent = p.init_latent
        if script_args[1]:
            if self.seed != -1:
                self.seed += p.n_iter
            p.seed = self.seed
            if self.subseed != -1:
                self.subseed += p.n_iter
            p.subseed = self.subseed

    def img2img_postprocess_batch_tab_each(self, p):
        if not self.find_script_args(p)[0]: return
        self.output_images.lock_size()
        self.source_latent = None

    def img2img_postprocess_batch_tab(self, p):
        if not self.find_script_args(p)[0]: return
        self.is_img2img_batch = False
        self.output_images.clear()
        self.source_latent = None

    def find_script_args(self, p):
        if not p.scripts.alwayson_scripts: return [False, False]

        for script in p.scripts.alwayson_scripts:
            if script.title().lower() == 'batch loopback':
                break

        return p.script_args[script.args_from:script.args_to]
