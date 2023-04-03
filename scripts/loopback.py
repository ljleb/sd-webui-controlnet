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
        self.init_latent = None
        self.increment_img2img_batch_seed = (False, False)
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
            return [gr.State(0.), gr.State(1.), gr.State(False)]

        self.increment_img2img_batch_seed = (True, self.increment_img2img_batch_seed[1])

        extension_name = 'batch_loopback'
        with gr.Accordion(label='Batch Loopback', open=False, elem_id=extension_name):
            with gr.Row():
                with gr.Column(scale=3):
                    loopback_mix = gr.Slider(
                        label='Loopback mix',
                        value=0.,
                        minimum=0.,
                        maximum=1.,
                        step=.01,
                        elem_id=f'{extension_name}_mix',
                    )

                with gr.Column(min_width=160):
                    wet_mix = gr.Slider(
                        label='Wet mix ', # extra space fixes broken maximum for some reason -_-
                        value=1.,
                        minimum=0.,
                        maximum=1.,
                        step=.01,
                        elem_id=f'{extension_name}_wet_mix',
                    )

                    follow_loopback_mix = gr.Checkbox(
                        label='Follow loopback mix',
                        value=False,
                        elem_id=f'{extension_name}_follow_loopback_mix',
                    )
                    follow_loopback_mix.change(
                        fn=lambda a: gr.Slider.update(interactive=not a),
                        inputs=[follow_loopback_mix],
                        outputs=[wet_mix],
                    )

            def block():
                increment_seed = gr.Checkbox(
                    label='Increment seed in img2img batch tab',
                    value=self.increment_img2img_batch_seed[0],
                    elem_id=f'{extension_name}_increment_seed',
                )
                increment_seed.change(fn=self.__update_increment_seed, inputs=increment_seed)
            block()

        return [loopback_mix, wet_mix, follow_loopback_mix]

    def __update_increment_seed(self, new_value):
        self.increment_img2img_batch_seed = (new_value, self.increment_img2img_batch_seed[1])

    def process(self, p, loopback_mix, wet_mix, follow_loopback_mix):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not loopback_mix: return

        if not self.is_img2img_batch:
            self.output_images.clear()

    def process_batch(self, p, loopback_mix, wet_mix, follow_loopback_mix, **kwargs):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not loopback_mix: return
        if self.is_img2img_batch:
            if not self.output_images.size_locked: return

        else:
            if not self.output_images: return

            self.init_latent = p.init_latent

        if follow_loopback_mix:
            wet_mix = loopback_mix

        last_latent = self.__to_latent(p, self.output_images.get_current())
        feedback_latent = self.init_latent * (1. - wet_mix) + last_latent * wet_mix
        p.init_latent = p.init_latent * (1. - loopback_mix) + feedback_latent * loopback_mix

    def postprocess_batch(self, p, loopback_mix, wet_mix, follow_loopback_mix, **kwargs):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not loopback_mix: return

        images = [torchvision.transforms.ToPILImage()(image) for image in kwargs['images']]
        self.output_images.append(images)
        if not self.is_img2img_batch:
            self.output_images.lock_size()
            p.init_latent = self.init_latent

    def postprocess(self, p, processed, loopback_mix, wet_mix, follow_loopback_mix):
        if not self.is_img2img_batch:
            self.output_images.clear()

    def __to_latent(self, p, images):
        to_stack = []
        for image in images:
            image = numpy.array(image).astype(numpy.float32) / 255.0
            image = numpy.moveaxis(image, 2, 0)
            to_stack.append(image)

        stacked_images = 2. * torch.from_numpy(numpy.array(to_stack)).to(shared.device) - 1.
        with devices.autocast():
            return p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(stacked_images))

    def img2img_process_batch_tab(self, p):
        self.is_img2img_batch = True
        self.output_images.clear()
        self.init_latent = None
        self.increment_img2img_batch_seed = (self.increment_img2img_batch_seed[0], self.increment_img2img_batch_seed[0])
        self.seed = p.seed
        self.subseed = p.subseed

    def img2img_process_batch_tab_each(self, p):
        self.init_latent = p.init_latent
        if self.increment_img2img_batch_seed[1]:
            if self.seed != -1:
                self.seed += p.n_iter
            p.seed = self.seed
            if self.subseed != -1:
                self.subseed += p.n_iter
            p.subseed = self.subseed

    def img2img_postprocess_batch_tab_each(self, p):
        self.output_images.lock_size()
        self.init_latent = None

    def img2img_postprocess_batch_tab(self, p):
        self.is_img2img_batch = False
        self.output_images.clear()
        self.init_latent = None
