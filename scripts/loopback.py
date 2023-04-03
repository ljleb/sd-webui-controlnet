import dataclasses
from typing import List

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


@dataclasses.dataclass
class GrowingCircularBuffer:
    buffer: List[List[Image.Image]] = dataclasses.field(default_factory=list)
    current_index: int = 0
    size_locked: bool = False

    def lock_size(self, lock=True):
        self.size_locked = lock

    def append(self, value):
        if self.size_locked:
            self.buffer[self.current_index % len(self.buffer)] = value
        else:
            self.buffer.append(value)

        self.current_index += 1

    def clear(self):
        for k, v in GrowingCircularBuffer().__dict__.items():
            setattr(self, k, v)

    def get_current(self):
        return self.buffer[(self.current_index - 1) % len(self.buffer)]

    def __bool__(self):
        return bool(self.buffer)


class BatchLoopbackScript(scripts.Script):
    def __init__(self):
        self.is_img2img_batch = False
        self.output_images = GrowingCircularBuffer()
        self.init_latent = None
        self.init_images = None

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
            return [gr.State(0.0)]

        with gr.Accordion(label='Batch Loopback', open=False, elem_id='batch_loopback'):
            loopback_mix = gr.Slider(
                label='Loopback mix',
                value=0.0,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                elem_id='batch_loopback_mix')

        return [loopback_mix]

    def process(self, p, loopback_mix):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not loopback_mix: return

        if not self.is_img2img_batch:
            self.output_images.clear()

    def process_batch(self, p, loopback_mix, **kwargs):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not loopback_mix: return
        if self.is_img2img_batch:
            if not self.output_images.size_locked: return

        else:
            if not self.output_images: return

            self.init_latent = p.init_latent
            self.init_images = list(p.init_images)

        last_output = self.output_images.get_current()

        with devices.autocast():
            to_stack = []
            for image in last_output:
                image = numpy.array(image).astype(numpy.float32) / 255.0
                image = numpy.moveaxis(image, 2, 0)
                to_stack.append(image)

            last_latent = 2. * torch.from_numpy(numpy.array(to_stack)).to(shared.device) - 1.
            last_latent = p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(last_latent))
            p.init_latent = p.init_latent * (1.0 - loopback_mix) + last_latent * loopback_mix

    def postprocess_batch(self, p, loopback_mix, **kwargs):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not loopback_mix: return

        images = [torchvision.transforms.ToPILImage()(image) for image in kwargs['images']]
        self.output_images.append(images)
        if not self.is_img2img_batch:
            self.output_images.lock_size()

    def postprocess(self, p, processed, loopback_mix):
        if not self.is_img2img_batch:
            self.output_images.clear()

    def img2img_process_batch_tab(self, p):
        self.is_img2img_batch = True
        self.output_images.clear()
        self.init_latent = None
        self.init_images = None
        self.seed = p.seed
        self.subseed = p.subseed

    def img2img_process_batch_tab_each(self, p):
        self.init_latent = p.init_latent
        self.init_images = p.init_images
        if self.seed != -1:
            self.seed += p.n_iter
        p.seed = self.seed
        if self.subseed != -1:
            self.subseed += p.n_iter
        p.subseed = self.subseed

    def img2img_postprocess_batch_tab_each(self, p):
        self.output_images.lock_size()
        self.init_latent = None
        self.init_images = None

    def img2img_postprocess_batch_tab(self, p):
        self.is_img2img_batch = False
        self.output_images.clear()
        self.init_latent = None
        self.init_images = None
