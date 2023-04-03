import numpy
import torch

from modules import scripts, processing, shared, devices
import gradio as gr
import torchvision

from modules import img2img


img2img_process_batch_tab_callbacks = []
img2img_process_batch_tab_each_callbacks = []
img2img_postprocess_batch_tab_each_callbacks = []
img2img_postprocess_batch_tab_callbacks = []


def img2img_process_batch_hijack(*args, **kwargs):
    for callback in img2img_process_batch_tab_callbacks:
        callback()

    res_was_none = False
    p = None
    def img2img_scripts_run_hijack(pp, *args):
        nonlocal res_was_none, p
        p = pp
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
        return getattr(img2img, '__batch_loopback_original_process_batch')(*args, **kwargs)
    finally:
        if res_was_none:
            for callback in img2img_postprocess_batch_tab_each_callbacks:
                callback(p)

        scripts.scripts_img2img.run = original_img2img_scripts_run
        for callback in img2img_postprocess_batch_tab_callbacks:
            callback()


if hasattr(img2img, '__batch_loopback_original_process_batch'):
    # reset in case extension was updated
    img2img.process_batch = getattr(img2img, '__batch_loopback_original_process_batch')

setattr(img2img, '__batch_loopback_original_process_batch', img2img.process_batch)
img2img.process_batch = img2img_process_batch_hijack


class BatchLoopbackScript(scripts.Script):
    def __init__(self):
        self.img2img_batch_index = 0
        self.is_img2img_batch = False
        self.output_images = []
        self.init_latent = None
        self.init_images = None
        self.grow_last_images = True
        self.last_output_index = 0

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
            self.last_output_index = 0

    def process_batch(self, p, loopback_mix, **kwargs):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not loopback_mix: return
        if self.is_img2img_batch:
            if self.img2img_batch_index <= 0: return

            last_output = self.output_images[self.last_output_index]

        else:
            if not self.output_images: return

            last_output = self.output_images[-1]
            self.init_latent = p.init_latent
            self.init_images = list(p.init_images)

        with devices.autocast():
            to_stack = []
            for image in last_output:
                image = numpy.array(image).astype(numpy.float32) / 255.0
                image = numpy.moveaxis(image, 2, 0)
                to_stack.append(image)

            last_latent = 2. * torch.stack(to_stack).to(shared.device) - 1.
            last_latent = shared.sd_model.get_first_stage_encoding(shared.sd_model.encode_first_stage(last_latent))
            p.init_latent = p.init_latent * (1.0 - loopback_mix) + last_latent * loopback_mix
        pass

    def postprocess_batch(self, p, loopback_mix, **kwargs):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not loopback_mix: return

        images = [torchvision.transforms.ToPILImage()(image) for image in kwargs['images']]
        if self.grow_last_images:
            self.output_images.append(images)
        else:
            self.output_images[self.last_output_index] = images

        self.last_output_index += 1

    def postprocess(self, p, processed, loopback_mix):
        if not self.is_img2img_batch:
            self.output_images.clear()
            self.last_output_index = 0

        self.seed = p.seed
        self.subseed = p.subseed

    def img2img_process_batch_tab(self):
        self.img2img_batch_index = 0
        self.is_img2img_batch = True
        self.output_images.clear()
        self.init_latent = None
        self.init_images = None
        self.grow_last_images = True

    def img2img_process_batch_tab_each(self, p):
        self.last_output_index = 0
        self.init_latent = p.init_latent
        self.init_images = p.init_images
        p.seed = self.seed
        p.subseed = self.subseed

    def img2img_postprocess_batch_tab_each(self, p):
        self.img2img_batch_index += 1
        self.grow_last_images = False
        self.init_latent = None
        self.init_images = None

    def img2img_postprocess_batch_tab(self):
        self.img2img_batch_index = 0
        self.is_img2img_batch = False
        self.output_images.clear()
        self.init_latent = None
        self.init_images = None
        self.grow_last_images = True
