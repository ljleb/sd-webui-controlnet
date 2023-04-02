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
    def img2img_scripts_run_hijack(*args, **kwargs):
        nonlocal res_was_none
        if res_was_none:
            for callback in img2img_postprocess_batch_tab_each_callbacks:
                callback()

        for callback in img2img_process_batch_tab_each_callbacks:
            callback()

        res = original_img2img_scripts_run(*args, **kwargs)
        if res is not None:
            for callback in img2img_postprocess_batch_tab_each_callbacks:
                callback()
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
                callback()

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
        self.last_images = []
        self.grow_last_images = True
        self.current_image_index = 0

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
            self.last_images.clear()
            self.current_image_index = 0

    def process_batch(self, p, loopback_mix, **kwargs):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not loopback_mix: return
        if self.is_img2img_batch:
            if self.img2img_batch_index == 0: return
        else:
            if not self.last_images: return

        last_batch = self.last_images[self.current_image_index if self.is_img2img_batch else self.current_image_index - 1]
        current_latent = p.init_latent
        current_images = list(p.init_images)

        p.init_images.clear()
        p.init_images.extend(last_batch)
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        p.init_images.clear()
        p.init_images.extend(current_images)

        p.init_latent = p.init_latent * loopback_mix + current_latent * (1.0 - loopback_mix)
        pass

    def postprocess_batch(self, p, loopback_mix, **kwargs):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not loopback_mix: return

        if not self.is_img2img_batch:
            if shared.state.job_no == shared.state.job_count - 1:
                self.last_images.clear()
                self.current_image_index = 0
                return

        images = [torchvision.transforms.ToPILImage()(image) for image in kwargs['images']]
        if self.grow_last_images:
            self.last_images.append(images)
        else:
            self.last_images[self.current_image_index] = images

        self.current_image_index += 1

    def img2img_process_batch_tab(self):
        self.img2img_batch_index = 0
        self.is_img2img_batch = True
        self.last_images.clear()
        self.grow_last_images = True

    def img2img_process_batch_tab_each(self):
        self.current_image_index = 0

    def img2img_postprocess_batch_tab_each(self):
        self.img2img_batch_index += 1
        self.grow_last_images = False

    def img2img_postprocess_batch_tab(self):
        self.img2img_batch_index = 0
        self.is_img2img_batch = False
        self.last_images.clear()
        self.grow_last_images = True
