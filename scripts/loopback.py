from PIL import Image

from modules import scripts, script_callbacks, processing, shared, devices
import gradio as gr
import torchvision
import numpy


img2img_batch_loopback = gr.Checkbox(
    label='Batch loopback',
    value=False,
    elem_id='controlnet_img2img_batch_loopback')


class BatchLoopbackScript(scripts.Script):
    def title(self):
        return 'Batch Loopback'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        global img2img_batch_loopback
        local_img2img_batch_loopback = gr.Checkbox(visible=False)
        img2img_batch_loopback.change(
            fn=lambda x: x,
            inputs=[img2img_batch_loopback],
            outputs=[local_img2img_batch_loopback])
        return [local_img2img_batch_loopback]

    def postprocess_batch(self, p: processing.StableDiffusionProcessing, img2img_batch_loopback, **kwargs):
        if not isinstance(p, processing.StableDiffusionProcessingImg2Img): return
        if not img2img_batch_loopback: return

        image = torchvision.transforms.ToPILImage()(kwargs['images'][0])
        p.init_images.clear()
        p.init_images.append(image)
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)


def on_after_component(component, **_kwargs):
    global img2img_batch_loopback
    if getattr(component, 'elem_id', None) == 'img2img_tiling':
        img2img_batch_loopback.render()
        return

    # use existing checkbox if/when webui implements it
    if getattr(component, 'img2img_batch_loopback', None) == 'img2img_batch_loopback':
        img2img_batch_loopback.unrender()
        img2img_batch_loopback = component
        return


script_callbacks.on_after_component(on_after_component)
