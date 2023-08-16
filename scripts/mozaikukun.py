import gradio as gr

import modules.scripts as scripts
from modules import script_callbacks, images
from modules.shared import opts
from modules.processing import create_infotext

from mozaikukun_ui import full_ui, inputs_ui, mosaic_process

class Mozaikukun(scripts.Script):

    def title(self):
        return "Mozaikukun"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Mozaikukun", open=False):
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enabled",
                    value=False,
                    visible=True,
                )
                save_original = gr.Checkbox(
                    label="Save original image",
                    value=False,
                    visible=True,
                )
                elements = [enabled, save_original]
            with gr.Row():
                elements.extend(inputs_ui())
        return elements

    def postprocess_image(
            self, p, pp, enabled, save_original, *args):

        if not enabled:
            return

        def infotext(index=0, use_main_prompt=False):
            return create_infotext(p, p.prompts, p.seeds, p.subseeds, use_main_prompt=use_main_prompt, index=index, all_negative_prompts=p.negative_prompts)

        if save_original:
            images.save_image(pp.image, p.outpath_samples, "", p.seeds[p.batch_index], p.prompts[p.batch_index], opts.samples_format, info=infotext(p.batch_index), p=p)

        pp.image = mosaic_process(pp.image, *args)

def on_ui_tabs():
    return (full_ui(), "Mozaikukun", "mozaikukun"),

script_callbacks.on_ui_tabs(on_ui_tabs)
