import gradio as gr

import modules.scripts as scripts
from modules import script_callbacks

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
                elements = [enabled]
            with gr.Row():
                elements.extend(inputs_ui())
        return elements

    def postprocess_image(
            self, p, pp, enabled, *args):

        if not enabled:
            return

        pp.image = mosaic_process(pp.image, *args)

def on_ui_tabs():
    return (full_ui(), "Mozaikukun", "mozaikukun"),

script_callbacks.on_ui_tabs(on_ui_tabs)
