import os
from PIL import Image
from pathlib import Path
from collections import OrderedDict

import gradio as gr
import numpy as np
from ultralytics import YOLO

import modules.scripts as scripts

from mozaikukun import process_and_analyze_image

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

def get_segmenter_models():
    model_dir = Path(os.path.join(THIS_DIR, "../"))
    model_paths = [
        p
        for p in model_dir.rglob("*")
        if p.is_file() and p.suffix in (".pt", ".pth") and "myseg" in p.name
    ]
    models = OrderedDict()
    for path in model_paths:
        if path.name in models:
            continue
        models[path.name] = str(path)
    return models

DEFAULT_SEGMENTER = "myseg7.pt"

class Mozaikukun(scripts.Script):
    object_detector = YOLO(os.path.join(THIS_DIR, "../", "yolov8x.pt"))
    segmenter_models = get_segmenter_models()

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
            with gr.Row():
                segmenter_names = list(self.segmenter_models.keys())
                segmenter_name = gr.Dropdown(
                    label="Mozaikukun model",
                    choices=segmenter_names,
                    value=DEFAULT_SEGMENTER if DEFAULT_SEGMENTER in segmenter_names else segmenter_names[0],
                    visible=True,
                    type="value"
                )
            with gr.Row():
                pussy = gr.Radio(label="pussy", choices=['raw', 'mosaic'], value='mosaic')
                penis = gr.Radio(label="penis", choices=['raw', 'mosaic'], value='mosaic')
                sex = gr.Radio(label="sex", choices=['raw', 'mosaic'], value='mosaic')
                anus = gr.Radio(label="anus", choices=['raw', 'mosaic'], value='raw')
                nipples = gr.Radio(label="nipples", choices=['raw', 'mosaic'], value='raw')
        return [enabled, segmenter_name, pussy, penis, sex, anus, nipples]
    
    def mosaic_process(self, input_img, segmenter_name, pussy, penis, sex, anus, nipple):
        img = Image.fromarray(np.uint8(input_img))
        img = img.convert("RGBA")

        process_mode = {
            "pussy": pussy,
            "penis": penis,
            "sex": sex,
            "anus": anus,
            "nipple": nipple,
        }

        result = process_and_analyze_image(img, self.object_detector, YOLO(self.segmenter_models[segmenter_name]))
        for key in result.keys():
            if key not in process_mode:
                continue

            if process_mode.get(key) == "mosaic":
                for mosaic_img in result[key]:
                    mosaic_img = mosaic_img.convert("RGBA")
                    img = Image.alpha_composite(img, mosaic_img)

        return img

    def postprocess_image(
            self, p, pp, enabled, segmenter_name, pussy, penis, sex, anus, nipples):

        if not enabled:
            return

        pp.image = self.mosaic_process(pp.image, segmenter_name, pussy, penis, sex, anus, nipples)
