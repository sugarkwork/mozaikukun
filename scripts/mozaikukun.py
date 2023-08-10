import os
from PIL import Image

import gradio as gr
import numpy as np
from ultralytics import YOLO

import modules.scripts as scripts
from modules.processing import process_images
from modules.shared import opts

from mozaikukun import process_and_analyze_image

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

class Mozaikukun(scripts.Script):
    object_detector = YOLO(os.path.join(THIS_DIR, "../", "yolov8x.pt"))
    segmenter = YOLO(os.path.join(THIS_DIR, "../", "myseg3.pt"))

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
                pussy = gr.Radio(label="pussy", choices=['raw', 'mosaic'], value='mosaic')
                penis = gr.Radio(label="penis", choices=['raw', 'mosaic'], value='mosaic')
                sex = gr.Radio(label="sex", choices=['raw', 'mosaic'], value='mosaic')
                anus = gr.Radio(label="anus", choices=['raw', 'mosaic'], value='raw')
                nipples = gr.Radio(label="nipples", choices=['raw', 'mosaic'], value='raw')
        return [enabled, pussy, penis, sex, anus, nipples]
    
    def mosaic_process(self, input_img, pussy, penis, sex, anus, nipple):
        img = Image.fromarray(np.uint8(input_img))
        img = img.convert("RGBA")

        process_mode = {
            "pussy": pussy,
            "penis": penis,
            "sex": sex,
            "anus": anus,
            "nipple": nipple,
        }

        result = process_and_analyze_image(img, self.object_detector, self.segmenter)
        for key in result.keys():
            if key not in process_mode:
                continue

            if process_mode.get(key) == "mosaic":
                for mosaic_img in result[key]:
                    mosaic_img = mosaic_img.convert("RGBA")
                    img = Image.alpha_composite(img, mosaic_img)

        return img

    def postprocess_image(
            self, p, pp, enabled, pussy, penis, sex, anus, nipples):

        if not enabled:
            return

        pp.image = self.mosaic_process(pp.image, pussy, penis, sex, anus, nipples)
