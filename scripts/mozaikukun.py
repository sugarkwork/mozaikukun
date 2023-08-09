import os
from PIL import Image

import gradio as gr
import numpy as np
from ultralytics import YOLO

import modules.scripts as scripts
from modules import images
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
        return True

    def ui(self, is_img2img):
        pussy = gr.Radio(label="pussy", choices=['raw', 'mosaic'], value='mosaic')
        penis = gr.Radio(label="penis", choices=['raw', 'mosaic'], value='mosaic')
        sex = gr.Radio(label="sex", choices=['raw', 'mosaic'], value='mosaic')
        anus = gr.Radio(label="anus", choices=['raw', 'mosaic'], value='raw')
        nipples = gr.Radio(label="nipples", choices=['raw', 'mosaic'], value='raw')
        return [pussy, penis, sex, anus, nipples]
    
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

    def run(self, p, pussy, penis, sex, anus, nipples):
        p.do_not_save_samples = True

        proc = process_images(p)

        for i in range(len(proc.images)):
            ret = self.mosaic_process(proc.images[i], pussy, penis, sex, anus, nipples)
            proc.images[i] = ret
            images.save_image(proc.images[i], p.outpath_samples, "", proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p)

        return proc
