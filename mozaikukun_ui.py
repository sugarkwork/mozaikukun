import os
import numpy as np
import gradio as gr
from PIL import Image

import mozaikukun as moza
from ultralytics import YOLO

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

object_detector = YOLO("yolov8x.pt")
segmenter = YOLO(os.path.normpath(os.path.join(THIS_DIR, "myseg7.pt")))


def mosaic_process(input_img, pussy, penis, sex, anus, nipple, margin, blur_radius, rate):
    img = Image.fromarray(np.uint8(input_img))
    img = img.convert("RGBA")

    process_mode = {
        "pussy": pussy,
        "penis": penis,
        "sex": sex,
        "anus": anus,
        "nipple": nipple,
    }

    result = moza.process_and_analyze_image(img, object_detector, segmenter)
    for key in result.keys():
        if key not in process_mode:
            continue

        for detect_obj in result[key]:
            process_image = detect_obj.get_image(
                process_mode.get(key, "mosaic"),
                {"margin": margin, "blur_radius": blur_radius, "rate": float(rate)/100.0})
            if process_image is None:
                continue
            img = Image.alpha_composite(img, process_image.convert("RGBA"))

    return img

def inputs_ui():
    with gr.Column():
        pussy = gr.Dropdown(label="pussy", value="mosaic", choices=['raw', 'mosaic', 'white'])
        penis = gr.Dropdown(label="penis", value="mosaic", choices=['raw', 'mosaic', 'white'])
        sex = gr.Dropdown(label="sex", value="mosaic", choices=['raw', 'mosaic', 'white'])
        anus = gr.Dropdown(label="anus", value="raw", choices=['raw', 'mosaic', 'white'])
        nipples = gr.Dropdown(label="nipples", value="raw", choices=['raw', 'mosaic', 'white'])
    with gr.Column():
        margin = gr.Slider(label="margin", value=5)
        blur_radius = gr.Slider(label="blur_radius (white)", value=5)
        rate = gr.Slider(label="rate (mosaic)", value=100, minimum=40, maximum=200)
    return [pussy, penis, sex, anus, nipples, margin, blur_radius, rate]

def full_ui():
    with gr.Blocks() as ui:
        with gr.Row():
            img = gr.Image()
            inputs = [img]
            inputs.extend(inputs_ui())
        with gr.Row():
            btn = gr.Button("convert")
        with gr.Row():
            result = gr.Image()
        btn.click(mosaic_process, inputs=inputs, outputs=[result])
    return ui
