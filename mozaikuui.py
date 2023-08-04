import numpy as np
import gradio as gr
from PIL import Image

import mozaikukun as moza
from ultralytics import YOLO

object_detector = YOLO("yolov8x.pt")
segmenter = YOLO("myseg3.pt")


def mosaic_process(input_img, pussy, penis, sex, anus, nipple):
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

        if process_mode.get(key) == "mosaic":
            for mosaic_img in result[key]:
                mosaic_img = mosaic_img.convert("RGBA")
                mosaic_img.save("hoge.png")
                img = Image.alpha_composite(img, mosaic_img)

    return img


demo = gr.Interface(
    fn=mosaic_process,
    inputs=[
        gr.Image(),
        gr.Radio(choices=['raw', 'mosaic'], value='mosaic'),
        gr.Radio(choices=['raw', 'mosaic'], value='mosaic'),
        gr.Radio(choices=['raw', 'mosaic'], value='mosaic'),
        gr.Radio(choices=['raw', 'mosaic'], value='raw'),
        gr.Radio(choices=['raw', 'mosaic'], value='raw'),
    ],
    outputs=["image"])

demo.launch()
