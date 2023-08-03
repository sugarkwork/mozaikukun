import math
import os
from glob import glob
from multiprocessing import Process, Queue
import json
import time

from PIL.Image import Image
from ultralytics import YOLO
from PIL import Image, ImageDraw
from typing import Dict, Tuple

BLOCK_SIZE_RATIO = 100
MARGIN_FACTOR = 3
MARGIN_EXTRA = 20


def calculate_pixel_block_and_margin(image: Image.Image) -> Tuple[int, int]:
    """
    Calculate the pixel block size and margin based on image dimension.
    """
    block_size = math.ceil(max(image.width, image.height) / BLOCK_SIZE_RATIO)
    margin = block_size * MARGIN_FACTOR + MARGIN_EXTRA
    return block_size, margin


def resize_bounding_box(bounding_box: Dict[str, int], margin: int, block_size: int) -> Dict[str, int]:
    """
    Resize the bounding box by adding the margin.
    """
    bounding_box["x1"] = int((bounding_box["x1"] - margin) / block_size) * block_size
    bounding_box["x2"] = int(math.ceil((bounding_box["x2"] + margin) / block_size) * block_size)

    bounding_box["y1"] = int((bounding_box["y1"] - margin) / block_size) * block_size
    bounding_box["y2"] = int(math.ceil((bounding_box["y2"] + margin) / block_size) * block_size)

    return bounding_box


def create_transparent_mask(segment_box: Dict[str, int], segments: Dict[str, list]) -> tuple[Image, Image]:
    """
    Create a transparent mask for image segmentation.
    """
    empty_mask = Image.new("RGBA", (
        int(segment_box["x2"] - segment_box["x1"]), int(segment_box["y2"] - segment_box["y1"])), (0, 0, 0, 0))
    mask = Image.new('RGBA', (
        int(segment_box["x2"] - segment_box["x1"]), int(segment_box["y2"] - segment_box["y1"])), (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask)

    adjusted_x_segments = [math.ceil(segment_x) - segment_box["x1"] for segment_x in segments['x']]
    adjusted_y_segments = [math.ceil(segment_y) - segment_box["y1"] for segment_y in segments['y']]

    polygon_points = list(zip(adjusted_x_segments, adjusted_y_segments))
    mask_draw.polygon(polygon_points, fill=(255, 255, 255, 255))

    for i in range(len(polygon_points) - 1):
        mask_draw.line([polygon_points[i], polygon_points[i + 1]], fill=(255, 255, 255, 255), width=5)

    return mask, empty_mask


def apply_mask_on_image(cropped_region: Image.Image, segment_box: Dict[str, int], mask: Image.Image,
                        empty_mask: Image.Image) -> Image.Image:
    """
    Apply the created mask on the image.
    """
    segment_region = cropped_region.crop(
        (segment_box["x1"], segment_box["y1"], segment_box["x2"], segment_box["y2"]))
    empty_mask.paste(segment_region)

    masked_image = Image.new("RGBA", empty_mask.size)
    masked_image.paste(empty_mask, mask=mask)
    return masked_image


def generate_mosaic(masked_image: Image.Image, block_size: int) -> Image.Image:
    """
    Generate a mosaic image from the masked image and block size.
    """
    small_masked_image = masked_image.resize(
        (int(masked_image.size[0] // block_size), int(masked_image.size[1] // block_size)), Image.BILINEAR)
    mosaic_masked_image = small_masked_image.resize(masked_image.size, Image.NEAREST)
    return mosaic_masked_image


def image_detection_worker(process_id: int, input_queue: Queue, result_queue: Queue):
    print(f"start subprocess {process_id}")
    object_detector = YOLO("yolov8x.pt")
    segmenter = YOLO("myseg3.pt")
    while True:
        img, name = input_queue.get()
        print(f"{process_id}: get {name}")
        start_time = time.time()
        result_queue.put((process_and_analyze_image(img, object_detector, segmenter), name))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{process_id}: put ({elapsed_time} sec) {name}")


def process_and_analyze_image(image: Image.Image, object_detector: YOLO, segmenter: YOLO) -> Dict:
    """
    Process a single image, analyze it and save the result.
    """
    original_image = image.convert("RGBA")

    block_size, margin = calculate_pixel_block_and_margin(original_image)

    detection_results = object_detector(original_image, save=False, device='1', project="yolov8x", name="pname1",
                                        verbose=False)

    result = {
        "penis": [],
        "sex": [],
        "pussy": []
    }

    for detection in detection_results:
        for detected_object in json.loads(detection.tojson()):
            if detected_object["name"] != "person":
                continue
            bounding_box = resize_bounding_box(detected_object["box"], margin, block_size)

            cropped_region = original_image.crop(
                (bounding_box["x1"], bounding_box["y1"], bounding_box["x2"], bounding_box["y2"]))

            segmentation_results = segmenter(cropped_region, save=False, device='1', project="myseg2", name="pname2",
                                             verbose=False)
            for segmentation in segmentation_results:
                for segmented_object in json.loads(segmentation.tojson()):
                    name = segmented_object["name"]
                    if name != "pussy" and name != "sex" and name != "penis":
                        continue
                    segments = segmented_object["segments"]
                    segment_box = resize_bounding_box(segmented_object["box"], margin, block_size)

                    if len(segments.get("x", [])) <= 2:
                        continue
                    if len(segments.get("y", [])) <= 2:
                        continue

                    mask, empty_mask = create_transparent_mask(segment_box, segments)
                    masked_image = apply_mask_on_image(cropped_region, segment_box, mask, empty_mask)
                    mosaic_masked_image = generate_mosaic(masked_image, block_size)

                    final_image = Image.new('RGBA', original_image.size)
                    mosaic_position = (
                        int(bounding_box["x1"] + segment_box["x1"]), int(bounding_box["y1"] + segment_box["y1"]))
                    final_image.paste(mosaic_masked_image, mosaic_position)
                    result[name].append(increase_image_opacity(final_image))

    return result


def increase_image_opacity(img: Image.Image, rate: int = 10) -> Image.Image:
    img = img.convert("RGBA")  # ensure image has alpha channel

    datas = img.getdata()
    new_data = []
    for item in datas:
        if item[3] != 0:
            new_data.append((item[0], item[1], item[2], min(255, int(item[3] * rate))))
        else:
            new_data.append(item)  # leave fully transparent pixel as it is
    img.putdata(new_data)
    return img


def main():
    print("Start processing...")
    worker_count = 3

    worker_processes = []
    task_queue = Queue()
    result_queue = Queue()

    for worker_number in range(worker_count):
        p = Process(target=image_detection_worker, args=(worker_number, task_queue, result_queue))
        p.start()
        worker_processes.append(p)

    output_dir = "output"

    os.makedirs(output_dir, exist_ok=True)

    for image_path in glob("input/*.jpg"):
        task_queue.put((Image.open(image_path), image_path))

    sensitive_areas = ["pussy", "penis", "sex"]

    while True:
        result_data, image_name = result_queue.get()
        for sensitive_area in sensitive_areas:
            image_number = 0
            for result_image in result_data[sensitive_area]:
                new_image_filename = (f"{os.path.splitext(os.path.basename(image_name))[0]}_"
                                      f"{sensitive_area}_{image_number}.png")
                image_save_path = os.path.join(output_dir, new_image_filename)
                result_image.save(image_save_path)
                image_number += 1


if __name__ == '__main__':
    main()
