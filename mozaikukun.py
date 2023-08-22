import math
import os
from glob import glob
from multiprocessing import Process, Queue
import json
import time
from _queue import Empty
from PIL.Image import Image
from PIL import ImageFilter
from ultralytics import YOLO
from PIL import Image, ImageDraw
from typing import Dict, Tuple

try:
    # Ignore import errors as we only use the console version at this time.
    import psd_save
except:
    pass


BLOCK_SIZE_RATIO = 100
MARGIN_FACTOR = 3
MARGIN_EXTRA = 20
DEVICE = 'cpu'  # Specify the device number when using CUDA. Example(CUDA): DEVICE='0' Example(cpu): DEVICE='cpu'


class DetectedObject:
    def __init__(self, block_size: int, bounding_box: dict, cropped_region: Image.Image, margin: int,
                 original_image: Image.Image, segments: dict[str, list], segment_box: dict[str, int]):
        self.block_size = block_size
        self.bounding_box = bounding_box
        self.cropped_region = cropped_region
        self.margin = margin
        self.original_image = original_image
        self.segments = segments
        self.segment_box = segment_box

    def get_image(self, process_type="mosaic", options: dict = None) -> Image.Image:
        supported_types = {
            "raw": self.get_raw,
            "mosaic": self.get_mosaic,
            "white": self.get_white,
            "star": self.get_star,
            "heart": self.get_heart,
            "blur": self.get_blur,
        }
        if process_type not in supported_types:
            process_type = "mosaic"

        return supported_types[process_type](options)

    def get_star(self, _: dict = None) -> None:
        return None

    def get_heart(self, _: dict = None) -> None:
        return None

    def get_blur(self, _: dict = None) -> None:
        return None

    def get_raw(self, _: dict = None) -> None:
        return None

    def get_mosaic(self, options: dict = None) -> Image.Image:
        if options is None:
            options = {}
        rate = min(max(float(options.get("rate", 1.0)), 0.0), 1.0)
        margin = min(max(int(options.get("margin", 5)), 0), 1024)
        increase_level = min(max(int(options.get("increase_level", 10)), 0), 1024)

        mask, empty_mask = self.create_transparent_mask(self.segment_box, self.segments, margin)
        masked_image = self.apply_mask_on_image(self.cropped_region, self.segment_box, mask, empty_mask)
        mosaic_masked_image = self.generate_mosaic(masked_image, int(self.block_size * rate))
        final_image = Image.new('RGBA', self.original_image.size)
        mosaic_position = (
            int(self.bounding_box["x1"] + self.segment_box["x1"]),
            int(self.bounding_box["y1"] + self.segment_box["y1"]))
        final_image.paste(mosaic_masked_image, mosaic_position)
        return self.increase_image_opacity(final_image, increase_level)

    def get_white(self, options: dict = None) -> Image.Image:
        if options is None:
            options = {}
        blur_radius = min(max(float(options.get("blur_radius", 5)), 0.0), 100)
        margin = min(max(int(options.get("margin", 5)), 0), 1024)

        # draw
        mask, _ = self.create_transparent_mask(self.segment_box, self.segments, margin)
        final_image = Image.new('RGBA', self.original_image.size)
        mosaic_position = (
            int(self.bounding_box["x1"] + self.segment_box["x1"]),
            int(self.bounding_box["y1"] + self.segment_box["y1"]))
        final_image.paste(mask, mosaic_position)

        # color change
        r, g, b, a = final_image.split()
        white_r = Image.new('L', r.size, 255)
        white_g = Image.new('L', g.size, 255)
        white_b = Image.new('L', b.size, 255)
        final_image = Image.merge('RGBA', (white_r, white_g, white_b, a))

        if blur_radius == 0:
            return final_image

        return self.filter_blur(final_image, blur_radius)

    def filter_blur(self, image: Image.Image, blur_radius: float):
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        blurred_rgb_image = rgb_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred_alpha_channel = a.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        final_image = Image.merge('RGBA', (
            blurred_rgb_image.getchannel('R'), blurred_rgb_image.getchannel('G'), blurred_rgb_image.getchannel('B'),
            blurred_alpha_channel))

        return final_image

    def create_transparent_mask(self, segment_box: Dict[str, int], segments: Dict[str, list],
                                margin: int = 5) -> tuple[Image, Image]:
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

        if margin > 0:
            mask_draw.line(polygon_points, fill=(255, 255, 255, 255), width=margin, joint='curve')

        return mask, empty_mask

    def apply_mask_on_image(self, cropped_region: Image.Image, segment_box: Dict[str, int], mask: Image.Image,
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

    def generate_mosaic(self, masked_image: Image.Image, block_size: int) -> Image.Image:
        """
        Generate a mosaic image from the masked image and block size.
        """
        small_masked_image = masked_image.resize(
            (int(masked_image.size[0] // block_size), int(masked_image.size[1] // block_size)), Image.BILINEAR)
        mosaic_masked_image = small_masked_image.resize(masked_image.size, Image.NEAREST)
        return mosaic_masked_image

    def increase_image_opacity(self, img: Image.Image, rate: int = 10) -> Image.Image:
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


def resize_bounding_box(bounding_box: Dict[str, int], margin: int, block_size: int) -> Dict[str, int]:
    """
    Resize the bounding box by adding the margin.
    """
    bounding_box["x1"] = int((bounding_box["x1"] - margin) / block_size) * block_size
    bounding_box["x2"] = int(math.ceil((bounding_box["x2"] + margin) / block_size) * block_size)

    bounding_box["y1"] = int((bounding_box["y1"] - margin) / block_size) * block_size
    bounding_box["y2"] = int(math.ceil((bounding_box["y2"] + margin) / block_size) * block_size)

    return bounding_box


def calculate_pixel_block_and_margin(image: Image.Image) -> Tuple[int, int]:
    """
    Calculate the pixel block size and margin based on image dimension.
    """
    block_size = math.ceil(max(image.width, image.height) / BLOCK_SIZE_RATIO)
    margin = block_size * MARGIN_FACTOR + MARGIN_EXTRA
    return block_size, margin


def image_detection_worker(process_id: int, input_queue: Queue, result_queue: Queue):
    print(f"start subprocess {process_id}")
    object_detector = YOLO("yolov8x.pt")
    segmenter = YOLO("myseg9.pt")
    while True:
        img, name, options = input_queue.get()
        if img == None and name == None and options == None:
            break
        print(f"{process_id}: get {name}")
        start_time = time.time()
        result_queue.put((process_and_analyze_image(img, object_detector, segmenter), name))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{process_id}: put ({elapsed_time} sec) {name}")

    print(f"exit subprocess {process_id}")


def process_and_analyze_image(image: Image.Image, object_detector: YOLO, segmenter: YOLO) -> Dict[str, list]:
    """
    Process a single image, analyze it and save the result.
    """
    original_image = image.convert("RGBA")

    block_size, margin = calculate_pixel_block_and_margin(original_image)

    detection_results = object_detector(original_image, save=False, device=DEVICE, project="yolov8x", name="pname1",
                                        verbose=False)

    result = {}

    for detection in detection_results:
        for detected_object in json.loads(detection.tojson()):
            if detected_object["name"] != "person":
                continue
            bounding_box = resize_bounding_box(detected_object["box"], margin, block_size)

            cropped_region = original_image.crop(
                (bounding_box["x1"], bounding_box["y1"], bounding_box["x2"], bounding_box["y2"]))

            segmentation_results = segmenter(cropped_region, save=False, device=DEVICE, project="myseg2", name="pname2",
                                             verbose=False)
            for segmentation in segmentation_results:
                for segmented_object in json.loads(segmentation.tojson()):
                    name = segmented_object["name"]
                    segments = segmented_object["segments"]
                    segment_box = resize_bounding_box(segmented_object["box"], margin, block_size)

                    if len(segments.get("x", [])) <= 2:
                        continue
                    if len(segments.get("y", [])) <= 2:
                        continue

                    detected = DetectedObject(block_size, bounding_box, cropped_region, margin, original_image,
                                              segments, segment_box)
                    if name not in result:
                        result[name] = []

                    result[name].append(detected)

    return result


def main():
    print("Start processing...")
    worker_count = 3

    worker_processes = []
    task_queue = Queue()
    result_queue = Queue()

    for worker_number in range(worker_count):
        p = Process(
            target=image_detection_worker,
            args=(worker_number, task_queue, result_queue))
        p.start()
        worker_processes.append(p)

    # dir
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    images = {}

    # task queue
    for image_path in glob("input/*.jpg"):
        images[image_path] = Image.open(image_path)
        task_queue.put((images[image_path], image_path, {}))

    # exit command
    for _ in range(worker_count):
        task_queue.put((None, None, None))

    sensitive_areas = [
        "pussy",
        "penis",
        "sex"
    ]

    options = {
        "margin": 5,          # Extend the detected range (px)
        "rate": 1.0,          # Change the percentage of the mosaic size (1/100 of the long side of the image).
        "increase_level": 5,  # The percentage to make transparent parts opaque.
        "blur_radius": 10     # Specify the blur size.
    }

    while not result_queue.empty() or any(p.is_alive() for p in worker_processes):
        try:
            result_data, image_name = result_queue.get(timeout=1)
        except Empty:
            continue

        input_img = images.get(image_name)
        psd = psd_save.Psd()
        psd.add_image(input_img, os.path.splitext(os.path.basename(image_name))[0])

        for sensitive_area in sensitive_areas:
            image_number = 0
            if sensitive_area not in result_data:
                continue
            for result_image in result_data[sensitive_area]:
                layer_name = f"{sensitive_area} {image_number}"
                psd.add_image(result_image.get_image("mosaic", options=options), layer_name)
                image_number += 1

        psd.save(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_name))[0]}.psd"))

    for worker in worker_processes:
        worker.join()
        print(f"join subprocess : {worker}")


if __name__ == '__main__':
    main()
