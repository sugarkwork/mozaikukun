from PIL import Image
from pytoshop import layers
from pytoshop.enums import BlendMode
from pytoshop.core import PsdFile
import numpy as np


# If you do not install pytoshop after installing Cyphon, it will not work properly.
# ex:
#   pip install cython
#   pip install pytoshop pillow numpy


class PsdLayer:
    def __init__(self, image: Image, name: str, blending_mode: BlendMode = BlendMode.normal):
        self.image = image.convert("RGBA")
        self.image_data = np.array(self.image)
        self.name = name
        self.blending_mode = blending_mode


class Psd:
    def __init__(self):
        self.psd_file = None

    def add_image(self, image: Image, name: str, blending_mode: BlendMode = BlendMode.normal) -> PsdLayer:
        layer = PsdLayer(image, name, blending_mode)
        self.add_layer(layer)
        return layer

    def add_layer(self, layer: PsdLayer):
        if self.psd_file is None:
            self.psd_file = PsdFile(num_channels=3, height=layer.image_data.shape[0], width=layer.image_data.shape[1])

        channel_data = [layers.ChannelImageData(image=layer.image_data[:, :, i], compression=1) for i in range(4)]

        layer_record = layers.LayerRecord(
            channels={-1: channel_data[3], 0: channel_data[0], 1: channel_data[1], 2: channel_data[2]},
            top=0, bottom=layer.image_data.shape[0], left=0, right=layer.image_data.shape[1],
            blend_mode=layer.blending_mode,
            name=layer.name,
            opacity=255,
        )

        self.psd_file.layer_and_mask_info.layer_info.layer_records.append(layer_record)

    def save(self, output_path: str):
        with open(output_path, 'wb') as output_file:
            self.psd_file.write(output_file)


def main():
    psd = Psd()
    psd.add_image(Image.open("input/ai1.jpg"), "base_layer")
    psd.add_image(Image.open("output/ai1_pussy_0.png"), "pussy_0")
    psd.add_image(Image.open("output/ai1_pussy_1.png"), "pussy_1")
    psd.save("test.psd")


if __name__ == '__main__':
    main()
