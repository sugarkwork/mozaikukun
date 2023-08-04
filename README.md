# mozaikukun

AI will automatically mosaic the obscene part of the obscene image.

## Usage (Windows)
* git clone https://github.com/sugarkwork/mozaikukun
* Run webui.bat

## Manual installation (Windows)
* pip3 install pillow
* pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
* pip3 install ultralytics
* pip3 install gradio

## Run in console

* Create an input directory.
* Put the image you want to mosaic in the input directory.
* Run: python mozaikukun.py
* An output directory will be created and the layers of the mosaicked image will be output.
