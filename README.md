# mozaikukun

AI will automatically mosaic the obscene part of the obscene image.

## Usage (Windows)
* git clone https://github.com/sugarkwork/mozaikukun
* Run webui.bat

## Manual installation (Windows)
* git clone https://github.com/sugarkwork/mozaikukun
* cd mozaikukun
* python -m venv venv
* venv\scripts\activate
* pip install cython
* pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
* pip install https://download.lfd.uci.edu/pythonlibs/archived/pytoshop-1.2.1-cp310-cp310-win_amd64.whl
* pip install -r requirements.txt

## Run in console

* Create an "input" directory.
* Put the image you want to mosaic in the input directory.
* Run: python mozaikukun.py
* An output directory will be created and the layers of the mosaicked image will be output.

# Future development schedule
- Change mosaic size.
- Batch processing (directory or ZIP file)
- Hiding processing other than mosaic (white fill, Gaussian blur, etc.)
- Make it possible to decide whether to apply or not for each detected part.

## Stable diffusion webui
1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter https://github.com/sugarkwork/mozaikukun.git to "URL for extension's git repository".
4. Press "Install" button.
5. Stop webui once and please specify `--disable-safe-unpickle` commandline argument
