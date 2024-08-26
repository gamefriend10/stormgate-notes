# For internal development

stormgate-notes-dev.exe will run with console so you can debug. You can also just run the script locally.

## Packaging Commands

### Pre-requisites:

Install tesseract-ocr: https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file#installing-tesseract
- "Install Tesseract via pre-built binary package"
- "Windows"
- "Tesseract at UB Mannheim"
- "tesseract-ocr-w64-setup-5.4.0.20240606.exe"

Then run:

```./package.sh```

Success!

(I've noticed windows defender flags Stormgate Notes as malware and while I can tell you I didn't include malware, you shouldn't trust me and should verify the code yourself.)

## What package.sh is doing

### For no terminal (only GUI):

```pyinstaller --onefile --windowed --add-data "C:\Program Files\Tesseract-OCR\tesseract.exe;." --add-data "C:\Program Files\Tesseract-OCR\tessdata;./tessdata" --name "stormgate-notes.exe" script.py```

### For terminal and GUI to show up:

```pyinstaller --onefile --add-data "C:\Program Files\Tesseract-OCR\tesseract.exe;." --add-data "C:\Program Files\Tesseract-OCR\tessdata;./tessdata" --name "stormgate-notes-dev.exe" script.py```

And then add in `faction_images` and `saved_input` folders on the same level as the executable.

So the file structure should look like:

```
faction_images
saved_input
stormgate-notes.exe
stormgate-notes-dev.exe
```
