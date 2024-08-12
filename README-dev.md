# For internal development

stormgate-notes-dev.exe will run with console so you can debug. You can also just run the script locally.

## Packaging Commands

For no terminal (only GUI):

```pyinstaller --onefile --windowed --add-data "C:\Program Files\Tesseract-OCR\tesseract.exe;." --add-data "C:\Program Files\Tesseract-OCR\tessdata;./tessdata" --name "stormgate-notes.exe" script.py```

For terminal and GUI to show up:

```pyinstaller --onefile --add-data "C:\Program Files\Tesseract-OCR\tesseract.exe;." --add-data "C:\Program Files\Tesseract-OCR\tessdata;./tessdata" --name "stormgate-notes-dev.exe" script.py```

And then add in `faction_images` and `saved_input` folders on the same level as the executable.

So the file structure should look like:

```
faction_images
saved_input
stormgate-notes.exe
stormgate-notes-dev.exe
```
