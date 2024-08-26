#!/bin/bash

# Define the paths and names
TESSERACT_PATH="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
TESSDATA_PATH="C:\\Program Files\\Tesseract-OCR\\tessdata"
SCRIPT_NAME="script.py"
EXE_NAME="stormgate-notes.exe"
EXE_DEV_NAME="stormgate-notes-dev.exe"
FACTION_IMAGES_DIR="faction_images"
SAVED_INPUT_DIR="saved_input"
ZIP_NAME="stormgate-notes.zip"

# Create executables with and without terminal
pyinstaller --onefile --windowed --add-data "${TESSERACT_PATH};." --add-data "${TESSDATA_PATH};./tessdata" --name "${EXE_NAME}" $SCRIPT_NAME
pyinstaller --onefile --add-data "${TESSERACT_PATH};." --add-data "${TESSDATA_PATH};./tessdata" --name "${EXE_DEV_NAME}" $SCRIPT_NAME

# Move executables to the correct location
mv dist/$EXE_NAME .
mv dist/$EXE_DEV_NAME .

# Create the zip file
zip -r $ZIP_NAME $FACTION_IMAGES_DIR $SAVED_INPUT_DIR $EXE_NAME $EXE_DEV_NAME

# Confirm the zip was created
if [ -f "$ZIP_NAME" ]; then
    echo "Packaging and zipping completed successfully."
    echo "The following files have been zipped into ${ZIP_NAME}:"
    echo " - ${FACTION_IMAGES_DIR}/"
    echo " - ${SAVED_INPUT_DIR}/"
    echo " - ${EXE_NAME}"
    echo " - ${EXE_DEV_NAME}"
else
    echo "Error: Zip file was not created."
fi

# Clean up the build directories
rm -rf build dist *.spec $EXE_NAME $EXE_DEV_NAME

read -n 1 -s -r -p "Press any key to exit..."
