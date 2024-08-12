import sys
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
from PIL import Image
import time
import mss
import threading
import tkinter as tk
import json
import os

# Determine if the script is running as a PyInstaller bundle
if getattr(sys, 'frozen', False):
    # If so, set the Tesseract executable path relative to the executable location
    tesseract_exe_path = os.path.join(sys._MEIPASS, 'tesseract.exe')
    tessdata_dir = os.path.join(sys._MEIPASS, 'tessdata')
else:
    # Use the regular path if running the script directly
    tesseract_exe_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    tessdata_dir = r"C:\Program Files\Tesseract-OCR\tessdata"

pytesseract.pytesseract.tesseract_cmd = tesseract_exe_path
os.environ['TESSDATA_PREFIX'] = tessdata_dir

# Program config
font = ("Arial", 12, "")
text_bot_height = 9

# Global variables
running = False
config = {}

def get_all_file_names(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Specifically for finding images
def preprocess_image_for_image(image, scale_factor=0.5):
    # Convert to PIL Image to rescale
    pil_image = Image.frombytes('RGB', image.size, image.rgb)
    pil_image = pil_image.resize((int(pil_image.width * scale_factor), int(pil_image.height * scale_factor)))

    # Convert image to grayscale
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2GRAY)
    
    return gray

# Specifically for finding text
def preprocess_image_for_text(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Display the result (Optional)
    # cv2.imshow('Image', np.array(thresholded))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return thresholded

# Assumes all images to search for are in 'faction_images' folder
# Also assumes that each faction image name corresponds to factions specified in config.json
def check_for_images(processed_image):
    res = []

    for img_file_name in get_all_file_names('faction_images'):
        img_path = 'faction_images/' + img_file_name
        template = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Error loading image: {img_path}")
            continue
        
        # Match template
        result = cv2.matchTemplate(processed_image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print(max_val, max_loc)

        # Set a threshold for the match
        threshold = 0.5
        if max_val >= threshold:
            print(f"Image '{img_path}' found on the screen at '{max_loc}'!")
            img_file_name_no_file_ext = img_file_name.rsplit('.', 1)[0]
            res.append({
                "faction": img_file_name_no_file_ext,
                "coord": max_loc
            })

            # Draw a rectangle around the detected image and 
            # h, w = template.shape
            # top_left = max_loc
            # bottom_right = (top_left[0] + w, top_left[1] + h)
            # processed_image = cv2.rectangle(np.array(processed_image), top_left, bottom_right, (0, 255, 0), 2)
            
    # Display the result (Optional)
    # cv2.imshow('Image', np.array(processed_image))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return res

def check_for_text_existence(processed_image, target_text_list):
    # Use Tesseract to extract text from the preprocessed image
    custom_config = r'--oem 3 --psm 11'
    text = pytesseract.image_to_string(processed_image, config=custom_config)
    
    # Print the text to check what Tesseract recognized
    print("===Text Found Beginning===")
    print(text)
    print("===Text Found End===")
    
    # Display the result (Optional)
    # cv2.imshow('Image', np.array(processed_image))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Check if any target text is in the extracted text
    for target_text in target_text_list:
        if target_text.lower() in text.lower():
            print(f"'{target_text}' found on the screen!")
            return target_text

def check_for_text_location(processed_image, target_text):
    # Extract text and bounding boxes
    custom_config = r'--oem 3 --psm 11'
    data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=Output.DICT)

    # Initialize variables to store bounding box coordinates
    phrase_coords = None
    current_text = ""

    # Iterate through the text data
    for i in range(len(data['text'])):
        if data['text'][i].strip():
            current_text += data['text'][i] + " "
            if target_text.lower() in current_text.lower():
                # Calculate the bounding box for the target_text
                x = min(data['left'][i-len(target_text.split())+1], data['left'][i])
                y = min(data['top'][i-len(target_text.split())+1], data['top'][i])
                w = max(data['left'][i] + data['width'][i], data['left'][i-len(target_text.split())+1] + data['width'][i-len(target_text.split())+1]) - x
                h = max(data['top'][i] + data['height'][i], data['top'][i-len(target_text.split())+1] + data['height'][i-len(target_text.split())+1]) - y
                phrase_coords = (x, y, x + w, y + h)
                break

    if phrase_coords:
        print(f"target_text '{target_text}' found at coordinates: {phrase_coords}")
        return phrase_coords
    else:
        print(f"target_text '{target_text}' not found.")

def core_loop():
    while running:
        time.sleep(1)
        # Core loop
        with mss.mss() as sct:
            # Capture one entire monitor
            screenshot = sct.grab(sct.monitors[int(monitor_number_input.get())])
            
            # Preprocess the image for image search
            scale_factor = 0.5 # to reduce image size and therefore computer resource load
            processed_image_for_image = preprocess_image_for_image(screenshot, scale_factor)
            screenshot_middle_x = screenshot.width / 2 * scale_factor

            # Preprocess the image for text search
            processed_image_for_text = preprocess_image_for_text(screenshot)

            map_names = load_from_config("maps")
            map = check_for_text_existence(processed_image_for_text, map_names)
            if not map:
                print("No map found")
                continue

            # Find factions
            factions = check_for_images(processed_image_for_image)
            if not factions:
                print("No factions found")
                continue

            # if only 1 faction, then it's a mirror
            left_faction = factions[0]
            right_faction = factions[0]

            # if 2 factions
            if len(factions) > 1:
                if left_faction["coord"][0] < screenshot_middle_x:
                    left_faction = factions[0]
                    right_faction = factions[1]
                else:
                    right_faction = factions[0]
                    left_faction = factions[1]

            # Find username
            username_coords = check_for_text_location(processed_image_for_text, username_input.get())
            if not username_coords:
                print("No username found")
                continue
            user_on_left = True
            if username_coords[0] > screenshot_middle_x:
                user_on_left = False

            # Change your_faction and opponent_faction
            if user_on_left:
                your_faction_var.set(left_faction["faction"])
                opponent_faction_var.set(right_faction["faction"])
            else:
                your_faction_var.set(right_faction["faction"])
                opponent_faction_var.set(left_faction["faction"])

            # Change map
            map_var.set(map)

            # Call button functions
            on_update_your_faction(your_faction_var.get())
            on_update_map(map_var.get())

    start_button.config(state=tk.NORMAL)
    status_label.config(text="Status: Stopped")

def save_config():
    print("Saving config...")

    config["username"] = username_input.get()
    config["monitor_number"] = int(monitor_number_input.get())
    config["your_faction"] = your_faction_var.get()
    config["opponent_faction"] = opponent_faction_var.get()
    config["map"] = map_var.get()

    with open("saved_input/config.json", 'w') as file:
        json.dump(config, file, indent=4)  # indent=4 for pretty-printing
        
    print("Successfully saved config.")

def load_from_config(key):
    global config
    if key in config:
        return config[key]
    print("Couldn't find " + key + " in config.")
    return ""

def save_notes(notes_text_input, notes_file_name):
    content = notes_text_input.get("1.0", tk.END).strip()
    file_name = 'saved_input/' + notes_file_name
    with open(file_name, 'w') as file:
        file.write(content)

# Assumes file name in saved_input folder
def load_notes(notes_text_input, notes_file_name):
    file_name = 'saved_input/' + notes_file_name
    notes_text_input.delete(1.0, tk.END)  # Clear any existing content

    if not os.path.exists(file_name):
        return

    with open(file_name, 'r') as file:
        content = file.read()
        notes_text_input.insert(tk.END, content)

def start():
    global running
    running = True
    start_button.config(state=tk.DISABLED)
    status_label.config(text="Status: Running")
    # Start the core loop in a new thread to keep the UI responsive
    threading.Thread(target=core_loop, args=(), daemon=True).start()

def stop():
    global running

    if not running:
        return
    
    running = False
    status_label.config(text="Status: Pausing...")

    # Start_button is reenabled when core_loop ends

def on_close():
    global running
    running = False
    save_config()
    save_notes(general_notes_input, 'General_Notes.txt')
    save_current_faction_notes()
    save_current_matchup_notes()
    save_current_map_notes()
    save_current_map_matchup_notes()
    root.destroy()

def on_update_your_faction(your_faction):
    save_current_faction_notes()

    load_current_faction_notes()
    
    update_matchup()

def save_current_faction_notes():
    current_notes_file_name = "General_" + load_from_config("your_faction") + "_Notes.txt"
    save_notes(faction_notes_input, current_notes_file_name)

def load_current_faction_notes():
    your_faction = your_faction_var.get()

    # Update the general faction notes label
    faction_notes_input_label.config(text=faction_notes_input_label_string.format(your_faction=your_faction))

    # Load in your new faction notes
    new_notes_file_name = "General_" + your_faction + "_Notes.txt"
    load_notes(faction_notes_input, new_notes_file_name)

def on_update_opponent_faction(opponent_faction):
    update_matchup()

def update_matchup():
    save_current_matchup_notes()
    save_current_map_matchup_notes()
    
    load_new_matchup_notes()
    load_new_map_matchup_notes()

    # Set new factions to in-memory (as opposed to saved to txt) config
    config["your_faction"] = your_faction_var.get()
    config["opponent_faction"] = opponent_faction_var.get()

def save_current_matchup_notes():
    current_matchup = get_matchup(load_from_config("your_faction"), load_from_config("opponent_faction"))
    current_notes_file_name = "General_" + current_matchup + "_Notes.txt"
    save_notes(matchup_notes_input, current_notes_file_name)

def load_new_matchup_notes():
    matchup = get_matchup(your_faction_var.get(), opponent_faction_var.get())

    # Update matchup notes label
    matchup_notes_input_label.config(text=matchup_notes_input_label_string.format(matchup=matchup))

    # Load in new matchup notes
    matchup_notes_file_name = "General_" + matchup + "_Notes.txt"
    load_notes(matchup_notes_input, matchup_notes_file_name)

# Takes in both faction strings, returns first letters separated by 'v' e.g. "CvC"
def get_matchup(your_faction_string, opponent_faction_string):
    return your_faction_string[0] + "v" + opponent_faction_string[0]

def on_update_map(map):
    save_current_map_notes()
    save_current_map_matchup_notes()

    load_new_map_notes()
    load_new_map_matchup_notes()

    # Set new map to in-memory (as opposed to saved to txt) config
    config["map"] = map

def save_current_map_notes():
    map_name_no_spaces = load_from_config("map").replace(" ", "")
    current_notes_file_name = "General_Map" + map_name_no_spaces + "_Notes.txt"
    save_notes(map_notes_input, current_notes_file_name)

def save_current_map_matchup_notes():
    current_matchup = get_matchup(load_from_config("your_faction"), load_from_config("opponent_faction"))
    map_name_no_spaces = load_from_config("map").replace(" ", "")
    current_notes_file_name = "Map" + map_name_no_spaces + "_" + current_matchup + "_Notes.txt"
    save_notes(map_matchup_notes_input, current_notes_file_name)

def load_new_map_notes():
    map = map_var.get()

    # Update map notes label
    map_notes_input_label.config(text=map_notes_input_label_string.format(map=map))

    # Load in new map notes
    map_name_no_spaces = map.replace(" ", "")
    map_notes_file_name = "General_Map" + map_name_no_spaces + "_Notes.txt"
    load_notes(map_notes_input, map_notes_file_name)

def load_new_map_matchup_notes():
    matchup = get_matchup(your_faction_var.get(), opponent_faction_var.get())
    map = map_var.get()

    # Update map matchup notes label
    map_matchup_notes_input_label.config(text=map_matchup_notes_input_label_string.format(map=map, matchup=matchup))

    # Load in new map matchup notes
    map_name_no_spaces = map.replace(" ", "")
    map_matchup_notes_file_name = "Map" + map_name_no_spaces + "_" + matchup + "_Notes.txt"
    load_notes(map_matchup_notes_input, map_matchup_notes_file_name)

def test_monitor():
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[int(monitor_number_input.get())])
        cv2.imshow('Image', np.array(screenshot))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Load config
print("Loading config...")
with open("saved_input/config.json", 'r') as file:
    config = json.load(file)
print("Successfuly loaded config.")

# Create the GUI
root = tk.Tk()
root.title("Stormgate Notes")

status_label = tk.Label(root, text="Status: Stopped")

start_button = tk.Button(root, text="Start", command=start)

stop_button = tk.Button(root, text="Stop", command=stop)

username_input_label = tk.Label(root, text="Username:")
username_input = tk.Entry(root)
username_input.insert(0, load_from_config("username"))

monitor_number_input_label = tk.Label(root, text="Monitor Number:")
monitor_number_var = tk.StringVar(root)
monitor_number_var.set(load_from_config("monitor_number"))
monitor_number_input = tk.Spinbox(root, from_=1, to=10, textvariable=monitor_number_var)

test_monitor_button = tk.Button(root, text="Test Monitor", command=test_monitor)

general_notes_input_label = tk.Label(root, text="General Notes:")
general_notes_input = tk.Text(root, height=text_bot_height, width=40, font=font)
load_notes(general_notes_input, 'General_Notes.txt')

factions = load_from_config("factions")

your_faction_label = tk.Label(root, text="Your Faction:")
your_faction_var = tk.StringVar()
your_faction_var.set(load_from_config("your_faction"))  # Set first value
your_faction_dropdown = tk.OptionMenu(root, your_faction_var, *factions, command=on_update_your_faction)

opponent_faction_label = tk.Label(root, text="Opponent Faction:")
opponent_faction_var = tk.StringVar()
opponent_faction_var.set(load_from_config("opponent_faction"))  # Set first value
opponent_faction_dropdown = tk.OptionMenu(root, opponent_faction_var, *factions, command=on_update_opponent_faction)

faction_notes_input_label_string = "General <{your_faction}> Notes:"
faction_notes_input_label = tk.Label(root, text=faction_notes_input_label_string)
faction_notes_input = tk.Text(root, height=text_bot_height, width=40, font=font)
load_current_faction_notes()

matchup_notes_input_label_string = "General <{matchup}> Notes:"
matchup_notes_input_label = tk.Label(root, text=matchup_notes_input_label_string)
matchup_notes_input = tk.Text(root, height=text_bot_height, width=40, font=font)
load_new_matchup_notes()

map_label = tk.Label(root, text="Map:")
maps = load_from_config("maps")
map_var = tk.StringVar()
map_var.set(load_from_config("map"))  # Set first value
map_dropdown = tk.OptionMenu(root, map_var, *maps, command=on_update_map)

map_notes_input_label_string = "General <{map}> Notes:"
map_notes_input_label = tk.Label(root, text=map_notes_input_label_string)
map_notes_input = tk.Text(root, height=text_bot_height, width=40, font=font)
load_new_map_notes()

map_matchup_notes_input_label_string = "<{map}> <{matchup}> Notes:"
map_matchup_notes_input_label = tk.Label(root, text=map_matchup_notes_input_label_string)
map_matchup_notes_input = tk.Text(root, height=text_bot_height, width=40, font=font)
load_new_map_matchup_notes()

status_label.grid(row=0, column=0)
start_button.grid(row=0, column=1)
stop_button.grid(row=0, column=2)

username_input_label.grid(row=1, column=0)
username_input.grid(row=1, column=1, columnspan=2)

monitor_number_input_label.grid(row=2, column=0)
monitor_number_input.grid(row=2, column=1, columnspan=2)
test_monitor_button.grid(row=2, column=3)

general_notes_input_label.grid(row=3, column=0)
general_notes_input.grid(row=3, column=1, columnspan=4, sticky="nsew")

your_faction_label.grid(row=4, column=1)
your_faction_dropdown.grid(row=4, column=2)
opponent_faction_label.grid(row=4, column=3)
opponent_faction_dropdown.grid(row=4, column=4)

faction_notes_input_label.grid(row=5, column=0)
faction_notes_input.grid(row=5, column=1, columnspan=4, sticky="nsew")

matchup_notes_input_label.grid(row=6, column=0)
matchup_notes_input.grid(row=6, column=1, columnspan=4, sticky="nsew")

map_label.grid(row=7, column=1)
map_dropdown.grid(row=7, column=2)

map_notes_input_label.grid(row=8, column=0)
map_notes_input.grid(row=8, column=1, columnspan=4, sticky="nsew")

map_matchup_notes_input_label.grid(row=9, column=0)
map_matchup_notes_input.grid(row=9, column=1, columnspan=4, sticky="nsew")

# Make all rows and columns resizable
total_rows=10
total_columns=10
for row in range(total_rows):
    root.grid_rowconfigure(row, weight=1)

for col in range(total_columns):
    root.grid_columnconfigure(col, weight=1)

root.protocol("WM_DELETE_WINDOW", on_close)  # Bind the window close event

root.mainloop()
