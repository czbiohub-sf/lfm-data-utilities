#!/usr/bin/env python3

import time
import os
import subprocess
from datetime import datetime

import RPi.GPIO as GPIO
import board
import neopixel

# HARDWARE CONSTANTS
# GPIO pins
BUTTON_PIN = 23
NEOPIXEL_PIN = board.D18
NUM_PIXELS = 24
PIXEL_ORDER = neopixel.GRB
BPP = 4  # Bytes per pixel (R, G, B, W)
LED_BRIGHTNESS = 0.3
SLEEP_BEFORE_CAPTURE = 3
LED_PATTERN = (255, 0, 150)
OFF_PATTERN = (0, 0, 0)

# Camera arguments
IMG_BRIGHTNESS = "0"  # Range is -1 to +1
METERING = "spot"  # Only weight the center of the image
WHITE_BALANCE = "auto"  # Could be better to fix wb gains but fine for now
IMG_ROI = "0, 0, 1, 1"
PREVIEW_SIZE = "100,100,640,480"


# Directory to save images
BASE_DIR = "/home/pi/Desktop/qc_images"


def capture_image(save_dir):

    # Get batch and chip_ids
    while True:
        try:
            batch_id = input("Please enter a batch ID: ").strip()
            if not batch_id:
                print("Batch ID cannot be empty. Please try again.")
                continue
            break
        except EOFError:  # Apparently hitting CTRL-D inputs an EOF character
            print("\nNo input received for batch_id. Aborting capture.")
            return

    while True:
        try:
            chip_id = input("Please enter a flowcell ID: ").strip()
            if not chip_id:
                print("Flowcell ID cannot be empty. Please try again.")
                continue
            break
        except EOFError:
            print("\nNo input received for chip_id. Aborting capture.")
            return

    timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    image_path = os.path.join(save_dir, f"{timestamp_str}_{batch_id}_{chip_id}.jpg")

    cam_command = [
        "rpicam-still",
        "--output",
        image_path,
        "--brightness",
        IMG_BRIGHTNESS,
        "--encoding",
        "jpg",
        "--metering",
        METERING,
        "--roi",
        IMG_ROI,
        "-p",
        PREVIEW_SIZE,
    ]

    try:
        ans = subprocess.run(cam_command, check=True)
        print(f"Image saved to {image_path}")

    except subprocess.CalledProcessError as e:
        print(f"image capture failed with error: {e.returncode}")


def main():

    # Initialize GPIO
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        neopixels = neopixel.NeoPixel(
            NEOPIXEL_PIN,
            NUM_PIXELS,
            brightness=LED_BRIGHTNESS,
            auto_write=True,
            pixel_order=PIXEL_ORDER,
        )
    except Exception as e:
        print(f"Failed to initialize GPIO or NeoPixels: {e}")
        return

    # Light up the neopixels
    print("Turning on neopixels")
    neopixels.fill(LED_PATTERN)
    neopixels.show()

    # Create a directory for the current session
    session_dir_name = datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_dir = os.path.join(BASE_DIR, session_dir_name)

    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {save_dir}: {e}")
        return

    print("Press the button to capture an image...")
    print("Or press CTRL-C to exit the program...")

    # Main loop
    try:
        while True:
            # The button is active low
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                print("Button pressed! Capturing image...")
                capture_image(save_dir)

                # Simple debounce: wait until button is released
                while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    time.sleep(1)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Exiting on keyboard interrupt...")

    finally:
        print("Cleaning up GPIO and camera...")
        neopixels.fill(OFF_PATTERN)
        neopixels.show()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
