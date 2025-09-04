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


def get_batch_id() -> str:
    while True:
        try:
            batch_id = input("Please enter a batch ID: ").strip()
            if not batch_id:
                print("Batch ID cannot be empty. Please try again.")
                continue

            if not all(c.isalnum() or c in ["_", "-"] for c in batch_id):
                print(
                    "Batch ID can only contain alphanumeric characters, underscores, or dashes. Please try again."
                )
                continue

            break
        except EOFError:  # Apparently hitting CTRL-D inputs an EOF character
            print("\nNo input received for batch_id. Aborting capture.")
            return

    return batch_id


def capture_image(save_dir: str, img_index: int) -> None:
    timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    image_path = os.path.join(save_dir, f"{timestamp_str}_{img_index}.jpg")

    if os.path.exists(image_path):
        print(
            f"Image {image_path} already exists. Please press the button and try again."
        )
        return

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
        subprocess.run(cam_command, check=True)
        print(f"Image saved to {image_path}")

    except subprocess.CalledProcessError as e:
        print(f"image capture failed with error: {e.returncode}")


def main() -> None:
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
    batch_id = get_batch_id()
    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
    batch_dir_name = f"{date_string}_{batch_id}"
    save_dir = os.path.join(BASE_DIR, batch_dir_name)

    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {save_dir}: {e}")
        return

    print("Press the button to capture an image...")
    print("Or press CTRL-C to exit the program...")

    img_index = 0

    # Main loop
    try:
        while True:
            # The button is active low
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                print("Button pressed! Capturing image...")
                img_index += 1
                capture_image(save_dir, img_index)
                print("Press the button to capture the next image...")

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
