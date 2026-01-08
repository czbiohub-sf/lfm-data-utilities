from avt_cam import AVTCam
import cv2
import os
import logging
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep exposure times on an AVT camera and save images."
    )

    parser.add_argument(
        "-s", "--start",
        type=float,
        default=None,
        help="Starting exposure time (use camera's current exposure if omitted)."
    )

    parser.add_argument(
        "-c", "--count",
        type=int,
        default=100,
        help="Number of images to capture (default: 100)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    cam = AVTCam(preview=True)
    cam.open(camera_id="DEV_1AB22C018DE3")

    # Get current exposure and increment from camera
    exp_time, inc = cam.get_exposure_time()

    # Override starting exposure if provided
    if args.start is not None:
        exp_time = args.start

    logging.info(f"Starting exposure: {exp_time}, increment: {inc}")
    logging.info(f"Capturing {args.count} images")

    os.makedirs('exposure_sweep_imgs', exist_ok=True)

    for i in range(args.count):
        logging.info(f"Setting exposure to {exp_time}")
        cam.set_exposure_time(exp_time)

        img = cam.snap()
        filename = f'exposure_sweep_imgs/img_{round(exp_time)}.png'
        cv2.imwrite(filename, img)
        logging.info(f"Wrote {filename}")

        exp_time += inc

    cam.close()
    logging.info("Camera closed.")


if __name__ == "__main__":
    main()
from avt_cam import AVTCam
import cv2
import os
import logging

def main():

	cam = AVTCam(preview=True)
	cam.open(camera_id="DEV_1AB22C018DE3")
    
	start_exp_time = 0
	cam.set_exposure_time(start_exp_time)
	exp_time, inc = cam.get_exposure_time()

	os.makedirs('exposure_sweep_imgs', exist_ok=True)

	for i in range(100):
		cam.set_exposure_time(exp_time)
		img = cam.snap()
		cv2.imwrite(f'exposure_sweep_imgs/img_{round(exp_time)}.png', img)
		exp_time = exp_time + (i * inc)

	cam.close()

if __name__ == '__main__':
	main()

 