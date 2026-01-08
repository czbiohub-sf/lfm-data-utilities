import logging
from zaber_controller import ZaberCon
from avt_cam import AVTCam
import os
import cv2
from datetime import datetime

# Function for opening up camera to allow for manual focus
def live_focus(zaber_stage, camera):
    print("Position target in focus.\nClick Enter to take Z-stack centered at current position.")

    # Setting up text formatting for displaying instructions
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    thickness = 2
    padding = 8  # px padding inside the background box
    text_color = (255, 255, 255)  # white
    box_alpha = 0.6  # transparency for background box (0.0 transparent, 1.0 opaque)

    # Opening up preview window
    cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)

    # Set stage know mode to displacement for steps of 1
    for ax in zaber_stage.stage_alias.values():
        ax.settings.set('knob.distance', 1)
        ax.settings.set('knob.mode', 1)

    zaber_stage.manual_drive(True)

    # Get exposure and exposure increment from camera
    exposure, exp_inc = camera.get_exposure_time()
    
    # Stage movement step size
    stage_step = 1.9 # um

    while True:
        img = camera.snap()
        if img is None:
            continue

                # ---- Prepare overlay texts ----
        exp_text = f"Exposure: {int(exposure)} us"
        step_text = f"Exp Step: {int(exp_inc)} us"
        stage_step_text = f"Stage Step: {stage_step:.2f}"
        hint_lines = [
            "Controls:",
            "+ / -   : increase / decrease exposure",
            "[ / ]   : decrease / increase exp step",
            " Left / Right : move stage left / right",
            " Up / Down : stage step x10 / x0.1",
            "Enter   : take Z-stack (advance)",
            "Esc     : cancel / exit"
        ]

        # compute total overlay size by measuring each line
        lines = [exp_text, step_text, stage_step_text] + hint_lines
        sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
        max_w = max(w for (w, h) in sizes)
        total_h = sum(h for (w, h) in sizes) + (len(lines) - 1) * 6  # small line spacing

        # position: top-left with margin
        margin_x, margin_y = 10, 10
        x = margin_x
        y = margin_y + sizes[0][1]  # baseline for first text

        # create semi-transparent box on the image
        img_overlay = img.copy()
        rect_tl = (x - padding, y - sizes[0][1] - padding)
        rect_br = (x + max_w + padding, y + total_h + padding)
        cv2.rectangle(img_overlay, rect_tl, rect_br, (0, 0, 0), cv2.FILLED)
        # blend
        cv2.addWeighted(img_overlay, box_alpha, img, 1 - box_alpha, 0, img)

        # draw each line
        line_y = y
        for i, line in enumerate(lines):
            cv2.putText(img, line, (x, line_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
            # advance y by this line height + spacing
            line_h = sizes[i][1]
            line_y += line_h + 6

        # show
        cv2.imshow('Preview', img)
        key = cv2.waitKey(1) & 0xFF
	
        # ---- Exposure control ----
        if key == ord('+') or key == ord('='):  # treat '=' as shifted '+'
            exposure += exp_inc
            try:
                exposure = camera.set_exposure_time(exposure)
                print(f"Exposure set to {int(exposure)} us")
            except Exception:
                # If camera.set_exposure_time returns None or raises, keep local value and print a warning
                print("Warning: Unable to set camera exposure, keeping previous value.")

        elif key == ord('-'):
            exposure = max(1, exposure - exp_inc)
            try:    
                exposure = camera.set_exposure_time(exposure)
                print(f"Exposure set to {int(exposure)} us")
            except Exception:
                print("Warning: Unable to set camera exposure, keeping previous value.")

        # ---- Exposure Increment Control ----
        elif key == ord(']'):
            exp_inc *= 10
            print(f"Exposure increment set to {int(exp_inc)} us")

        elif key == ord('['):
            exp_inc = max(1, int(exp_inc // 10))
            print(f"Step set to {int(exp_inc)} us")

        # ---- Stage movement control (arrow keys) ----
        elif key == 81 or key == 2:  # Left arrow
            try:
                zaber_stage.move_arm('h', -stage_step, speed=1000, is_relative=True)
                print(f"Moved stage left by {stage_step}")
            except Exception as e:
                print(f"Warning: Unable to move stage: {e}")

        elif key == 83 or key == 3:  # Right arrow
            try:
                zaber_stage.move_arm('h', stage_step, speed=1000, is_relative=True)
                print(f"Moved stage right by {stage_step}")
            except Exception as e:
                print(f"Warning: Unable to move stage: {e}")

                print(f"Warning: Unable to move stage: {e}")

        # ---- Stage step size control (up/down arrows) ----
        elif key == 82 or key == 0:  # Up arrow - increase step 10x
            stage_step = min(25000, stage_step * 10)
            print(f"Stage step set to {stage_step:.2f}")

        elif key == 84 or key == 1:  # Down arrow - decrease step 10x
            stage_step = max(0.19, stage_step / 10)
            print(f"Stage step set to {stage_step:.2f}")

         # Enter advances / exits loop
        elif key in (13, 10):
            cv2.destroyAllWindows()
            zaber_stage.manual_drive(False)
            return True  # Continue with z-stack

        # Esc to cancel/exit script
        elif key == 27:
            print("Cancelled by user (Esc).")
            cv2.destroyAllWindows()
            zaber_stage.manual_drive(False)
            return False  # Exit script

    # Catch any other keys and exit loop
    cv2.destroyAllWindows()
    zaber_stage.manual_drive(False)
    return True


def main():
	
	BASE_DIR = '/home/pi/Desktop/zstacks' # Where zstack images folder is saved
	NUM_FRAMES = 80 # Number of images taken in stack
	STACK_STEP_SIZE = 0.19 #um needs to be in 0.19 increments
	EST_FOCUS_POS = 24731 #um Moves stage to this position before manual focus prompt
	START_EXPOSURE = 14265 #us
	
	logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
	
	user_note = input("Enter folder name prefix. ")
	
	# Setting up image save directory
	date_dir = datetime.now().strftime("%Y_%m_%d")
	session_dir_name = user_note + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M")
	save_dir = os.path.join(BASE_DIR, date_dir, session_dir_name)
	os.makedirs(save_dir, exist_ok=True)
	
	# Setting up stage
	zc = ZaberCon()
	print('Manual control disabled. Run script to completion to re-enable manual control.')
	zc.manual_drive(False)
	
	# Setting up camera
	cam = AVTCam(preview=True)
	cam.open(camera_id="DEV_1AB22C018DE3")
	
	cam.set_exposure_time(START_EXPOSURE)

	# Begin taking the z stack
	zc.move_arm('h', EST_FOCUS_POS, speed=1000, is_relative=False) # move close to focus point
	
	if not live_focus(zaber_stage=zc, camera=cam):
		print("Exiting script...")
		cam.close()
		zc.close()
		return
	
	focus_pos = zc.get_pos('h')
	
	# Move stage to starting position
	zc.move_arm('h', -NUM_FRAMES//2 * STACK_STEP_SIZE, speed=1000, is_relative=True)
	
	for i in range(NUM_FRAMES):
		print(i, zc.get_pos('h'))
		img = cam.snap()
		cv2.imwrite(f'{save_dir}/{i}.png', img)
		zc.move_arm('h', STACK_STEP_SIZE, speed=1000, is_relative=True)
		
	# Move stage back to focus position
	zc.move_arm('h', focus_pos, speed=1000, is_relative=False)
		
	cam.close()
	zc.close()
	
if __name__ == "__main__":
	main()
