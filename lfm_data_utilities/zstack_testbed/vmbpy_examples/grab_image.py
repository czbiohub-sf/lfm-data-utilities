# Synchronous grab
from vmbpy import *
import cv2

with VmbSystem.get_instance() as vmb:
	cams = vmb.get_all_cameras()
	with cams[0] as cam:
		# print(dir(cam))
		exposure_time = cam.ExposureTime
		time = exposure_time.get()
		inc = exposure_time.get_increment()
		exposure_time.set(time - (inc))
		print(exposure_time)
		# Aquire single frame synchronously
		frame = cam.get_frame()
		if frame.get_status() == FrameStatus.Complete:
			img = frame.as_numpy_ndarray()[:, :, 0].copy()
			# cv2.imshow('Frame', img)
			cv2.imwrite('Frame.png', img)
