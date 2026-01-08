# From example script: https://github.com/alliedvision/VimbaPython/blob/master/Examples/synchronous_grab.py

import cv2
import sys
import time
import queue
import vmbpy
import logging
import os

import numpy as np

from typing import Optional

from vmbpy import (
    Camera,
    VmbSystem,
    Stream,
    Frame,
    VmbCameraError,
    AllocationMode,
    FrameStatus,
)
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject


FRAME_HEIGHT = 2200
FRAME_WIDTH = 2752


class AVTCam(QObject):
    new_img = pyqtSignal(np.ndarray)
    streaming = pyqtSignal(bool)

    def __init__(self, preview : bool = False):
        super().__init__()

        sys.excepthook = self._camera_excepthook

        # self.queue: queue.Queue[Tuple[np.ndarray, float]] = queue.Queue(maxsize=1)
        self.queue = queue.Queue(maxsize=1)
        self.show_preview = preview


    def open(self, camera_id: Optional[str] = None):
        self.vmb = VmbSystem.get_instance()
        self.vmb.__enter__()
        
        if camera_id:
            try:
                self.cam = self.vmb.get_camera_by_id(camera_id)

            except VmbCameraError:
                # TODO test if this elegantly exits when camera is unplugged
                logging.error('Failed to access AVT camera \'{}\'. Abort.'.format(camera_id))
                raise IOError("Failed to access AVT camera")

        else:
            cams = self.vmb.get_all_cameras()
            logging.info("AVT cameras found: {}".format(len(cams)))

            if not cams:
                logging.error('No AVT cameras accessible. Abort.')
                raise IOError("No AVT cameras accessible")

            self.cam = cams[0]
        
        self.cam.__enter__()

        self._setup_camera()
        logging.info("Opened AVT camera")

    def start_stream(self):
        try:
            if not self.cam.is_streaming():
                self.cam.start_streaming(
                    handler=self._frame_handler,
                    # buffer_count=10,
                    allocation_mode=AllocationMode.AnnounceFrame
                )
                logging.debug("AVT camera started streaming")
                self.streaming.emit(True)
                return True
            else:
                return False
        except AttributeError as e:
            logging.error(e)

    def stop_stream(self):
        try:
            if self.cam.is_streaming():
                self.cam.stop_streaming()
                logging.debug("AVT camera stopped streaming")
                self.streaming.emit(False)
        except AttributeError as e:
            pass

    def toggle_stream(self):
        try:
            if not self.start_stream():
                self.cam.stop_streaming()
                logging.debug("AVT camera stopped streaming")
                self.streaming.emit(False)
        except AttributeError as e:
            pass

    def snap(self, show: bool = False) -> Optional[np.ndarray]:
        try:
            frame = self.cam.get_frame()
        except AttributeError as e:
            logging.error(e)

        if frame.get_status() == FrameStatus.Complete:
            img = frame.as_numpy_ndarray()[:, :, 0].copy()

            if show:
                cv2.imshow('Preview', img)

            return img

    def close(self):
        self.stop_stream()

        try:
            self.cam.__exit__(*sys.exc_info())
            logging.info("Closed AVT camera")
        except AttributeError as e:
            pass

        try:
            self.vmb.__exit__(*sys.exc_info())
            logging.info("Closed vmbpy")
        except AttributeError as e:
            pass

    def _setup_camera(self):
        # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
        try:
            stream = self.cam.get_streams()[0]
            stream.GVSPAdjustPacketSize.run()

            while not stream.GVSPAdjustPacketSize.is_done():
                pass

            logging.debug("AVT camera GeV packet size adjusted")

        except (AttributeError, VmbCameraError) as e:
            logging.debug(f'AVT camera GeV packet size adjustment aborted:\n{e}')
            pass

    def _frame_handler(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            img = frame.as_numpy_ndarray()[:, :, 0].copy()

            if self.show_preview:
                cv2.imshow('Preview', img)

            self.new_img.emit(img)

        self.cam.queue_frame(frame)

    def _camera_excepthook(self, *exc_info):
        self.stop_stream()

        try:
            self.cam.__exit__(*exc_info)
            logging.info("Closed camera")
        except AttributeError as e:
            pass
        try:
            self.vmb.__exit__(*exc_info)
            logging.info("Closed vmbpy")

        except AttributeError as e:
            pass
            
        sys.__excepthook__(*exc_info)

    def get_exposure_time(self):
        return self.cam.ExposureTime.get(), self.cam.ExposureTime.get_increment()

    def set_exposure_time(self, exposure_time: float) -> float:        
        cur_exposure_time = self.cam.ExposureTime.get()
        inc = self.cam.ExposureTime.get_increment()

        new_exposure_time = cur_exposure_time + round((exposure_time - cur_exposure_time) / inc) * inc

        self.cam.ExposureTime.set(new_exposure_time)

        return new_exposure_time


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    cam = AVTCam(preview=False)
    cam.open(camera_id="DEV_1AB22C018DE3")

    cam.set_exposure_time(5000)
    print(cam.get_exposure_time())

    cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)

    # Live preview until you press Enter
    while True:
        img = cam.snap()
        if img is None:
            continue

        cv2.imshow('Preview', img)

        # waitKey(1) gives OpenCV time to process GUI events
        key = cv2.waitKey(1) & 0xFF

        # 13 is Enter (Carriage Return). On some setups 10 (Line Feed) can show up.
        if key in (13, 10):  # Enter
            break
        # optional: press ESC to abort too
        if key == 27:  # ESC
            break

    cam.close()
    cv2.destroyAllWindows()