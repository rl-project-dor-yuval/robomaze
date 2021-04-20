import cv2
import numpy as np


class Recorder:
    def __init__(self):
        self.is_recording = False
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = None  # start_recording(...) has to be called to record

    def start_recording(self, file_name, fps, video_size):
        self.video = cv2.VideoWriter(file_name,
                                     self.fourcc,
                                     fps,
                                     video_size,
                                     True)
        self.is_recording = True

    def insert_frame(self, image_array):
        # remove the fourth layer which is the alpha
        colored_image = image_array[:, :, 0:3]
        fliped_image = np.flip(colored_image, axis=2)
        self.video.write(fliped_image)

    def save_recording_and_reset(self):
        self.video.release()
        self.is_recording = False
