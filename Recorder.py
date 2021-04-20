import cv2
import numpy as np
from datetime import datetime
import os


class Recorder:
    def __init__(self):
        self.is_recording = False
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = None  # start_recording(...) has to be called to record

        date_time_string = datetime.now().strftime("D%d-%m-%Y-T%H-%M-%S/")
        self.directory_path = "videos/" + date_time_string
        os.makedirs(self.directory_path, exist_ok=True)

    def start_recording(self, file_name, fps, video_size):
        self.video = cv2.VideoWriter(self.directory_path + file_name,
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
