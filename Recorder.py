import cv2
import numpy as np
from datetime import datetime
import os
import pybullet as p


class Recorder:
    """
     Recorder component to create video file from an episode using cv2.
     start_recording() is called at the beginning of an episode,
     every time insert_current_frame() is called, a frame with the current
     state of pybullet is added to the video and the video is saved when
     save_recording_and_reset() is called.

     attributes fps, video_size, zoom can be set only one time on __init__,
     if a setters would be created in the future, it is necessary to make new
     projection and view matrices have to be recreated.
    """
    is_recording: bool

    def __init__(self, maze_size, fps=24, video_size=(800, 600), zoom=1):
        self._maze_size = maze_size
        self._video_size = video_size
        self._fps = fps
        self._zoom = zoom
        self.is_recording = False
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video = None  # start_recording(...) has to be called to record

        # prepare files and directory:
        date_time_string = datetime.now().strftime("D%d-%m-%Y-T%H-%M-%S/")
        self._directory_path = "videos/" + date_time_string
        os.makedirs(self._directory_path, exist_ok=True)

        # setup camera:
        camera_distance = max(self._maze_size) / self._zoom  # depends on the longer axis
        focal_point = (maze_size[0] / 2, maze_size[1] / 2, 0)  # middle
        aspect = video_size[0]/video_size[1]

        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(distance=camera_distance,
                                                                yaw=90,
                                                                pitch=-90,
                                                                roll=0,
                                                                upAxisIndex=2,
                                                                cameraTargetPosition=focal_point
                                                                )
        self._projection_matrix = p.computeProjectionMatrixFOV(fov=70,
                                                               aspect=aspect,
                                                               nearVal=camera_distance - 4,
                                                               farVal=camera_distance + 1)

    def start_recording(self, file_name):
        self._video = cv2.VideoWriter(self._directory_path + file_name,
                                      self._fourcc,
                                      self._fps,
                                      self._video_size,
                                      True)
        self.is_recording = True

    def insert_current_frame(self):
        _, _, image_array, _, _ = p.getCameraImage(width=self._video_size[0],
                                                   height=self._video_size[1],
                                                   viewMatrix=self._view_matrix,
                                                   projectionMatrix=self._projection_matrix,
                                                   renderer=p.ER_TINY_RENDERER,
                                                   flags=p.ER_NO_SEGMENTATION_MASK)

        # remove the fourth layer which is the alpha
        colored_image = image_array[:, :, 0:3]
        # flip colors from rgb to bgr (for cv2 api)
        fliped_image = np.flip(colored_image, axis=2)

        self._video.write(fliped_image)

    def save_recording_and_reset(self):
        self._video.release()
        self.is_recording = False
