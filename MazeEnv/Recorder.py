import imageio
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

    def __init__(self, pybullet_client, maze_size, fps=24, video_size=(800, 600), zoom=1):
        self._pclient = pybullet_client
        self._maze_size = maze_size
        self._video_size = video_size
        self._fps = fps
        self._zoom = zoom
        self.is_recording = False
        self.vid_images = []
        self.path = None

        # prepare files and directory:
        date_time_string = datetime.now().strftime("D%d-%m-%Y-T%H-%M-%S/")
        self._directory_path = "videos/" + date_time_string

        # setup camera:
        camera_distance = max(self._maze_size) / self._zoom  # depends on the longer axis
        focal_point = [maze_size[0] / 2, maze_size[1] / 2, 0]  # middle
        focal_point[0] += 2
        aspect = video_size[0] / video_size[1]

        self._view_matrix = self._pclient.computeViewMatrixFromYawPitchRoll(distance=camera_distance,
                                                                            yaw=90,
                                                                            pitch=-65,
                                                                            roll=0,
                                                                            upAxisIndex=2,
                                                                            cameraTargetPosition=focal_point
                                                                            )

        # temporary fix that works for our uses. TODO: find better defenition for farval and nearval
        far_val = camera_distance + max(self._maze_size)/2

        self._projection_matrix = self._pclient.computeProjectionMatrixFOV(fov=70,
                                                                           aspect=aspect,
                                                                           nearVal=camera_distance - 6,
                                                                           farVal=far_val)

    def start_recording(self, file_name, custom_path=False):
        """
        :param file_name: the name of the file to save the video to
        :param custom_path: if true, then file name will be used as a full path to
                            the saved video, instead of the default path.
        """
        if custom_path:
            self.path = file_name
        else:
            os.makedirs(self._directory_path, exist_ok=True)
            self.path = self._directory_path + file_name

        self.vid_images = []
        self.is_recording = True

    def insert_current_frame(self):
        _, _, image_array, _, _ = self._pclient.getCameraImage(width=self._video_size[0],
                                                               height=self._video_size[1],
                                                               viewMatrix=self._view_matrix,
                                                               projectionMatrix=self._projection_matrix,
                                                               renderer=self._pclient.ER_TINY_RENDERER,
                                                               flags=self._pclient.ER_NO_SEGMENTATION_MASK)

        # remove the fourth layer which is the alpha
        colored_image = image_array[:, :, 0:3]
        # flip colors from rgb to bgr (for cv2 api)
        # fliped_image = np.flip(colored_image, axis=2)

        self.vid_images.append(colored_image)

    def save_recording_and_reset(self):
        imageio.mimsave(self.path, self.vid_images, fps=self._fps)
        self.is_recording = False
