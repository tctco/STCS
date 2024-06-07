from pathlib import Path
from typing import Union
import cv2


class VideoReader(cv2.VideoCapture):
    def __init__(self, path: Union[str, Path]):
        # check if file exists
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} does not exist.")
        super().__init__(str(path))
        self.fps = self.get(cv2.CAP_PROP_FPS)
        self.frame_cnt = int(self.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = int(self.get(cv2.CAP_PROP_FOURCC))
