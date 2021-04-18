import os
import time
from itertools import count, zip_longest
from typing import Iterator, List, Sequence, Optional

import cv2 as cv
import numpy as np
from natsort import natsorted

from model import Color, Box, Gender

FACE_DETECTION_MODEL = "models/face_detection/res10_300x300_ssd_iter_140000.caffemodel"

FACE_DETECTION_MODEL_CONFIG = "models/face_detection/deploy.prototxt.txt"
NETWORK_INPUT_SIZE = 300, 300
# Value from https://github.com/opencv/opencv/tree/master/samples/dnn
BLOB_MEAN_SUBTRACTION: Color = Color(r=123, b=104, g=177)


class Face:
    def __init__(self, id_: int, box: Box, gender: Gender = Gender.UNKNOWN):
        self.id_ = id_
        self.boxes = [box]
        self.gender = gender

    @property
    def latest_box(self) -> Optional[Box]:
        return self.boxes[-1] if self.boxes else None


class FaceTracker:
    def __init__(self, detection_threshold: float = 0.5):
        self.detection_threshold = detection_threshold
        self.face_detection_network = cv.dnn.readNetFromCaffe(
            prototxt=FACE_DETECTION_MODEL_CONFIG,
            caffeModel=FACE_DETECTION_MODEL,
        )

    def track_faces(self, clip_dir: str, out_base_dir: str, draw_on_dir: str = None):
        out_dir: str = f'{out_base_dir}{round(time.time())}/'
        os.makedirs(out_dir)

        frames: List[os.DirEntry] = natsorted(
            os.scandir(clip_dir), key=lambda dir_entry: dir_entry.name
        )
        draw_on_frames: List[os.DirEntry] = (
            []
            if draw_on_dir is None
            else natsorted(
                os.scandir(draw_on_dir), key=lambda dir_entry: dir_entry.name
            )
        )

        new_face_id: Iterator[int] = count(start=1)
        face_ids = {}

        for frame, draw_on_frame in zip_longest(frames, draw_on_frames):
            img: np.ndarray = cv.imread(frame.path)
            out_img = (
                img.copy() if draw_on_frame is None else cv.imread(draw_on_frame.path)
            )
            img_300x300: np.ndarray = cv.resize(src=img, dsize=NETWORK_INPUT_SIZE)
            img_caffe_blob: np.ndarray = cv.dnn.blobFromImage(
                image=img_300x300,
                scalefactor=1.0,
                size=NETWORK_INPUT_SIZE,
                mean=BLOB_MEAN_SUBTRACTION.to_bgr(),
            )
            self.face_detection_network.setInput(img_caffe_blob)
            inferences: np.ndarray = self.face_detection_network.forward()

            # Parsing inferences informed by:
            # https://answers.opencv.org/question/208419/can-someone-explain-the-output-of-forward-in-dnn-module/
            confidences: Sequence[float] = inferences[0, 0, :, 2]
            y_scale, x_scale, _ = img.shape
            # 1. Extract box coordinates (all between 0 and 1)
            #    (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
            # 2. Scale them up to original image size
            # 3. Round values and convert to int
            # 4. Collect to Box
            boxes: List[Box] = [
                Box.from_values(*values)
                for values in (
                    (
                        inferences[0, 0, :, 3:7]
                        * np.array([x_scale, y_scale, x_scale, y_scale])
                    )
                    .round()
                    .astype(int)
                )
            ]

            for confidence, box in zip(confidences, boxes):
                if confidence >= self.detection_threshold:
                    cv.rectangle(
                        img=out_img,
                        pt1=box.top_left,
                        pt2=box.bottom_right,
                        color=Color.red().to_bgr(),
                        thickness=2,
                    )
            cv.imwrite(filename=f'{out_dir}{frame.name}', img=out_img)


if __name__ == '__main__':
    FaceTracker().track_faces('media/clip_2/', 'out/face_detector/clip_1/')
