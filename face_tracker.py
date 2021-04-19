import os
import sys
from dataclasses import dataclass
from itertools import count, zip_longest
from typing import Iterator, List, Sequence, Dict, Set

import cv2 as cv
import dlib
import numpy as np

from gender_clasification import gender_faces
from logo_detector import run_template_detector
from model import Color, Box, Point, Face, GenderedFace
from utils import create_output_dir, load_and_sort_dir, write_boxes

DETECTION_MODEL = "models/face_detection/res10_300x300_ssd_iter_140000.caffemodel"
DETECTION_MODEL_CONFIG = "models/face_detection/deploy.prototxt.txt"
RECOGNITION_MODEL = 'models/face_recognition/dlib_face_recognition_resnet_model_v1.dat'
SHAPE_MODEL = 'models/face_recognition/shape_predictor_5_face_landmarks.dat'
DETECTION_NETWORK_INPUT_SIZE = 300, 300
# Value from https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c
BLOB_MEAN_SUBTRACTION: Color = Color(r=123, b=104, g=117)

"""
Challenge track robust to cuts
"""


@dataclass
class TrackedFace(GenderedFace):
    id_: int = None
    tracker: dlib.correlation_tracker = None
    staleness: int = 0


class FaceTracker:
    def __init__(
        self,
        detection_threshold: float = 0.5,
        recognition_threshold: float = 0.55,
        remember_identities: bool = False,
        tracking_threshold: float = 18.0,
        tracking_expiry: int = 3,
    ):
        # tracking_threshold maximum = 31.0 from testing
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        self.remember_identities = remember_identities
        self.tracking_threshold = tracking_threshold
        self.tracking_expiry = tracking_expiry
        self.face_detection_network = cv.dnn.readNetFromCaffe(
            prototxt=DETECTION_MODEL_CONFIG,
            caffeModel=DETECTION_MODEL,
        )
        self.face_shape_predictor: dlib.shape_predictor = dlib.shape_predictor(
            SHAPE_MODEL
        )
        self.face_recognition_model = dlib.face_recognition_model_v1(RECOGNITION_MODEL)
        self.tracking_expiry = tracking_expiry

    def faces_match(self, face_1: Face, face_2: Face) -> bool:

        diff: np.ndarray = face_1.descriptor - face_2.descriptor
        dist: float = np.linalg.norm(diff)
        return dist < self.recognition_threshold

    def recognize_face(self, img: np.ndarray, face_box: Box) -> Face:
        shape: dlib.full_object_detection = self.face_shape_predictor(
            img, face_box.to_dlib_rect()
        )
        descriptor: np.ndarray = np.asarray(
            self.face_recognition_model.compute_face_descriptor(img, shape)
        )
        return Face(box=face_box, shape=shape, descriptor=descriptor)

    def detect_face_boxes(self, img: np.ndarray) -> List[Box]:
        img_300x300: np.ndarray = cv.resize(src=img, dsize=DETECTION_NETWORK_INPUT_SIZE)
        img_caffe_blob: np.ndarray = cv.dnn.blobFromImage(
            image=img_300x300,
            scalefactor=1.0,
            size=DETECTION_NETWORK_INPUT_SIZE,
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

        return [
            box
            for confidence, box in zip(confidences, boxes)
            if confidence >= self.detection_threshold
        ]

    def track_faces(
        self,
        clip_dir: str,
        out_base_dir: str,
        draw_on_dir: str = None,
        detect_only: bool = False,
    ):
        # Setup
        # load image paths
        frames: List[os.DirEntry] = load_and_sort_dir(clip_dir)
        draw_on_frames: List[os.DirEntry] = load_and_sort_dir(draw_on_dir)
        assert len(draw_on_frames) in (0, len(frames))

        # create output directory
        out_dir: str = create_output_dir(out_base_dir)

        # initialize variables required for object tracking
        new_face_id: Iterator[int] = count(start=1)
        tracked_faces: Dict[int, TrackedFace] = {}

        # Iterate Through Video Frames
        for frame, draw_on_frame in zip_longest(frames, draw_on_frames):
            # load new frame
            img = cv.imread(frame.path)

            # load out_img
            out_img: np.ndarray = (
                img.copy() if draw_on_frame is None else cv.imread(draw_on_frame.path)
            )

            # ensure out_img is at least as large as img
            assert len(img.shape) == len(out_img.shape) and all(
                out_dim >= in_dim for in_dim, out_dim in zip(img.shape, out_img.shape)
            )

            detected_face_boxes: List[Box] = self.detect_face_boxes(img)

            # If tracking is disabled, draw the boxes and move to next frame
            if detect_only:
                write_boxes(
                    out_path=os.path.join(out_dir, frame.name),
                    out_img=out_img,
                    boxes=detected_face_boxes,
                )
                continue

            detected_faces: List[GenderedFace] = gender_faces(
                img=img,
                faces=[
                    self.recognize_face(img, detected_face_box)
                    for detected_face_box in detected_face_boxes
                ],
            )

            current_face_ids: Set[int] = set()
            lost_face_ids: Set[int] = set()

            # Iterate over the known (tracked) faces
            for tracked_face in tracked_faces.values():
                matched_detected_faces: List[GenderedFace] = [
                    detected_face
                    for detected_face in detected_faces
                    if self.faces_match(tracked_face, detected_face)
                ]

                if not matched_detected_faces:
                    # Tracked face was not matched to and detected face
                    # Increment staleness since we didn't detect this face
                    tracked_face.staleness += 1
                    # Update tracker with img and get confidence
                    tracked_confidence: float = tracked_face.tracker.update(img)
                    if (
                        tracked_face.staleness < self.tracking_expiry
                        and tracked_confidence >= self.tracking_threshold
                    ):
                        # Assume face is still in frame but we failed to detect
                        # Update box with predicted location box
                        predicted_box: Box = Box.from_dlib_rect(
                            tracked_face.tracker.get_position()
                        )
                        tracked_face.box = predicted_box
                        current_face_ids.add(tracked_face.id_)
                    else:
                        # Assume face has left frame because either it is too stale or confidence is too low
                        if self.remember_identities:
                            # Set effectively infinite staleness to force tracker reset if face is found again later
                            tracked_face.staleness = sys.maxsize
                        else:
                            lost_face_ids.add(tracked_face.id_)
                    continue

                # Tracked face was matched to one or more detected faces
                # Multiple matches should rarely happen if faces in frame are distinct. We take closest to prev location
                # TODO: Handle same person multiple times in frame
                matched_detected_face = min(
                    matched_detected_faces,
                    key=lambda face: tracked_face.box.distance_to(face.box),
                )
                # Update tracked_face
                tracked_face.descriptor = matched_detected_face.descriptor
                tracked_face.shape = matched_detected_face.descriptor
                tracked_face.box = matched_detected_face.box
                if tracked_face.staleness >= self.tracking_expiry:
                    # Face was not present in last frame so reset tracker
                    tracked_face.tracker = dlib.correlation_tracker()
                    tracked_face.tracker.start_track(
                        image=img, bounding_box=tracked_face.box.to_dlib_rect()
                    )
                else:
                    # Face was present in last frame so just update guess
                    tracked_face.tracker.update(
                        image=img, guess=tracked_face.box.to_dlib_rect()
                    )
                tracked_face.staleness = 0
                tracked_face.gender = matched_detected_face.gender
                tracked_face.gender_confidence = matched_detected_face.gender_confidence
                # Add tracked_face to current_ids to reflect that it is in the frame
                current_face_ids.add(tracked_face.id_)
                # remove matched_detected_face from detected_faces
                detected_faces.remove(matched_detected_face)

            # Delete all faces that were being tracked but are now lost
            # lost_face_ids will always be empty if self.remember_identities is True
            for id_ in lost_face_ids:
                del tracked_faces[id_]

            for new_face in detected_faces:
                # This is a new face (previously unseen)
                id_ = next(new_face_id)
                tracker: dlib.correlation_tracker = dlib.correlation_tracker()
                tracker.start_track(image=img, bounding_box=new_face.box.to_dlib_rect())
                tracked_faces[id_] = TrackedFace(
                    box=new_face.box,
                    descriptor=new_face.descriptor,
                    shape=new_face.shape,
                    id_=id_,
                    tracker=tracker,
                    gender=new_face.gender,
                    gender_confidence=new_face.gender_confidence,
                )
                current_face_ids.add(id_)

            write_boxes(
                out_path=os.path.join(out_dir, frame.name),
                out_img=out_img,
                boxes=[tracked_faces[id_].box for id_ in current_face_ids],
                labelss=[
                    [
                        (
                            f'Person {id_}',
                            Point(3, 14),
                        ),
                        (
                            f'{tracked_faces[id_].gender.name[0].upper()}: {round(100 * tracked_faces[id_].gender_confidence, 1)}%',
                            Point(3, 30),
                        ),
                    ]
                    for id_ in current_face_ids
                ],
                color=Color.yellow(),
            )

            print(
                f"Processed {frame.name}.  Currently tracking {len(tracked_faces)} faces"
            )
        return out_dir


if __name__ == '__main__':
    # Hacky demo to generate combined result
    for logo, clip in (
        ('flicks_and_the_city.jpg', 'clip_3'),
        ('nbc.jpg', 'clip_1'),
        ('clevver_news.jpg', 'clip_2'),
    ):
        face_out_dir = FaceTracker().track_faces(
            clip_dir=f'media/{clip}', out_base_dir=f'out/face_detector_final/{clip}/'
        )
