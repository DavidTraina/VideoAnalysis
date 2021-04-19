import os
from itertools import zip_longest, count
from typing import List, Dict, Optional, Iterator

import cv2 as cv
import dlib
import numpy as np

from face_tracker import TrackedFace
from model import Point, Box
from utils import write_boxes, create_output_dir, load_and_sort_dir


def track_faces(
    self,
    clip_dir: str,
    out_base_dir: str,
    draw_on_dir: str = None,
    detect_only: bool = False,
):
    """
    This is NOT recognition
    Tracking should be based on smooth object motion, not face recognition

    Steps Every frame:

    detect faces
    for old-face in tracked-faces:
        for new-face in detected-faces:
            if new-face in old-face-region and old-face in new-face-region:
                match new-face and old-face
                break
        if old-face not in matches and tracker.update(img) > thresh:
            match to tracked location

    for new-face not in matches:
        create new tracked-face
    """
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

    prev_img = None

    # Iterate Through Video Frames
    for frame, draw_on_frame in zip_longest(frames, draw_on_frames):
        # Read Images
        # read image to process
        img: np.ndarray = cv.imread(frame.path)
        # read image to draw on (if different)
        out_img = img.copy() if draw_on_frame is None else cv.imread(draw_on_frame.path)
        # ensure out_img is at least as large as img
        assert len(img.shape) == len(out_img.shape) and all(
            out_dim >= in_dim for in_dim, out_dim in zip(img.shape, out_img.shape)
        )

        detected_face_boxes: List[Box] = self.detect_faces(img)

        # If tracking is disabled, draw the boxes and move to next frame
        if detect_only:
            write_boxes(
                out_path=os.path.join(out_dir, frame.name),
                out_img=out_img,
                boxes=detected_face_boxes,
            )
            continue

        current_ids_to_detection_idx: Dict[int, Optional[int]] = {}
        lost_tracked_face_ids: List[int] = []

        # Iterate over the known (tracked) faces
        for tracked_face in tracked_faces.values():
            # Update the tracker with the new image
            # Tracker generates new predicted_rect from previous predicted_rect
            # Tracker returns its confidence that the face is inside new predicted_rect
            predicted_rect_confidence: float = tracked_face.tracker.update(img)
            if predicted_rect_confidence < self.tracking_threshold:
                # We've lost te object. Maybe due to a cut. Can't simply look for closest faces.
                # We assume the face is no longer present in img and stop tracking it
                print(
                    f"Too low: id={tracked_face.id_}, conf={predicted_rect_confidence}, frame={frame.name}"
                )
                lost_tracked_face_ids.append(tracked_face.id_)
                # TODO: In this case, maybe matchTemplate with found faces to see if one is above thresh
                continue
            predicted_rect: dlib.rectangle = tracked_face.tracker.get_position()
            tracked_last_rect: dlib.rectangle = tracked_face.box.to_dlib_rect()

            # Iterate over newly detected faces
            for detected_i, detected_face_box in enumerate(detected_face_boxes):

                # TODO Maybe just do distance based
                #  add confidence here?
                #  I think track motion and distance
                detected_rect = detected_face_box.to_dlib_rect()

                if (
                    # TODO: verify these are good checks. Maybe check that the l2 dist is minimal instead
                    #  need to make sure not modifying tracked faces as we go if we start computing minimums
                    #  THEY ARENT
                    # sanity check: face hasn't moved too much
                    tracked_last_rect.contains(detected_rect.center())
                    and detected_rect.contains(tracked_last_rect.center())
                    # sanity check: tracker prediction isn't too far from detection
                    and detected_rect.contains(predicted_rect.center())
                    and predicted_rect.contains(detected_rect.center())
                ):

                    # detected_face_box and tracked_face are the same face
                    # tracker was already update to this location
                    if tracked_face.id_ in current_ids_to_detection_idx:
                        print(
                            f'[ERROR]  {tracked_face.id_} found multiple times. Keeping first match'
                        )
                    else:
                        tracked_face.box = detected_face_box
                        current_ids_to_detection_idx[tracked_face.id_] = detected_i
                        new_tracker = dlib.correlation_tracker()
                        new_tracker.start_track(image=img, bounding_box=detected_rect)
                        tracked_face.tracker = new_tracker

            if tracked_face.id_ not in current_ids_to_detection_idx:
                assert predicted_rect_confidence >= self.tracking_threshold
                # Didn't detect this face, but tracker is confident it is at the predicted location.
                # We assume detector gave false negative
                tracked_face.box = Box.from_dlib_rect(predicted_rect)
                # tracker was updated to predicted_rect in update() call in condition
                current_ids_to_detection_idx[tracked_face.id_] = None

        # Remove lost face ids
        for lost_tracked_face_id in lost_tracked_face_ids:
            del tracked_faces[lost_tracked_face_id]

        tracked_detection_idxs = current_ids_to_detection_idx.values()

        # Track new faces
        for detected_i, detected_face_box in enumerate(detected_face_boxes):
            if detected_i not in tracked_detection_idxs:
                # Assume new face has entered frame and start tracking it
                id_ = next(new_face_id)
                tracker: dlib.correlation_tracker = dlib.correlation_tracker()
                tracker.start_track(
                    image=img, bounding_box=detected_face_box.to_dlib_rect()
                )
                tracked_faces[id_] = TrackedFace(
                    id_=id_, box=detected_face_box, tracker=tracker
                )
                current_ids_to_detection_idx[id_] = detected_i

        tracked_detection_idxs = current_ids_to_detection_idx.values()
        assert all(i in tracked_detection_idxs for i in range(len(detected_face_boxes)))
        assert len(current_ids_to_detection_idx) == len(tracked_faces)

        write_boxes(
            out_path=os.path.join(out_dir, frame.name),
            out_img=out_img,
            boxes=[face.box for face in tracked_faces.values()],
            labelss=[
                [(f'Person {face.id_}', Point(1, -9))]
                for face in tracked_faces.values()
            ],
        )

        # print(
        #     f"Processed {frame.name}.  Currently tracking {len(tracked_faces)} faces"
        # )
