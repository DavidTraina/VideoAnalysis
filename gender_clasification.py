from typing import List

import cv2
import numpy as np
from joblib import load
from imutils import face_utils
from sklearn.ensemble import AdaBoostClassifier
from model import Gender, Face, GenderedFace, Box

MODEL_PATH = "models/gender_classification/gclf.model"
GENDER_CLASSIFIER: AdaBoostClassifier = load(MODEL_PATH)
H = 250
L = 250


def gender_faces(img: np.ndarray, faces: List[Face]) -> List[GenderedFace]:
    """
    Classify the genders of the faces in an image

    :param img: The image
    :param faces: The bounding boxes of the faces in img
    :return: List of inferred genders and the confidence value parallel with face_boxes
    """

    ret = []
    pad_width = max(img.shape) + 100 # TODO: Make this smarter
    img = cv2.copyMakeBorder(
        img, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_REFLECT_101
    )
    for face in faces:
        # Extract each face, and resize to (H,L)
        center_x, center_y = face.box.center
        h = face.box.bottom_right_y - face.box.top_left_y
        w = face.box.bottom_right_x - face.box.top_left_x
        side_len = max(w, h)
        radius = side_len / 2
        box = Box.from_values(
            center_x - radius + pad_width,
            center_y - radius + pad_width,
            center_x + radius + pad_width,
            center_y + radius + pad_width,
        )
        face_crop = img[
            box.top_left_y : box.bottom_right_y, box.top_left_x : box.bottom_right_x, :
        ]
        face_crop = cv2.resize(face_crop, (H, L))

        # Extract features. If features are not found, use defaults.
        feat = face_utils.shape_to_np(face.shape)
        landmarks = np.zeros(8)
        x = box.top_left_x
        y = box.top_left_y
        landmarks[0] = feat[3][0] - feat[2][0] - x
        landmarks[1] = feat[1][0] - feat[0][0] - x
        landmarks[2] = feat[4][0] - x
        landmarks[3] = feat[4][0] - x
        landmarks[4] = feat[3][1] - feat[2][1] - y
        landmarks[5] = feat[1][1] - feat[0][1] - y
        landmarks[6] = feat[4][1] - y - 5
        landmarks[7] = feat[4][1] - y + 5

        # Classify
        X = np.concatenate((face_crop.reshape(L * H * 3), landmarks)).reshape(1, -1)
        y_prob = GENDER_CLASSIFIER.predict_proba(X)[0]
        y_pred = GENDER_CLASSIFIER.predict(X).astype(int)[0]

        # Create GenderedFace object
        g_face = GenderedFace.from_face(
            face=face, gender=Gender(y_pred), gender_confidence=y_prob[y_pred]
        )
        ret.append(g_face)

    return ret
