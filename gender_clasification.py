from typing import List

import cv2
import numpy as np
from joblib import load
from imutils import face_utils

from model import Gender, Face, GenderedFace

PATH = "./gclf.model"
GCLF = None
H = 250
L = 250


def gender_faces(img: np.ndarray, faces: List[Face]) -> List[GenderedFace]:
    """
    Classify the genders of the faces in an image

    :param img: The image
    :param faces: The bounding boxes of the faces in img
    :return: List of inferred genders and the confidence value parallel with face_boxes
    """

    global GCLF
    if GCLF == None:
        GCLF = load(PATH + "/gclf.model")

    ret = []
    for face in faces:
        box = face.box

        # Extract each face, and resize to (H,L)
        face = img[
            box.top_left_y : box.bottom_right_y, box.top_left_x : box.bottom_right_x, :
        ]
        face = cv2.resize(face, (H, L))

        # Extract features. If features are not found, use defaults.
        feat = face_utils.shape_to_np(face.shape)
        landmarks = np.zeros(8)
        try:
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
        except:
            landmarks = np.array([103, 147, 126, 126, 114, 114, 138, 160])
        
        # Classify
        X = np.concatenate((face.reshape(L * H * 3), landmarks)).reshape(1, -1)
        y_prob = GCLF.predict_proba(X)[0]
        y_pred = GCLF.predict(X).astype(int)[0]

        # Create GenderedFace object
        g_face = GenderedFace(face)
        g_face.gender = Gender.MALE if y_pred == 1 else Gender.FEMALE
        g_face.gender_confidence = y_prob[y_pred]
        ret.append(g_face)

    return ret
