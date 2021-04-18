from typing import List, Tuple
import numpy as np
from joblib import load
import cv2
from sklearn.ensemble import AdaBoostClassifier

from model import Box, Gender

PATH = "./classifiers"
GCLF = None
EYE = None
MOUTH = None
H = 250
L = 250

def load_classifiers():
    """
    Loads AdaBoost Gender classifier and Haar Cascades feature classifiers
    and stores them into global variables
    """
    global GCLF, EYE, MOUTH

    GCLF = load(PATH + "/gclf.model")

    EYE = cv2.CascadeClassifier()
    MOUTH = cv2.CascadeClassifier()
    EYE.load(PATH + "/haarcascade_eye.xml")
    MOUTH.load(PATH + "/haarcascade_smile.xml")


def gender_faces(img: np.ndarray, face_boxes: List[Box]) -> List[Tuple[Gender, float]]:
    """
    Classify the genders of the faces in an image

    :param img: The image
    :param face_boxes: The bounding boxes of the faces in img
    :return: List of inferred genders and the confidence value parallel with face_boxes
    """

    if (GCLF == None):
        load_classifiers()

    ret = []
    for box in face_boxes:
        # Extract each face, and resize to (H,L)
        face = img[box.top_left_y:box.bottom_right_y, box.top_left_x:box.bottom_right_x,:]
        face = cv2.resize(face, (H,L))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Find 8 landmarks. Set NOSE to default as Nose classifier is now defunct
        # Values are found by averaging training data
        landmarks = np.array([103,147,126,126,114,114,138,160])

        # Find eyes. If search fails, use defaults
        try:
            eyes = EYE.detectMultiScale(gray, 1.3, 5)
            (lx, ly, lw, lh) = eyes[0]
            landmarks[0] = lx + lw//2
            landmarks[4] = ly + lh//2
            (rx, ry, rw, rh) = eyes[1]
            landmarks[1] = rx + rw//2
            landmarks[5] = ry + rh//2
        except:
            pass

        # Find mouth. If search fails, use defaults
        try:
            mouth = MOUTH.detectMultiScale(gray, 1.7, 5)[0]
            landmarks[3] = mouth[0] + mouth[2]//2
            landmarks[7] = mouth[1] + mouth[3]//2
        except:
            pass

        # Classify
        X = np.concatenate((face.reshape(L*H*3), landmarks)).reshape(1,-1)
        y_prob = GCLF.predict_proba(X)[0]
        y_pred = GCLF.predict(X).astype(int)[0]

        ret.append((Gender.MALE if y_pred == 1 else Gender.FEMALE, y_prob[y_pred]))

    return ret


        



        



    

