from typing import List
import numpy as np

from model import Box, Gender


def gender_faces(img: np.ndarray, face_boxes: List[Box]) -> List[Gender]:
    """
    Classify the genders of the faces in an image

    :param img: The image
    :param face_boxes: The bounding boxes of the faces in img
    :return: List of inferred genders parallel with face_boxes
    """
    pass
