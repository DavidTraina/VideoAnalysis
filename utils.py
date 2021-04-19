import datetime
import os
import shutil
from itertools import zip_longest
from typing import List, Iterable, Optional, Tuple

import cv2 as cv
from natsort import natsorted
import numpy as np

from model import Box, Color, Point


def create_output_dir(
    out_dir: str, generate_basename: bool = True, overwrite: bool = False
) -> str:
    basename = (
        datetime.datetime.now().strftime('%m-%d-%Y__%H-%M-%S')
        if generate_basename
        else ''
    )
    out_dir = os.path.join(out_dir, basename)

    # handle already exists case
    if os.path.exists(out_dir):
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f'{out_dir} already exists and overwrite=False')

    # create directory
    os.makedirs(out_dir)
    return out_dir


def load_and_sort_dir(directory: str) -> List[os.DirEntry]:
    return (
        natsorted(os.scandir(directory), key=lambda dir_entry: dir_entry.name)
        if directory
        else []
    )


def translate_point(
    point: Point, translation: Point, min_point: Point = Point(0, 0)
) -> Point:
    return Point(
        x=max(point.x + translation.x, min_point.x),
        y=max(point.y + translation.y, min_point.y),
    )


def draw_box(
    out_img: np.ndarray,
    box: Box,
    labels: Iterable[Tuple[str, Point]],
    color: Color = Color.red(),
    line_thickness: int = 2,
) -> np.ndarray:
    cv.rectangle(
        img=out_img,
        pt1=box.top_left,
        pt2=box.bottom_right,
        color=color.to_bgr(),
        thickness=line_thickness,
    )
    for text, translation in labels:
        text_loc: Point = translate_point(
            Point(box.top_left_x, box.bottom_right_y), translation
        )
        cv.putText(
            img=out_img,
            text=text,
            org=text_loc,
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=Color.orange().to_bgr(),
            thickness=2,
        )

    return out_img


def write_boxes(
    out_path: str,
    out_img: np.ndarray,
    boxes: Iterable[Box],
    labelss: Iterable[Iterable[Tuple[str, Point]]] = tuple(),
    color: Color = Color.red(),
    line_thickness: int = 2,
):
    for box, labels in zip_longest(boxes, labelss):
        draw_box(out_img, box, labels, color, line_thickness)

    cv.imwrite(filename=out_path, img=out_img)
