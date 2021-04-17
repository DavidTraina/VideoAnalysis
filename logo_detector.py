import os
import time
from typing import Iterator, Tuple, Iterable

import cv2 as cv
import numpy as np

from model import Color


class TemplateDetector:
    def __init__(
        self,
        template: np.ndarray,
        max_matches: int = None,
        similarity_threshold: float = 0.8,
    ):
        assert max_matches is None or max_matches >= 0
        self.template = template
        self.temp_h, self.temp_w, self.temp_d = template.shape
        self.max_matches = max_matches
        self.similarity_threshold = similarity_threshold

    def _compute_similarity_map(self, img: np.ndarray) -> np.ndarray:
        # similarity_map[y, x] is the "similarity score" for img[y, x]
        # "similarity score" is the match strength for placing top left corner of template at img[y, x]
        # Normalized Correlation Coefficient method yields best results
        similarity_map: np.ndarray = cv.matchTemplate(
            img, self.template, cv.TM_CCOEFF_NORMED
        )
        img_h, img_w, img_d = img.shape
        assert similarity_map.shape == (
            img_h - self.temp_h + 1,
            img_w - self.temp_w + 1,
        )

        return similarity_map

    def _get_match_xy_indices(
        self, similarity_map: np.ndarray
    ) -> Iterable[Tuple[int, int]]:
        # Set any pixel below similarity_threshold to 0 to disqualify it
        similarity_map[similarity_map < self.similarity_threshold] = 0
        num_valid_matches = np.count_nonzero(similarity_map)
        n_matches = min(num_valid_matches, self.max_matches)
        # Get the indices of the num_matches best matches in descending order of similarity
        raveled_match_indices = np.argsort(similarity_map, axis=None)[::-1][:n_matches]
        match_ys, match_xs = np.unravel_index(
            raveled_match_indices, similarity_map.shape
        )

        return zip(match_xs, match_ys)

    def detect(self, img: np.ndarray, out_img: np.ndarray = None) -> np.ndarray:
        if out_img is None:
            out_img = img.copy()

        similarity_map: np.ndarray = self._compute_similarity_map(img)
        match_xy_indices: Iterable[Tuple[int, int]] = self._get_match_xy_indices(
            similarity_map
        )

        # For each match, draw the bounding box
        for x, y in match_xy_indices:
            top_left = x, y
            bottom_right = x + self.temp_w, y + self.temp_h
            cv.rectangle(
                img=out_img,
                pt1=top_left,
                pt2=bottom_right,
                color=Color.yellow().to_bgr(),
                thickness=2,
            )

        return out_img


def run_template_detector(logo_path: str, clip_dir: str, out_base_dir: str):
    detector = TemplateDetector(template=cv.imread(logo_path), max_matches=1)
    out_dir = f'{out_base_dir}{round(time.time())}/'
    os.mkdir(out_dir)
    # noinspection PyTypeChecker
    dir_iter: Iterator[os.DirEntry] = os.scandir(clip_dir)
    for img_dir_entry in dir_iter:
        img = cv.imread(img_dir_entry.path)
        out_img = detector.detect(img=img)
        cv.imwrite(filename=f'{out_dir}{img_dir_entry.name}', img=out_img)


if __name__ == '__main__':
    run_template_detector(
        logo_path='media/logos/nbc.jpg',
        clip_dir='media/clip_1/',
        out_base_dir='out/logo-detector/clip_1/',
    )
