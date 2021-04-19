import datetime
import os
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
        """
        Compute a similarity map for img such that similarity_map[y, x] is the "similarity score" for img[y, x].

        The "similarity score" is the match strength for placing top left corner of self.template at img[y, x] as
        computed by the Normalized Correlation Coefficient method. A higher value indicates a stronger match.

        The shape of similarity_map will be the shape of img, minus the shape of self.template, plus 1. This size
        reflects the possible locations for the placement of the top left corner of self.template within img without
        clipping over the edge.

        :param img: The image to compute the similarity map for
        :return: similarity_map
        """
        similarity_map: np.ndarray = cv.matchTemplate(
            img, self.template, cv.TM_CCOEFF_NORMED
        )

        return similarity_map

    def _get_match_xy_indices(
        self, similarity_map: np.ndarray
    ) -> Iterable[Tuple[int, int]]:
        """
        Return an iterable of the (x, y) coordinates of the valid matches in similarity_map, sorted in descending order.

        :param similarity_map: A similarity map as described in TemplateDetector._compute_similarity_map
        :return: The indices of the valid matches in similarity_map
        """
        # TODO: Prevent duplicate matches robustly instead of limiting to 1 match
        #  see https://stackoverflow.com/questions/21829469/removing-or-preventing-duplicate-template-matches-in-opencv-with-python

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

    def detect(
        self,
        img: np.ndarray,
        out_img: np.ndarray = None,
        color: Color = Color.yellow(),
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        Detect self.template in img and draw a box around it.

        :param img: The image to detect the template in
        :param out_img: The image to draw on (should be at least as large as img)
        :param color: The color of the bounding box
        :param line_thickness: The thickness of he bounding box line
        :return: out_img
        """

        # TODO: Make this scale invariant.
        #  see https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
        if out_img is None:
            out_img = img.copy()
        else:
            assert all(
                img_dim <= out_img_dim
                for img_dim, out_img_dim in zip(img.shape, out_img.shape)
            )

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
                color=color.to_bgr(),
                thickness=line_thickness,
            )

        return out_img


def run_template_detector(logo_path: str, clip_dir: str, out_base_dir: str) -> None:
    detector = TemplateDetector(template=cv.imread(logo_path), max_matches=1)
    out_dir: str = os.path.join(
        out_base_dir, datetime.datetime.now().strftime('%m-%d-%Y__%H-%M-%S')
    )
    os.makedirs(out_dir)
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
