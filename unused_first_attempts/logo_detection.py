# TODO: restore to running state for demonstration

"""
This is the first attempt at logo detection. It is overly complicated and performs poorly.
"""
import os
import time
from dataclasses import dataclass, astuple
from typing import Optional, Tuple, List, Collection
import numpy as np
import cv2 as cv


@dataclass(frozen=True)
class Color:
    b: int
    g: int
    r: int


def mask_excluded_colors(
    img: np.ndarray, excluded_colors: Collection[Color]
) -> Optional[np.ndarray]:
    if excluded_colors is None or not len(excluded_colors):
        return None
    h, w, d = img.shape
    assert d == 3
    # OpenCV uses 255 for keep, 0 for discard: https://stackoverflow.com/a/45828856
    mask = np.full((h, w), 255)
    for y in range(h):
        for x in range(w):
            img_px_color = Color(*np.int32(img[y, x, :]))
            if img_px_color in excluded_colors:
                mask[y, x] = 0
    return mask


def find_object_corners(
    out_name: str,
    out_dir: str,
    obj: np.ndarray,
    img: np.ndarray,
    obj_mask: Optional[np.ndarray] = None,
    lowes_ratio_threshold: float = 0.75,
) -> Optional[Tuple[float, float, float, float]]:
    assert obj is not None
    assert img is not None
    assert obj_mask is None or obj.shape[:2] == obj_mask.shape

    """Compute SIFT Features"""
    # Initialize SIFT detector
    sift_detector: cv.SIFT = cv.SIFT_create()

    # Compute object features
    obj_keypoints: List[cv.KeyPoint]
    obj_descriptors: np.ndarray
    obj_keypoints, obj_descriptors = sift_detector.detectAndCompute(
        image=obj, mask=obj_mask
    )

    # Compute image features
    img_keypoints: List[cv.KeyPoint]
    img_descriptors: np.ndarray
    img_keypoints, img_descriptors = sift_detector.detectAndCompute(
        image=img, mask=None
    )

    """Match Features with FLANN"""
    # FLANN parameters from https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    # flann_matcher: cv.FlannBasedMatcher = cv.FlannBasedMatcher_create()
    flann_matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches_2nn: List[List[cv.DMatch]] = flann_matcher.knnMatch(
        queryDescriptors=obj_descriptors, trainDescriptors=img_descriptors, k=2
    )

    """Prune matches with Lowe's Ratio Test"""
    # Lowe's Ratio Test is an outlier removal technique.
    # It ensures unambiguous matches by verifying that the best match is significantly better than the runner up.
    # Described in https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf#page=20
    matches: List[cv.DMatch] = sorted(
        [
            best_match
            for best_match, second_best_match in matches_2nn
            if best_match.distance < lowes_ratio_threshold * second_best_match.distance
        ],
        key=lambda match: match.distance,
    )

    """Compute Projective Transformation"""
    if len(matches) < 3:
        # print(f'Need at least 3 matches to locate object. Only found {len(matches)}')
        return None, None

    # todo EXPERIMENTAL:
    matches = matches[:3]

    # 'Ptr<cv::UMat>' is functionally equivalent to np.float32() https://stackoverflow.com/a/55815108
    src_pts = np.float32([obj_keypoints[m.queryIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([img_keypoints[m.trainIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )
    projective_transformation_matrix: np.ndarray  # 3x3 projective transformation matrix: obj -> img
    if len(matches) == 3:
        # print(
        #     'Found 3 matches. Locating object via affine transformation extended to projective transformation'
        # )
        affine_warp_transformation_matrix: np.ndarray = cv.getAffineTransform(
            src=src_pts, dst=dst_pts
        )
        assert affine_warp_transformation_matrix.shape == (2, 3)
        # Extend affine transformation to full projective transformation by adding [0, 0, 1] as bottom row
        projective_transformation_matrix = np.vstack(
            [affine_warp_transformation_matrix, [0, 0, 1]]
        )
        inlier_mask = None
    else:
        # print(
        #     f'Found {len(matches)} matches. Locating object via homography projective transformation'
        # )
        # Compute via RANSAC the homography projective transformation transformation:  obj -> img
        projective_transformation_matrix, inlier_mask = cv.findHomography(
            srcPoints=src_pts,
            dstPoints=dst_pts,
            method=cv.RANSAC,
            ransacReprojThreshold=5.0,  # todo tune
        )
        inlier_mask = inlier_mask.ravel().tolist()
        # assert projective_transformation_matrix is not None

    obj_corners = None
    if projective_transformation_matrix is not None:
        """Transform corners of obj into img"""
        h, w, d = obj.shape
        # get corners
        obj_corners = np.float32(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
        ).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(obj_corners, projective_transformation_matrix)

        """Visualize"""
        boxed_obj_img = cv.polylines(
            img, [np.int32(dst)], True, astuple(Color(b=255, g=0, r=0)), 5, cv.LINE_AA
        )
    else:
        # print("Unable to locate object")
        boxed_obj_img = img

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=inlier_mask,  # draw only inliers
        flags=2,
    )
    boxed_obj_img = cv.drawMatches(
        obj, obj_keypoints, boxed_obj_img, img_keypoints, matches, None, **draw_params
    )

    # cv.imwrite(f'{out_dir}/{out_name}', boxed_obj_img)
    # rgb_boxed_obj_img: np.ndarray = cv.cvtColor(
    #     boxed_obj_img, cv.COLOR_BGR2RGB
    # )  # cv uses BGR, Matplotlib uses RGB
    # plt.imshow(rgb_boxed_obj_img)
    # plt.show()
    # print(f"inliers: {inlier_mask}")

    return obj_corners, len(matches)


def find_all_logos(logo_path):
    # TESTING
    found, not_found, affine, homo, total_matches = 0, 0, 0, 0, 0
    logo = cv.imread(logo_path)
    out_dir = f'out/logo-detector/{round(time.time())}/'
    os.mkdir(out_dir)
    for img_dir_entry in os.scandir('../media/clip_2/'):
        img = cv.imread(img_dir_entry.path)
        res, num_matches = find_object_corners(img_dir_entry.name, out_dir, logo, img)
        if res is not None:
            found += 1
            total_matches += num_matches
            if num_matches > 3:
                homo += 1
            else:
                assert num_matches == 3
                affine += 1
        else:
            not_found += 1
    with open(f'{out_dir}/results.txt', 'x') as f:
        f.write(
            f'*****\nlogo: {logo_path}\nFound: {found}, not_found: {not_found}, affine: {affine}, homo: {homo}, avg_matches_for_founds: {total_matches / found}'
        )


if __name__ == '__main__':
    # logo = cv.imread('media/logos/cropped_clip_1_logo.jpg')
    # img = cv.imread('media/clip_1/023.jpg')
    # logo_mask = None  # mask_excluded_colors(logo, [Color(0, 0, 0)])
    # find_object_corners(obj=logo, img=img, obj_mask=logo_mask)
    find_all_logos('../media/logos/clevver_news.jpg')
