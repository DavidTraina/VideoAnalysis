import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

PATH = "/home/lawrence/420/project/clip_1"
OFFSET = 22
THRESHOLD = 800

""" Reports found scene boundaries and produces a Displaced Frame Difference (DFD) 
graph against each frame in video.

Usage:
python3 shotDetector.py --path *path to video* --threshold *DFD threshold* [--offset *starting frame of clip*]
"""
def main():
    # Load first frame and extract image info
    files = sorted([f.path for f in os.scandir(PATH)])
    F_tm1 = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    H, L = F_tm1.shape
    idxs = np.mgrid[0:H, 0:L]

    # Set non-existent frames to 0 DFD
    DFD = [0 for i in range(OFFSET)]

    # For each pair of frames
    for t in range(1, len(files)):
        F_t = cv2.imread(files[t], cv2.IMREAD_GRAYSCALE)
        
        # Compute flow (as ints)
        flow = cv2.calcOpticalFlowFarneback(F_tm1, F_t, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = np.rint(flow).astype(int)

        # Compute new pixel locations
        y = np.clip(idxs[0] + flow[:,:,1], 0, H-1)
        x = np.clip(idxs[1] + flow[:,:,0], 0, L-1)
        
        # Calculate modified DFD
        DFD.append(np.mean(F_tm1 - F_t[y,x])*np.mean(np.abs(flow)))
        if DFD[-1] > THRESHOLD:
            print("Shot boundary detected at t = {}".format(OFFSET - 1 + t))
        
        # Set current frame as previous frame.
        F_tm1 = F_t

    plt.title("DFD* scores for clip 3")
    plt.ylabel("DFD*(t)")
    plt.xlabel("t")
    plt.plot(DFD)
    plt.plot([THRESHOLD for _ in range(len(DFD))])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to video clip", type=str)
    parser.add_argument("--threshold", required=True, help="Sensitivity of algorithm", type=int)
    parser.add_argument("--offset", default=0, help="Adjusts plotting to reflect missing frames at start of clip", type=int)

    args        = parser.parse_args()
    PATH        = args.path
    THRESHOLD   = args.threshold
    OFFSET      = args.offset

    main()