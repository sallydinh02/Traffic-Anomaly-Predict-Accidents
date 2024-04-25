from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
from skimage.measure import label
import os
import glob

alpha = 0.01
diff = 99
M = 13100


def extract_motion_mask(video_path):
    cap = cv2.VideoCapture(str(video_path))

    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
    tqdm.write(f'FPS = {cap.get(cv2.CAP_PROP_FPS)}')

    gap = fps // 5

    last_frame = None
    out = 0

    for idx in tqdm(range(total_frame_count), leave=False):
        # cap.set(cv2.CAP_PROP_POS_MSEC, 0.2 * 1000 * idx)
        ret, frame = cap.read()

        if not ret:
            break

        if idx % gap == 0:
            if idx % (gap * 5) == 0:
                last_frame = frame

            diff_frame = cv2.subtract(frame, last_frame)
            diff_frame = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
            _, diff_mask = cv2.threshold(diff_frame, 85, 255, cv2.THRESH_BINARY)
            diff_mask[diff_mask==255] = 1

            if np.sum(diff_mask) <= M:
                out = cv2.bitwise_or(out, diff_frame)
                out = cv2.medianBlur(out, 3)
                out = cv2.GaussianBlur(out, (3, 3), 9)
                _, out = cv2.threshold(out, 84, 255, cv2.THRESH_BINARY)

    min_area = 10000
    mask = label(out, connectivity=1)
    num = np.max(mask)
    for i in range(1, int(num + 1)):
        if np.sum(mask == i) < min_area:
            mask[mask == i] = 0

    mask = np.expand_dims((mask > 0).astype(float), axis=-1)
    return mask

# For local folder
#TEST_FOLDER = Path('D:/dataset/aic19-track3-test-data')
#MASK_DIR = Path('D:/motion draft')

#MASK_DIR.mkdir(parents=True, exist_ok=True)

# For local folder
#for filename in os.scandir(TEST_FOLDER):
    #if filename.is_file():
        #mask=extract_motion_mask(filename.path)
        #newfile=str(MASK_DIR)+"/"+str(filename).split(" ")[1].split(".")[0].split("'")[1]+".jpg"
        #cv2.imwrite(newfile, mask*255)


TEST_FOLDER="/content/gdrive/MyDrive/Colab Notebooks/New anomaly videos Youtube/*.mp4"
MASK_DIR=Path("/content/gdrive/MyDrive/Colab Notebooks/Motion mask 16")

MASK_DIR.mkdir(parents=True, exist_ok=True)

for file in glob.glob(TEST_FOLDER):
    vidname=str(file).split(".")[0].split("/")[-1]
    mask=extract_motion_mask(file)
    maskImg=str(MASK_DIR)+"/"+vidname+".jpg"
    cv2.imwrite(maskImg, mask*255)