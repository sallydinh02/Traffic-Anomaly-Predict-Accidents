from pathlib import Path

import pandas as pd
import cv2
import numpy as np
import os


TEST_DIR=Path("/content/gdrive/MyDrive/Colab Notebooks/New anomaly videos Youtube")
DETECTION_DIR = Path("/content/gdrive/MyDrive/Colab Notebooks/Output detection")

min_detection_confidence = 0.2
#event_idle_time_max = 10 * 5  # 10 seconds
event_iou_continue_threshold = 0.3
#event_anomaly_min_time = 30 * 5  # 30 seconds
event_anomaly_min_conf = 0.65
#event_min_end_time = 20 * 5 # 20 seconds

event_idle_time_max = 8 * 5  # 8 seconds
event_min_end_time = 20 * 5 # 20 seconds
event_anomaly_min_time = 30 * 5  # 30 seconds


def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def area(box):
    return abs((box[2] - box[0]) * (box[3] - box[1]))


def get_fps(video_id):
    video_path = TEST_DIR / f'{video_id}.mp4'
    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps


def check_event(event):
    if event['last_update'] <= event_min_end_time:
        return False

    if event['last_update'] - event['start_time'] <= event_anomaly_min_time:
        return False

    if event['boxes'][-1][0] < 5 and area(event['boxes'][-1]) < 15 * 15:
        print(f'Small box killed')
        return False

    if np.mean(event['scores']) < event_anomaly_min_conf:
        return False

    return True


def detect_anomaly(video_id, raw=False):
    detection_csv = DETECTION_DIR / f'{video_id}.csv'

    if not detection_csv.exists():
        print(f'csv for video {video_id} not found!')
        return -1, -1

    fps = get_fps(video_id)
    gap = fps // 5

    df = pd.read_csv(detection_csv)
    df = df[df['score'] >= min_detection_confidence]

    events = []
    anomalies = []

    for frame_id, table in df.groupby(['frame_id']):
        time = int(frame_id) // gap

        new_candidates = []

        for x_min, y_min, x_max, y_max, score \
            in zip(table['x_min'], table['y_min'], table['x_max'], table['y_max'], table['score']):

            box = (x_min, y_min, x_max, y_max)

            max_iou = -1
            matched_event = None

            for event in events:
                if time - event['last_update'] > event_idle_time_max:
                    event['stale'] = True
                    if check_event(event):
                        anomalies.append(event)
                else:
                    iou_event = iou(box, event['boxes'][-1])
                    if iou_event > event_iou_continue_threshold:
                        if iou_event > max_iou:
                            max_iou = iou_event
                            matched_event = event

            if matched_event is not None:
                matched_event['last_update'] = time
                matched_event['boxes'].append(box)
                matched_event['scores'].append(score)
            else:
                new_candidates.append((box, score))

        for box, score in new_candidates:
            events.append({
                'start_time': time,
                'last_update': time,
                'boxes': [box],
                'scores': [score],
                'stale': False,
            })

        events = [event for event in events if not event['stale']]

    for event in events:
        if check_event(event):
            anomalies.append(event)

    if len(anomalies) > 0:
        soonest = 123456789
        conf = 123456789
        res_etime = None
        res_box = None
        for anomaly in anomalies:
            stime = int(anomaly['start_time'] * gap / fps) if not raw else int(anomaly['start_time'] * gap)
            etime = int(anomaly['last_update'] * gap / fps) if not raw else int(anomaly['last_update'] * gap)

            box = anomaly['boxes'][0]
            avg_score = np.mean(anomaly['scores'])
            print(
                f'Video {video_id}, anomaly start {stime}, end {etime}, len = {etime-stime}, box: {box}, score: {avg_score}'
            )
            if stime < soonest:
                soonest = stime
                conf = avg_score
                res_etime = etime
                res_box = box

        if raw:
            return soonest, res_etime, res_box, conf

        return soonest, conf
    else:
        print(f'Video {video_id}: no anomaly')
        return -1, -1


if __name__ == '__main__':
    print(f'Using min_detection_confidence = {min_detection_confidence}')
    with open('/content/gdrive/MyDrive/Colab Notebooks/submissions.txt', 'w') as f:
        for filename in os.scandir(TEST_DIR):
            video_id=str(filename).split(" ")[1].split(".")[0].split("'")[1]
            soonest, conf = detect_anomaly(video_id)
            if soonest != -1:
                print(video_id, soonest, conf, file=f)
