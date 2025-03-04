import time

import cv2

from rtmlib import PoseTracker, Wholebody, draw_skeleton, BodyWithFeet

# import numpy as np

device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

cap = cv2.VideoCapture('./demo_video.mp4')

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody = PoseTracker(
    Wholebody,
    det_frequency=100,
    to_openpose=openpose_skeleton,
    mode='performance',  # balanced, performance, lightweight
    backend=backend,
    device=device,
    tracking=False
    )

frame_idx = 0

if not cap.isOpened():
    print("Error opening video file")

while cap.isOpened():
    success, frame = cap.read()
    
    frame_idx += 1

    if not success:
        break
    s = time.time()
    keypoints, scores = wholebody(frame)
    det_time = time.time() - s
#    print('det: ', det_time)

    img_show = frame.copy()

    # if you want to use black background instead of original image,
    # img_show = np.zeros(img_show.shape, dtype=np.uint8)
    # print(scores)
    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=3.0)

#    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)
