import cv2

from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp

model_path = "pose_landmarker_heavy.task"
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.25,
    min_tracking_confidence=0.25,
    output_segmentation_masks=False,
    running_mode=VisionRunningMode.VIDEO,
)

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    # pose_landmarks_list = detection_result
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

cap = cv2.VideoCapture("demo.mp4")
#from mediapipe import solutions

# checks whether frames were extracted 
success = 1
with PoseLandmarker.create_from_options(options) as landmarker:
    while success:
        
        success, frame = cap.read()
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result =landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        annotated_image = draw_landmarks_on_image(frame, detection_result)
        cv2.imshow('frame', annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
