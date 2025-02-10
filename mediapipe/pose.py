import cv2
import csv
import mediapipe as mp

output_csv = 'pose_output.csv'

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2,model_complexity=2,smooth_landmarks=True,enable_segmentation=False)

def write_landmarks_to_csv(landmarks, frame_number, csv_data):
#    print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
#        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z, landmark.visibility])
#    print("\n")

csv_data = []
frame_number = 0
cap = cv2.VideoCapture('demo4.mp4')

success, frame = cap.read()
height, width, layers = frame.shape

max_height = 800
max_width = 1280
if height > max_height:
    new_h = max_height
    new_w = int(width * (max_height / height))
else:
    new_h = height
    new_w = width
if new_w > max_width:
    new_w = max_width
    new_h = int(height * (max_width / width))

cap.set(cv2.CAP_PROP_POS_FRAMES,0)

while success:
    success, frame = cap.read()
    if not success:
        break

    if frame_number % 2 == 0:
        frame = cv2.resize(frame, (new_w, new_h))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)

        if hasattr(pose_results.pose_landmarks, 'landmark'):
            write_landmarks_to_csv(pose_results.pose_landmarks.landmark, frame_number, csv_data)
            for landmark in pose_results.pose_landmarks.landmark: #uncomment to visualize all landmarks
                if landmark.visibility > 0.2:
                    landmark.visibility = 1.0
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Output', frame)
    frame_number += 1

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)
