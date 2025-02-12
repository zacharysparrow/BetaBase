## TODO
# major refactoring when everything works
# better to find still points directly instad of find peaks and infer still points
# need to do feet

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2

peak_width = 10
hold_dist_thresh = 0.01

df = pd.read_csv('pose_output.csv', dtype={'frame':int, 'landmark':str, 'x':float, 'y':float, 'z':float, 'visibility':float})
right_hand_data = df[df['landmark'].str.contains('RIGHT_INDEX')]
left_hand_data = df[df['landmark'].str.contains('LEFT_INDEX')]
right_foot_data = df[df['landmark'].str.contains('RIGHT_FOOT_INDEX')]
left_foot_data = df[df['landmark'].str.contains('LEFT_FOOT_INDEX')]

def calculate_velocity(coordinates, time_interval):
  positions = signal.savgol_filter(np.array([tuple(r) for r in coordinates.to_numpy()]), 10, 3, axis=0)
  velocities = []
  for i in range(len(positions) - time_interval):
    delta_x = positions[i + time_interval][0] - positions[i][0]
    delta_y = positions[i + time_interval][1] - positions[i][1]
#    delta_z = positions[i + time_interval][2] - positions[i][2]
#    delta_z = 0

    vx = delta_x / time_interval
    vy = delta_y / time_interval
#    vz = delta_z / time_interval
    velocities.append(np.linalg.norm((vx, vy)))
  return velocities

def find_body_positions(peak_coordinates, points, peak_width):
    body_pos = []
    for i in range(len(peak_coordinates) - 1):
        left_end = peak_coordinates[i] + peak_width
        right_end = peak_coordinates[i+1] - peak_width
        if left_end > right_end:
            left_end = round(np.mean([left_end, right_end])) - 1
            right_end = left_end + 2
        mean_pos = round(np.mean([left_end, right_end]))
        body_pos.append([mean_pos, np.mean(points[left_end:right_end], axis=0)])
    return body_pos

def remove_similar_points(points, threshold):
    filtered_points = []
    for point1 in points:
        similar_points = [point1]
        for point2 in points:
            if not point1.equals(point2) and np.linalg.norm(point1 - point2) < threshold:
                similar_points.append(point2)
                print("found dupe")
        avg_point = np.mean(similar_points, axis=0)
        filtered_points.append(avg_point)
    return np.unique(np.array(filtered_points, ndmin=2), axis=0)

time_interval = 3 #frames/2
right_hand_velocities = calculate_velocity(right_hand_data[["x","y"]], time_interval)
left_hand_velocities = calculate_velocity(left_hand_data[["x","y"]], time_interval)

right_hand_move_idx = signal.find_peaks(right_hand_velocities, height=0.004, prominence=0.004, distance=peak_width)[0]
if np.sign(np.diff(right_hand_velocities))[0] < 0:
    right_hand_move_idx = np.insert(right_hand_move_idx, 0, 0)
if np.sign(np.diff(right_hand_velocities))[-1] > 0:
    right_hand_move_idx = np.append(right_hand_move_idx, len(right_hand_data))

right_hand_still_data = find_body_positions(right_hand_move_idx, right_hand_data[["x","y"]], peak_width)
right_hand_still_idx = [i[0] for i in right_hand_still_data]
right_hand_still_pos = [i[1] for i in right_hand_still_data]
right_hand_still_pos = remove_similar_points(right_hand_still_pos, hold_dist_thresh)
right_hand_still_frames = [right_hand_data.frame.iloc[i] for i in right_hand_still_idx]
right_hand_still_vel = [right_hand_velocities[i] for i in right_hand_still_idx]

left_hand_move_idx = signal.find_peaks(left_hand_velocities, height=0.004, prominence=0.004, distance=peak_width)[0]
if np.sign(np.diff(left_hand_velocities))[0] < 0:
    left_hand_move_idx = np.insert(left_hand_move_idx, 0, 0)
if np.sign(np.diff(left_hand_velocities))[-1] > 0:
    left_hand_move_idx = np.append(left_hand_move_idx, len(left_hand_data))

left_hand_still_data = find_body_positions(left_hand_move_idx, left_hand_data[["x","y"]], peak_width)
left_hand_still_idx = [i[0] for i in left_hand_still_data]
left_hand_still_pos = [i[1] for i in left_hand_still_data]
left_hand_still_pos = remove_similar_points(left_hand_still_pos, hold_dist_thresh)
left_hand_still_frames = [left_hand_data.frame.iloc[i] for i in left_hand_still_idx]
left_hand_still_vel = [left_hand_velocities[i] for i in left_hand_still_idx]

right_foot_velocities = calculate_velocity(right_foot_data[["x","y"]], time_interval)
left_foot_velocities = calculate_velocity(left_foot_data[["x","y"]], time_interval)

print("Right hand holds:")
print(right_hand_still_pos)
print("Left hand holds:")
print(left_hand_still_pos)
#plt.plot(right_hand_data.frame, right_hand_data.y, label='Right')
#plt.plot(left_hand_data.frame, left_hand_data.y, label='Left')
#plt.title("Hand positions")
#plt.xlabel("Frame")
#plt.ylabel("y")
#plt.legend()
#plt.show()

plt.plot(right_hand_data.frame[:-time_interval], right_hand_velocities, label='Right')
plt.plot(left_hand_data.frame[:-time_interval], left_hand_velocities, label='Left')
plt.plot(right_hand_still_frames, right_hand_still_vel, 'o', label='Right still')
plt.plot(left_hand_still_frames, left_hand_still_vel, 'o', label='Left still')
plt.title("Hand velocity")
plt.xlabel("Frame")
plt.ylabel("Velocity")
plt.legend()
plt.show()

cap = cv2.VideoCapture("demo4.mp4")

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

    frame = cv2.resize(frame, (new_w, new_h))
    #Plot the points on the frame
    for p in right_hand_still_pos:
        px = int(new_w * p[0])
        py = int(new_h * p[1])
        cv2.circle(frame, [px, py], 5, (0, 0, 255))
    for p in left_hand_still_pos:
        px = int(new_w * p[0])
        py = int(new_h * p[1])
        cv2.circle(frame, [px, py], 5, (255, 0, 0))

    cv2.imshow('Right hand holds', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#plt.plot(right_foot_data.frame, right_foot_data.y, label='Right')
#plt.plot(left_foot_data.frame, left_foot_data.y, label='Left')
#plt.title("Feet positions")
#plt.xlabel("Frame")
#plt.ylabel("y")
#plt.legend()
#plt.show()
#
#plt.plot(right_foot_data.frame[:-time_interval], right_foot_velocities, label='Right')
#plt.plot(left_foot_data.frame[:-time_interval], left_foot_velocities, label='Left')
#plt.title("Foot velocity")
#plt.xlabel("Frame")
#plt.ylabel("Velocity")
#plt.legend()
#plt.show()
# it would be nice if we could set up a class so we could do stuff like this:
# left_hand_pos = pose_data.left_hand.pos()
# for the position at each frame, and
# right_hand_vel = pose_data.right_hand.velocity()
# to get the velocity of the right hand for each frame, etc.
