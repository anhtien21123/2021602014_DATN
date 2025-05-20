import pickle
import cv2
import numpy as np
import os
import sys

sys.path.append('../')
from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)
        cumulative_movement = [0, 0]  # Lưu tích lũy chuyển động camera

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        
        if old_features is None:
            # Nếu không tìm thấy features nào, trả về mặc định
            return camera_movement

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            
            # Tính optical flow
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)
            
            # Kiểm tra nếu không tìm thấy features
            if new_features is None:
                # Cập nhật old_gray và tiếp tục
                old_gray = frame_gray.copy()
                continue
                
            # Lọc ra các features tốt (status=1)
            good_old = old_features[status == 1]
            good_new = new_features[status == 1]
            
            # Kiểm tra nếu không có good features
            if len(good_old) == 0 or len(good_new) == 0:
                # Tìm features mới và tiếp tục
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                old_gray = frame_gray.copy()
                continue

            # Tính vector chuyển động trung bình của tất cả features
            movements_x = []
            movements_y = []
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                new_point = new.ravel()
                old_point = old.ravel()
                
                x_movement, y_movement = measure_xy_distance(old_point, new_point)
                
                # Chỉ xem xét chuyển động có ý nghĩa
                distance = np.sqrt(x_movement**2 + y_movement**2)
                if distance > self.minimum_distance and distance < 100:  # Lọc outliers
                    movements_x.append(x_movement)
                    movements_y.append(y_movement)
            
            # Nếu có đủ chuyển động có ý nghĩa
            if len(movements_x) > 5:
                # Loại bỏ outliers bằng cách sử dụng trung vị thay vì trung bình
                camera_movement_x = np.median(movements_x) 
                camera_movement_y = np.median(movements_y)
                
                # Cập nhật cumulative_movement
                cumulative_movement[0] += camera_movement_x
                cumulative_movement[1] += camera_movement_y
                
                # Lưu chuyển động tích lũy
                camera_movement[frame_num] = [cumulative_movement[0], cumulative_movement[1]]
            else:
                # Nếu không có đủ chuyển động, sử dụng giá trị của frame trước đó
                camera_movement[frame_num] = camera_movement[frame_num - 1]
            
            # Tìm features mới cho frame tiếp theo
            old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            if old_features is None:
                # Nếu không tìm thấy features, sử dụng giá trị của frame trước đó
                old_features = good_new.reshape(-1, 1, 2)
            
            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames