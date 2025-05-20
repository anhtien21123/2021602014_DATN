from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self, model_path):
        # Khởi tạo YOLO model với các tham số tối ưu cho phát hiện cầu thủ
        self.model = YOLO(model_path)
        
        # Cấu hình các tham số phát hiện
        self.model.overrides['conf'] = 0.05  # Ngưỡng tin cậy thấp để phát hiện nhiều đối tượng hơn
        self.model.overrides['iou'] = 0.3    # Ngưỡng IoU thấp hơn để không bỏ sót cầu thủ
        self.model.overrides['max_det'] = 100  # Tăng số lượng phát hiện tối đa trong mỗi frame
        
        # Khởi tạo tracker mặc định không có tham số
        # Phiên bản ByteTrack trong supervision của bạn không chấp nhận các tham số đó
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # Sử dụng các tham số cụ thể để cải thiện phát hiện
            detections_batch = self.model.predict(
                frames[i:i + batch_size],
                conf=0.05,           # Ngưỡng tin cậy thấp để phát hiện nhiều đối tượng hơn
                iou=0.3,             # Ngưỡng IoU thấp hơn để không bỏ sót cầu thủ
                max_det=100,         # Tăng số lượng phát hiện tối đa trong mỗi frame
                classes=[0, 1, 2]    # Chỉ phát hiện player(0), goalkeeper(1), referee(2)
            )
            detections += detections_batch
        return detections

    def clean_tracks(self, tracks):
        """
        Làm sạch dữ liệu theo dõi để loại bỏ các phát hiện trùng lặp hoặc không hợp lệ
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                # Các ID đã xử lý để tránh trùng lặp
                processed_ids = set()
                
                # Danh sách tạm thời các track hợp lệ cho frame này
                valid_tracks = {}
                
                for track_id, track_info in list(track.items()):
                    # Nếu không có bbox hoặc bbox không hợp lệ, bỏ qua
                    bbox = track_info.get('bbox', [])
                    
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                        
                    try:
                        # Đảm bảo mọi giá trị trong bbox đều là số
                        x1, y1, x2, y2 = map(float, bbox)
                        
                        # Loại bỏ các bbox không hợp lệ (kích thước quá nhỏ hoặc giá trị âm)
                        if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0 or (x2 - x1) < 5 or (y2 - y1) < 5:
                            continue
                            
                        # Đảm bảo bbox có định dạng đúng
                        track_info['bbox'] = [float(x1), float(y1), float(x2), float(y2)]
                        
                        # Thêm vào danh sách hợp lệ
                        if track_id not in processed_ids:
                            valid_tracks[track_id] = track_info
                            processed_ids.add(track_id)
                            
                    except (ValueError, TypeError):
                        # Bỏ qua các bbox có giá trị không phải số
                        continue
                
                # Cập nhật lại dữ liệu trong frame
                tracks[object_type][frame_num] = valid_tracks
                
        return tracks

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Phát hiện đối tượng trong các frames
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # Khởi tạo danh sách tracks cho mỗi frame
        for _ in range(len(frames)):
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

        # Xử lý từng frame
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Chuyển đổi sang định dạng supervision Detection
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Chuyển đổi goalkeeper -> player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if class_id in cls_names and cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Cập nhật các tracks với detections mới
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Lấy thông tin cho players và referees
            if detection_with_tracks is not None and len(detection_with_tracks) > 0:
                for det in detection_with_tracks:
                    if len(det) >= 5:  # Đảm bảo có đủ thông tin
                        bbox = det[0].tolist()
                        cls_id = det[3]
                        track_id = det[4]

                        if cls_id == cls_names_inv['player']:
                            tracks["players"][frame_num][track_id] = {"bbox": bbox}

                        if cls_id == cls_names_inv['referee']:
                            tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Lấy thông tin cho ball (không sử dụng tracker cho ball)
            for det in detection_supervision:
                if det[3] == cls_names_inv['ball']:
                    bbox = det[0].tolist()
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Làm sạch dữ liệu
        tracks = self.clean_tracks(tracks)

        # Lưu stub nếu cần
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectagle
        overlay = frame.copy()
        
        # Draw Ball Control Bar
        cv2.rectangle(overlay, (1380, 800), (1900, 980), (255, 255, 255), -1)
        
        # Add text
        cv2.putText(overlay, "Ball Control", (1550, 830), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Calculate Team Ball Control
        team_1_id = 1
        team_2_id = 2
        
        # Convert to strings and integers for proper comparison
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        
        # Handle both string and integer team IDs
        team_1_num_frames = np.sum(team_ball_control_till_frame == str(team_1_id)) + np.sum(team_ball_control_till_frame == team_1_id)
        team_2_num_frames = np.sum(team_ball_control_till_frame == str(team_2_id)) + np.sum(team_ball_control_till_frame == team_2_id)
        
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames > 0:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames
        else:
            team_1 = team_2 = 0

        cv2.putText(frame, f"Team {team_1_id} Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)
        cv2.putText(frame, f"Team {team_2_id} Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Kiểm tra nếu vượt quá số lượng frames trong tracks
            if frame_num < len(tracks["players"]):
                player_dict = tracks["players"][frame_num]
            else:
                player_dict = {}

            if frame_num < len(tracks["ball"]):
                ball_dict = tracks["ball"][frame_num]
            else:
                ball_dict = {}

            if frame_num < len(tracks["referees"]):
                referee_dict = tracks["referees"][frame_num]
            else:
                referee_dict = {}
                
            frame_height, frame_width = frame.shape[:2]

            # Vẽ trọng tài
            for _, referee in referee_dict.items():
                bbox = referee.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue
                    
                # Kiểm tra xem trọng tài có nằm ít nhất một phần trong khung hình không
                x1, y1, x2, y2 = map(float, bbox)
                
                if (x1 < frame_width and x2 > 0 and y1 < frame_height and y2 > 0):
                    frame = self.draw_ellipse(frame, bbox, (0, 255, 255))

            # Vẽ cầu thủ - Đảm bảo vẽ tất cả cầu thủ, kể cả những người ở rìa khung hình
            for track_id, player in player_dict.items():
                bbox = player.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue
                
                # Đảm bảo hiển thị cả những cầu thủ ở rìa khung hình
                x1, y1, x2, y2 = map(float, bbox)
                
                # Kiểm tra xem cầu thủ có nằm ít nhất một phần trong khung hình không
                if (x1 < frame_width and x2 > 0 and y1 < frame_height and y2 > 0):
                    color = player.get("team_color", (0, 0, 255))
                    frame = self.draw_ellipse(frame, bbox, color, track_id)

                    if player.get('has_ball', False):
                        frame = self.draw_traingle(frame, bbox, (0, 0, 255))
                        
                    # Phần hiển thị tốc độ và quãng đường đã được xử lý trong 
                    # speed_and_distance_estimator.draw_speed_and_distance

            # Vẽ bóng
            for _, ball in ball_dict.items():
                bbox = ball.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue
                    
                # Kiểm tra xem bóng có nằm ít nhất một phần trong khung hình không
                x1, y1, x2, y2 = map(float, bbox)
                
                if (x1 < frame_width and x2 > 0 and y1 < frame_height and y2 > 0):
                    frame = self.draw_traingle(frame, bbox, (0, 255, 0))

            # Vẽ thông tin kiểm soát bóng
            if frame_num < len(team_ball_control):
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames