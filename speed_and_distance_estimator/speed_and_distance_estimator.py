import cv2
import sys

sys.path.append('../')
from utils import measure_distance, get_foot_position


class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue
                
            number_of_frames = len(object_tracks)
            
            # Tạo dictionary để lưu vị trí cuối cùng của mỗi cầu thủ
            last_known_positions = {}
            last_known_frame = {}
            
            # Khởi tạo tổng quãng đường cho mỗi cầu thủ
            for frame_num in range(number_of_frames):
                for track_id in object_tracks[frame_num].keys():
                    if object not in total_distance:
                        total_distance[object] = {}
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
            
            # Tính toán tốc độ và quãng đường cho tất cả cầu thủ qua các frame
            for frame_num in range(number_of_frames):
                # Lấy các cầu thủ hiện diện trong frame hiện tại
                current_tracks = object_tracks[frame_num]
                
                for track_id, track_info in current_tracks.items():
                    current_position = track_info.get('position_transformed')
                    
                    if current_position is None:
                        continue
                        
                    # Nếu đã biết vị trí trước đó của cầu thủ này
                    if track_id in last_known_positions:
                        prev_position = last_known_positions[track_id]
                        prev_frame = last_known_frame[track_id]
                        
                        # Tính khoảng cách giữa hai vị trí
                        distance_covered = measure_distance(prev_position, current_position)
                        
                        # Tính thời gian giữa hai frame
                        time_elapsed = (frame_num - prev_frame) / self.frame_rate
                        
                        if time_elapsed > 0:
                            # Tính tốc độ
                            speed_meteres_per_second = distance_covered / time_elapsed
                            speed_km_per_hour = speed_meteres_per_second * 3.6
                            
                            # Cộng dồn quãng đường
                            total_distance[object][track_id] += distance_covered
                            
                            # Cập nhật tốc độ và quãng đường cho cầu thủ trong frame hiện tại
                            object_tracks[frame_num][track_id]['speed'] = speed_meteres_per_second
                            object_tracks[frame_num][track_id]['speed_kmh'] = speed_km_per_hour
                            object_tracks[frame_num][track_id]['distance'] = total_distance[object][track_id]
                    
                    # Cập nhật vị trí cuối cùng đã biết
                    last_known_positions[track_id] = current_position
                    last_known_frame[track_id] = frame_num

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue
                    
                # Kiểm tra nếu frame_num hợp lệ
                if frame_num >= len(object_tracks):
                    continue
                    
                for track_id, track_info in object_tracks[frame_num].items():
                    # Lấy bbox từ track
                    bbox = track_info.get('bbox')
                    if bbox is None:
                        continue
                    
                    # Lấy vị trí để hiển thị thông tin
                    position = get_foot_position(bbox)
                    position = list(position)
                    position[1] += 40  # Điều chỉnh vị trí cho phù hợp
                    position = tuple(map(int, position))
                    
                    # Lấy tốc độ km/h hoặc tốc độ m/s và chuyển đổi
                    speed_kmh = track_info.get('speed_kmh')
                    if speed_kmh is None:
                        speed_ms = track_info.get('speed')
                        if speed_ms is not None:
                            speed_kmh = speed_ms * 3.6
                    
                    # Lấy tổng quãng đường đã di chuyển
                    total_distance = track_info.get('distance')
                    
                    # Nếu có thông tin tốc độ, hiển thị
                    if speed_kmh is not None:
                        # Hiển thị tốc độ với nền để dễ nhìn
                        text = f"{speed_kmh:.1f} km/h"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, 
                                     (position[0]-2, position[1]-text_size[1]-2),
                                     (position[0]+text_size[0]+2, position[1]+2),
                                     (255, 255, 255), -1)
                        cv2.putText(frame, text, position, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    # Nếu có thông tin quãng đường, hiển thị
                    if total_distance is not None:
                        # Điều chỉnh vị trí để hiển thị dưới tốc độ
                        distance_position = (position[0], position[1] + 20)
                        
                        # Hiển thị quãng đường với nền để dễ nhìn
                        text = f"{total_distance:.1f} m"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, 
                                     (distance_position[0]-2, distance_position[1]-text_size[1]-2),
                                     (distance_position[0]+text_size[0]+2, distance_position[1]+2),
                                     (255, 255, 255), -1)
                        cv2.putText(frame, text, distance_position,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            
            output_frames.append(frame)

        return output_frames