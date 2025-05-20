from utils import read_video, save_video, export_players_to_csv
from trackers import Tracker
import cv2
import numpy as np
import traceback
import os
import glob
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def get_latest_video(directory="input_video"):
    """Lấy file video mới nhất trong thư mục input_video"""
    video_files = glob.glob(os.path.join(directory, "*.mp4"))
    if not video_files:
        # Nếu không tìm thấy file mp4, thử tìm các định dạng khác
        video_files = glob.glob(os.path.join(directory, "*.avi")) + glob.glob(os.path.join(directory, "*.mov"))
    
    if not video_files:
        # Không tìm thấy file video nào
        print(f"CẢNH BÁO: Không tìm thấy file video nào trong thư mục {directory}")
        return None
    
    # Sắp xếp theo thời gian chỉnh sửa, mới nhất ở đầu tiên
    latest_video = max(video_files, key=os.path.getmtime)
    
    # Log thông tin về các file video tìm thấy
    print(f"Danh sách tất cả video ({len(video_files)}):")
    for video in video_files:
        print(f"  - {video} (Thời gian sửa đổi: {os.path.getmtime(video)})")
    
    print(f"Video mới nhất được chọn: {latest_video}")
    return latest_video


def main():
    try:
        # Ghi trạng thái phân tích
        os.makedirs("output_data", exist_ok=True)
        with open("output_data/analysis_status.txt", "w") as f:
            f.write("running")
            
        # Lấy video mới nhất trong thư mục input_video
        video_path = get_latest_video()
        if not video_path:
            print("Không tìm thấy video nào trong thư mục input_video")
            with open("output_data/analysis_status.txt", "w") as f:
                f.write("failed")
            return
            
        print(f"Đang xử lý video: {video_path}")
        
        # Read Video
        print("Đang đọc video...")
        video_frames = read_video(video_path)
        
        if not video_frames or len(video_frames) == 0:
            print("Không thể đọc video hoặc video không có frame")
            with open("output_data/analysis_status.txt", "w") as f:
                f.write("failed")
            return

        print(f"Đã đọc thành công video với {len(video_frames)} frames")

        # Initialize Tracker
        print("Khởi tạo tracker...")
        tracker = Tracker('models/best.pt')

        try:
            print("Đang lấy thông tin theo dõi đối tượng...")
            tracks = tracker.get_object_tracks(video_frames,
                                            read_from_stub=False,
                                            stub_path='stubs/track_stubs.pkl')  # Không sử dụng stub để theo dõi video mới
            
            if 'players' not in tracks or len(tracks['players']) == 0:
                print("Không thể theo dõi người chơi trong video")
                with open("output_data/analysis_status.txt", "w") as f:
                    f.write("failed")
                return
                
            print(f"Đã theo dõi được {len(tracks['players'])} frames của người chơi")
            
            # Get object positions 
            print("Đang thêm vị trí cho các đối tượng...")
            tracker.add_position_to_tracks(tracks)
        except Exception as e:
            print(f"Lỗi khi theo dõi đối tượng: {str(e)}")
            traceback.print_exc()
            with open("output_data/analysis_status.txt", "w") as f:
                f.write("failed")
            return

        try:
            # camera movement estimator
            print("Ước tính chuyển động camera...")
            camera_movement_estimator = CameraMovementEstimator(video_frames[0])
            camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                    read_from_stub=False,
                                                                                    stub_path='stubs/camera_movement_stub.pkl')  # Không sử dụng stub
            camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
        except Exception as e:
            print(f"Lỗi khi ước tính chuyển động camera: {str(e)}")
            traceback.print_exc()
            with open("output_data/analysis_status.txt", "w") as f:
                f.write("failed")
            return

        try:
            # View Trasnformer
            print("Biến đổi góc nhìn...")
            view_transformer = ViewTransformer()
            view_transformer.add_transformed_position_to_tracks(tracks)
        except Exception as e:
            print(f"Lỗi khi biến đổi góc nhìn: {str(e)}")
            traceback.print_exc()
            with open("output_data/analysis_status.txt", "w") as f:
                f.write("failed")
            return

        try:
            # Interpolate Ball Positions
            print("Nội suy vị trí bóng...")
            if 'ball' in tracks:
                tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
            else:
                print("Không tìm thấy thông tin theo dõi bóng")
                tracks["ball"] = [{} for _ in range(len(tracks["players"]))]
        except Exception as e:
            print(f"Lỗi khi nội suy vị trí bóng: {str(e)}")
            traceback.print_exc()
            with open("output_data/analysis_status.txt", "w") as f:
                f.write("failed")
            return

        try:
            # Speed and distance estimator
            print("Ước tính tốc độ và khoảng cách...")
            speed_and_distance_estimator = SpeedAndDistance_Estimator()
            speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
        except Exception as e:
            print(f"Lỗi khi ước tính tốc độ và khoảng cách: {str(e)}")
            traceback.print_exc()
            with open("output_data/analysis_status.txt", "w") as f:
                f.write("failed")
            return

        try:
            # Assign Player Teams
            print("Gán đội cho các cầu thủ...")
            team_assigner = TeamAssigner()
            if len(tracks['players']) > 0:
                team_assigner.assign_team_color(video_frames[0],
                                            tracks['players'][0])

            for frame_num, player_track in enumerate(tracks['players']):
                if frame_num < len(video_frames):  # Đảm bảo không vượt quá số frame video
                    for player_id, track in player_track.items():
                        team = team_assigner.get_player_team(video_frames[frame_num],
                                                        track['bbox'],
                                                        player_id)
                        tracks['players'][frame_num][player_id]['team'] = team
                        tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
        except Exception as e:
            print(f"Lỗi khi gán đội cho cầu thủ: {str(e)}")
            traceback.print_exc()
            with open("output_data/analysis_status.txt", "w") as f:
                f.write("failed")
            return

        try:
            # Assign Ball Aquisition
            print("Xác định cầu thủ kiểm soát bóng...")
            player_assigner = PlayerBallAssigner()
            team_ball_control = []
            
            if 'ball' in tracks:
                for frame_num, player_track in enumerate(tracks['players']):
                    if frame_num < len(tracks['ball']):  # Kiểm tra frame_num có hợp lệ không
                        ball_bbox = None
                        if 1 in tracks['ball'][frame_num]:
                            ball_bbox = tracks['ball'][frame_num][1].get('bbox')
                        assigned_player = -1
                        
                        if ball_bbox is not None:
                            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

                        if assigned_player != -1:
                            tracks['players'][frame_num][assigned_player]['has_ball'] = True
                            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
                        else:
                            team_ball_control.append(team_ball_control[-1] if team_ball_control else 'A')
            
            team_ball_control = np.array(team_ball_control)
            print(f"Số lượng frame có thông tin kiểm soát bóng: {len(team_ball_control)}")
        except Exception as e:
            print(f"Lỗi khi xác định cầu thủ kiểm soát bóng: {str(e)}")
            traceback.print_exc()
            team_ball_control = np.array(['A'] * len(video_frames))  # Tạo giá trị mặc định

        # Đảm bảo số lượng frame trong team_ball_control khớp với số frame video
        try:
            if len(team_ball_control) < len(video_frames):
                # Nếu thiếu, thêm giá trị cuối cùng vào
                print(f"Thiếu {len(video_frames) - len(team_ball_control)} frame thông tin kiểm soát bóng, đang thêm giá trị mặc định...")
                last_value = team_ball_control[-1] if len(team_ball_control) > 0 else 'A'
                team_ball_control = np.append(team_ball_control, 
                                            np.array([last_value] * (len(video_frames) - len(team_ball_control))))
            elif len(team_ball_control) > len(video_frames):
                # Nếu thừa, cắt bớt
                print(f"Thừa {len(team_ball_control) - len(video_frames)} frame thông tin kiểm soát bóng, đang cắt bớt...")
                team_ball_control = team_ball_control[:len(video_frames)]
        except Exception as e:
            print(f"Lỗi khi điều chỉnh thông tin kiểm soát bóng: {str(e)}")
            traceback.print_exc()
            team_ball_control = np.array(['A'] * len(video_frames))  # Tạo giá trị mặc định

        try:
            # Draw output
            print("Đang vẽ đồ họa lên video...")
            ## Draw object Tracks
            output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

            ## Draw Camera movement
            output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

            ## Draw Speed and Distance
            output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
        except Exception as e:
            print(f"Lỗi khi vẽ đồ họa lên video: {str(e)}")
            traceback.print_exc()
            output_video_frames = video_frames  # Sử dụng video gốc nếu có lỗi

        try:
            # Save video
            print("Đang lưu video đầu ra...")
            save_video(output_video_frames, 'output_video/output_video.avi')
        except Exception as e:
            print(f"Lỗi khi lưu video: {str(e)}")
            traceback.print_exc()

        try:
            # Export player stats to CSV
            print("Đang xuất thống kê cầu thủ ra CSV...")
            export_players_to_csv(tracks, 'output_data/player_stats.csv')
        except Exception as e:
            print(f"Lỗi khi xuất thống kê cầu thủ: {str(e)}")
            traceback.print_exc()
        
        # Lưu thống kê kiểm soát bóng
        try:
            print("Đang lưu thông tin kiểm soát bóng...")
            team_1_id = 1
            team_2_id = 2
            
            team_1_possession = 0
            team_2_possession = 0
            
            if len(team_ball_control) > 0:
                team_1_num_frames = np.sum(team_ball_control == str(team_1_id)) + np.sum(team_ball_control == team_1_id)
                team_2_num_frames = np.sum(team_ball_control == str(team_2_id)) + np.sum(team_ball_control == team_2_id)
                
                total_frames = team_1_num_frames + team_2_num_frames
                if total_frames > 0:
                    team_1_possession = team_1_num_frames / total_frames
                    team_2_possession = team_2_num_frames / total_frames
            
            # Lưu thông tin vào file
            with open("output_data/ball_possession.txt", "w") as f:
                f.write(f"1: {team_1_possession * 100:.1f}%\n")
                f.write(f"2: {team_2_possession * 100:.1f}%\n")
            
            # Lưu thông tin tốc độ tối đa và quãng đường trung bình cho mỗi đội
            max_speeds = {}
            total_distance = {1: 0.0, 2: 0.0}  # Tổng quãng đường cho mỗi đội
            
            for frame_num, player_track in enumerate(tracks['players']):
                for player_id, track in player_track.items():
                    team = track.get('team', 0)
                    
                    # Tính tốc độ tối đa
                    if 'speed' in track:
                        if team not in max_speeds:
                            max_speeds[team] = 0
                        max_speeds[team] = max(max_speeds[team], track['speed'])
                    
                    # Tính tổng quãng đường di chuyển
                    if 'distance' in track and team in [1, 2]:
                        # Chỉ tính dữ liệu từ frame cuối cùng để lấy tổng quãng đường
                        if frame_num == len(tracks['players']) - 1:
                            total_distance[team] += track['distance']
            
            # Tính quãng đường trung bình trên mỗi cầu thủ (chia cho 11 cầu thủ mặc định)
            avg_distance_team1 = total_distance[1] / 11  # Chia cho 11 cầu thủ mặc định
            avg_distance_team2 = total_distance[2] / 11  # Chia cho 11 cầu thủ mặc định
            
            with open("output_data/team_speeds.txt", "w") as f:
                for team, speed in max_speeds.items():
                    f.write(f"Team {team}: {speed:.1f}\n")
            
            # Lưu thông tin quãng đường trung bình
            with open("output_data/team_distances.txt", "w") as f:
                f.write(f"Team 1 avg distance: {avg_distance_team1:.1f} m\n")
                f.write(f"Team 2 avg distance: {avg_distance_team2:.1f} m\n")
                    
        except Exception as e:
            print(f"Lỗi khi lưu thông tin kiểm soát bóng: {str(e)}")
            traceback.print_exc()
            
        print("Phân tích video hoàn tất!")
        with open("output_data/analysis_status.txt", "w") as f:
            f.write("completed")
        
    except Exception as e:
        print(f"Lỗi trong quá trình phân tích: {str(e)}")
        traceback.print_exc()
        with open("output_data/analysis_status.txt", "w") as f:
            f.write("failed")


if __name__ == '__main__':
    main()