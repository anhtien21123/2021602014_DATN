import csv
import os
import numpy as np

def calculate_total_distance(player_tracks):
    """Tính tổng quãng đường di chuyển của cầu thủ"""
    total_distance = 0
    for i in range(1, len(player_tracks)):
        if player_tracks[i-1]['position_transformed'] is not None and player_tracks[i]['position_transformed'] is not None:
            pos_prev = player_tracks[i-1]['position_transformed']
            pos_curr = player_tracks[i]['position_transformed']
            distance = np.sqrt((pos_curr[0] - pos_prev[0])**2 + (pos_curr[1] - pos_prev[1])**2)
            total_distance += distance
    return total_distance

def export_players_to_csv(tracks, output_file="player_stats.csv"):
    """
    Xuất thông tin cầu thủ vào file CSV
    
    Parameters:
    - tracks: dictionary chứa thông tin theo dõi cầu thủ
    - output_file: đường dẫn đến file CSV đầu ra
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Tạo dictionary lưu trữ dữ liệu của từng cầu thủ
    players_data = {}
    
    # Lặp qua từng frame để thu thập dữ liệu
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            if player_id not in players_data:
                players_data[player_id] = {
                    'id': player_id,
                    'team': track.get('team', 'Unknown'),
                    'tracks': []
                }
            
            players_data[player_id]['tracks'].append(track)
    
    # Tính toán quãng đường di chuyển cho mỗi cầu thủ
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['player_id', 'team', 'total_distance_meters']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for player_id, player_data in players_data.items():
            total_distance = calculate_total_distance(player_data['tracks'])
            writer.writerow({
                'player_id': player_id,
                'team': player_data['team'],
                'total_distance_meters': round(total_distance, 2)
            })
    
    print(f"Player stats exported to {output_file}") 