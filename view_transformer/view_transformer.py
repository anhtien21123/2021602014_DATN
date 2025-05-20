import numpy as np
import cv2


class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array([[110, 1035],
                                        [265, 275],
                                        [910, 260],
                                        [1640, 915]])

        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        
        # Nếu điểm nằm trong sân, thực hiện biến đổi bình thường
        if is_inside:
            reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
            tranform_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)
            return tranform_point.reshape(-1, 2)
        else:
            # Nếu điểm nằm ngoài sân, tìm điểm gần nhất trên biên của sân
            # Tính khoảng cách đến các điểm trên biên
            dist = cv2.pointPolygonTest(self.pixel_vertices, p, True)
            
            # Tìm cạnh gần nhất và điểm gần nhất trên cạnh đó
            closest_edge_points = []
            for i in range(len(self.pixel_vertices)):
                p1 = self.pixel_vertices[i]
                p2 = self.pixel_vertices[(i + 1) % len(self.pixel_vertices)]
                
                # Tính vector từ p1 đến p2
                edge_vector = p2 - p1
                edge_length = np.sqrt(np.sum(edge_vector**2))
                edge_unit = edge_vector / edge_length
                
                # Tính vector từ p1 đến p
                point_vector = np.array(p) - p1
                
                # Tính projection của point_vector lên edge_unit
                projection_length = np.dot(point_vector, edge_unit)
                
                # Nếu projection nằm trên cạnh
                if 0 <= projection_length <= edge_length:
                    # Tính điểm projection trên cạnh
                    projection_point = p1 + projection_length * edge_unit
                    
                    # Tính khoảng cách từ p đến projection_point
                    distance = np.sqrt(np.sum((np.array(p) - projection_point)**2))
                    
                    closest_edge_points.append((distance, projection_point))
            
            if closest_edge_points:
                # Lấy điểm gần nhất
                closest_edge_points.sort(key=lambda x: x[0])
                closest_point = closest_edge_points[0][1]
                
                # Biến đổi điểm gần nhất
                reshaped_point = closest_point.reshape(-1, 1, 2).astype(np.float32)
                tranform_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)
                return tranform_point.reshape(-1, 2)
            
            return None

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed