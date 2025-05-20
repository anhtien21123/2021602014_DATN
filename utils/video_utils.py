import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(ouput_video_frames, output_video_path):
    # Sửa đổi định dạng đầu ra thành MP4 nếu đường dẫn kết thúc bằng .avi
    if output_video_path.endswith('.avi'):
        web_output_path = output_video_path.replace('.avi', '.mp4')
    else:
        web_output_path = output_video_path
        
    # Lưu AVI format cho các chức năng nội bộ nếu cần
    if 'output_video.avi' in output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
        for frame in ouput_video_frames:
            out.write(frame)
        out.release()
    
    # Lưu MP4 format cho web với codec H.264 để tương thích tốt hơn với trình duyệt
    try:
        # Sử dụng codec H.264 thay vì mp4v để tương thích tốt hơn với trình duyệt web
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(web_output_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
        for frame in ouput_video_frames:
            out.write(frame)
        out.release()
        print(f"Đã lưu video dạng web-friendly tại: {web_output_path}")
    except Exception as e:
        print(f"Lỗi khi lưu video MP4: {str(e)}")
        # Thử sử dụng codec thay thế nếu avc1 không khả dụng
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # Thử codec H264 thay thế
            out = cv2.VideoWriter(web_output_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
            for frame in ouput_video_frames:
                out.write(frame)
            out.release()
            print(f"Đã lưu video với codec H264 tại: {web_output_path}")
        except Exception as e2:
            print(f"Lỗi khi lưu video MP4 với codec thay thế: {str(e2)}")
