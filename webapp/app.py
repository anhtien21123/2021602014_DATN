from fastapi import FastAPI, File, UploadFile, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import shutil
import subprocess
import pandas as pd
import uvicorn
from pathlib import Path
import glob
import time
import tempfile

app = FastAPI(title="Hệ Thống Phân Tích Bóng Đá")

# Đường dẫn gốc của ứng dụng
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if __name__ != "__main__" else os.path.dirname(os.path.abspath(__file__))

# Cấu hình đường dẫn cho templates và static files
templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

templates = Jinja2Templates(directory=templates_path)
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Đảm bảo các thư mục tồn tại
os.makedirs("input_video", exist_ok=True)
os.makedirs("output_video", exist_ok=True)
os.makedirs("output_data", exist_ok=True)
os.makedirs(static_path, exist_ok=True)

# Biến toàn cục để theo dõi trạng thái phân tích
analysis_status = {
    "status": "idle",  # idle, running, completed, failed
    "start_time": None,
    "end_time": None,
    "progress": 0,
    "message": "",
    "error": ""
}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(request: Request, file: UploadFile = File(...)):
    # Tạo tên file mới với timestamp để tránh trùng lặp
    timestamp = int(time.time())
    file_ext = os.path.splitext(file.filename)[1]
    new_filename = f"video_{timestamp}{file_ext}"
    file_path = os.path.join("input_video", new_filename)
    
    try:
        # Lưu file vào thư mục upload
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Đảm bảo file được lưu với timestamp mới nhất
        os.utime(file_path, None)  # Cập nhật thời gian truy cập và sửa đổi
        
        print(f"Đã tải lên và lưu file video: {file_path}")
        print(f"Thời gian chỉnh sửa: {os.path.getmtime(file_path)}")
        
        return RedirectResponse(url="/", status_code=303)
    except Exception as e:
        print(f"Lỗi khi tải lên video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Lỗi khi tải lên video: {str(e)}"}
        )

def run_video_analysis():
    """Hàm chạy phân tích video trong background task"""
    global analysis_status
    
    analysis_status["status"] = "running"
    analysis_status["start_time"] = time.time()
    analysis_status["message"] = "Đang phân tích video..."
    
    try:
        # Đảm bảo output_data tồn tại
        os.makedirs("output_data", exist_ok=True)
        
        # Dọn dẹp các file video cũ trong thư mục static
        clean_old_video_files()
        
        # Chạy script phân tích
        result = subprocess.run(
            ["python", os.path.join(BASE_DIR, "main.py")], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Kiểm tra kết quả
        if result.returncode == 0 and os.path.exists("output_data/player_stats.csv"):
            analysis_status["status"] = "completed"
            analysis_status["message"] = "Phân tích hoàn tất thành công"
        else:
            analysis_status["status"] = "failed"
            analysis_status["message"] = f"Phân tích thất bại với mã lỗi {result.returncode}"
            analysis_status["error"] = result.stderr
    
    except Exception as e:
        analysis_status["status"] = "failed"
        analysis_status["message"] = f"Lỗi khi phân tích video: {str(e)}"
        analysis_status["error"] = str(e)
    
    analysis_status["end_time"] = time.time()
    analysis_status["progress"] = 100

def clean_old_video_files():
    """Xóa các file video cũ trong thư mục static để tiết kiệm không gian đĩa"""
    try:
        # Lấy tất cả các file video trong thư mục static
        mp4_files = glob.glob(os.path.join(static_path, "output_video_*.mp4"))
        avi_files = glob.glob(os.path.join(static_path, "output_video_*.avi"))
        video_files = mp4_files + avi_files
        
        # Giữ lại tối đa 3 video gần nhất
        if len(video_files) > 3:
            # Sắp xếp theo thời gian sửa đổi (cũ nhất đầu tiên)
            video_files.sort(key=os.path.getmtime)
            
            # Xóa các file cũ
            for file_to_delete in video_files[:-3]:  # Giữ lại 3 file mới nhất
                try:
                    os.remove(file_to_delete)
                    print(f"Đã xóa file video cũ: {file_to_delete}")
                except Exception as e:
                    print(f"Không thể xóa file video cũ {file_to_delete}: {str(e)}")
    except Exception as e:
        print(f"Lỗi khi dọn dẹp file video cũ: {str(e)}")

@app.post("/analyze")
async def analyze_video(request: Request, background_tasks: BackgroundTasks):
    global analysis_status
    
    # Kiểm tra có video trong thư mục input_video không
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(glob.glob(os.path.join("input_video", ext)))
    
    if not video_files:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Vui lòng tải lên video trước khi phân tích"}
        )
    
    # Kiểm tra nếu đang có phân tích đang chạy
    if analysis_status["status"] == "running":
        return JSONResponse(
            content={"status": "running", "message": "Đang có phân tích video đang chạy, vui lòng đợi"}
        )
    
    # Xác định video mới nhất theo thời gian chỉnh sửa
    latest_video = max(video_files, key=os.path.getmtime)
    print(f"Chuẩn bị phân tích video mới nhất: {latest_video}")
    print(f"Thời gian chỉnh sửa: {os.path.getmtime(latest_video)}")

    # Đảm bảo thư mục đầu ra trống
    if os.path.exists("output_video/output_video.avi"):
        try:
            os.remove("output_video/output_video.avi")
            print("Đã xóa video AVI đầu ra cũ")
        except Exception as e:
            print(f"Không thể xóa video AVI đầu ra cũ: {str(e)}")
    
    if os.path.exists("output_video/output_video.mp4"):
        try:
            os.remove("output_video/output_video.mp4")
            print("Đã xóa video MP4 đầu ra cũ")
        except Exception as e:
            print(f"Không thể xóa video MP4 đầu ra cũ: {str(e)}")
    
    # Xóa các file tạm trong thư mục static
    if os.path.exists(os.path.join(static_path, "output_video.avi")):
        try:
            os.remove(os.path.join(static_path, "output_video.avi"))
            print("Đã xóa file AVI tạm trong thư mục static")
        except Exception as e:
            print(f"Không thể xóa file AVI tạm: {str(e)}")
            
    if os.path.exists(os.path.join(static_path, "output_video.mp4")):
        try:
            os.remove(os.path.join(static_path, "output_video.mp4"))
            print("Đã xóa file MP4 tạm trong thư mục static")
        except Exception as e:
            print(f"Không thể xóa file MP4 tạm: {str(e)}")

    if os.path.exists("output_data/player_stats.csv"):
        try:
            os.remove("output_data/player_stats.csv")
            print("Đã xóa file CSV đầu ra cũ")
        except Exception as e:
            print(f"Không thể xóa file CSV đầu ra cũ: {str(e)}")

    # Reset trạng thái phân tích
    analysis_status["status"] = "running"
    analysis_status["start_time"] = time.time()
    analysis_status["end_time"] = None
    analysis_status["progress"] = 0
    analysis_status["message"] = "Đang chuẩn bị phân tích video..."
    analysis_status["error"] = ""
    
    # Chạy phân tích trong background
    background_tasks.add_task(run_video_analysis)
    
    return JSONResponse(
        content={
            "status": "started", 
            "message": "Đã bắt đầu phân tích video, vui lòng kiểm tra trạng thái phân tích"
        }
    )

@app.get("/analysis-status")
async def get_analysis_status():
    """Trả về trạng thái hiện tại của quá trình phân tích"""
    global analysis_status
    
    # Kiểm tra từ file status nếu có
    if os.path.exists("output_data/analysis_status.txt"):
        try:
            with open("output_data/analysis_status.txt", "r") as f:
                status_from_file = f.read().strip()
                
                # Cập nhật trạng thái từ file nếu cần
                if status_from_file == "completed" and analysis_status["status"] == "running":
                    analysis_status["status"] = "completed"
                    analysis_status["message"] = "Phân tích hoàn tất thành công"
                    analysis_status["progress"] = 100
                    analysis_status["end_time"] = time.time()
                elif status_from_file == "failed" and analysis_status["status"] == "running":
                    analysis_status["status"] = "failed"
                    analysis_status["message"] = "Phân tích thất bại"
                    analysis_status["progress"] = 100
                    analysis_status["end_time"] = time.time()
        except Exception:
            pass
    
    # Kiểm tra kết quả nếu trạng thái đang chạy nhưng đã có file kết quả
    if analysis_status["status"] == "running" and os.path.exists("output_data/player_stats.csv"):
        # Kiểm tra thời gian sửa đổi của file
        file_mtime = os.path.getmtime("output_data/player_stats.csv")
        if file_mtime > analysis_status["start_time"]:
            analysis_status["status"] = "completed"
            analysis_status["message"] = "Phân tích hoàn tất thành công"
            analysis_status["progress"] = 100
            analysis_status["end_time"] = time.time()
    
    # Tính thời gian đã chạy nếu đang trong quá trình phân tích
    if analysis_status["status"] == "running":
        elapsed_time = time.time() - analysis_status["start_time"]
        # Ước tính tiến độ dựa trên thời gian (giả sử phân tích mất khoảng 5 phút)
        estimated_total_time = 300  # 5 phút
        progress = min(95, int((elapsed_time / estimated_total_time) * 100))
        analysis_status["progress"] = progress
    
    return analysis_status

@app.get("/results", response_class=HTMLResponse)
async def show_results(request: Request):
    # Đọc dữ liệu từ CSV
    player_stats = pd.DataFrame()
    if os.path.exists("output_data/player_stats.csv"):
        player_stats = pd.read_csv("output_data/player_stats.csv")
    
    # Chuẩn bị dữ liệu cho template
    teams = {}
    if not player_stats.empty:
        for team in player_stats['team'].unique():
            team_data = player_stats[player_stats['team'] == team]
            teams[team] = team_data.to_dict('records')
    
    # Đường dẫn đến video đầu ra
    output_video_path = None
    

    if os.path.exists("output_video/output_video.mp4"):
        timestamp = int(time.time())  # Thêm timestamp để ngăn cache
        output_video_name = f"output_video_{timestamp}.mp4"
        target_path = os.path.join(static_path, output_video_name)
        output_video_path = f"/static/{output_video_name}"
        
        try:
            # Copy video to static folder for access với tên mới có timestamp
            shutil.copy("output_video/output_video.mp4", target_path)
            print(f"Đã sao chép video MP4 vào thư mục static: {target_path}")
            
            # Thêm quyền đọc cho tất cả người dùng nếu cần
            try:
                os.chmod(target_path, 0o644)
            except:
                pass
        except Exception as e:
            print(f"Không thể copy video MP4 đầu ra: {str(e)}")
    
    # Nếu không có MP4, thử dùng AVI
    elif os.path.exists("output_video/output_video.avi"):
        timestamp = int(time.time())
        output_video_name = f"output_video_{timestamp}.avi"
        target_path = os.path.join(static_path, output_video_name)
        output_video_path = f"/static/{output_video_name}"
        
        try:
            # Copy video to static folder với tên mới có timestamp
            shutil.copy("output_video/output_video.avi", target_path)
            print(f"Đã sao chép video AVI vào thư mục static: {target_path}")
            
            # Thêm quyền đọc cho tất cả người dùng nếu cần
            try:
                os.chmod(target_path, 0o644)
            except:
                pass
        except Exception as e:
            print(f"Không thể copy video AVI đầu ra: {str(e)}")
    
    return templates.TemplateResponse(
        "results.html", 
        {
            "request": request, 
            "teams": teams,
            "video_path": output_video_path
        }
    )

@app.get("/compare", response_class=HTMLResponse)
async def compare_teams(request: Request):
    # Đọc dữ liệu từ CSV
    player_stats = pd.DataFrame()
    if os.path.exists("output_data/player_stats.csv"):
        player_stats = pd.read_csv("output_data/player_stats.csv")
    
    # Tính toán thống kê theo đội
    team_stats = {}
    
    # Đọc dữ liệu kiểm soát bóng từ phân tích video
    team_possession = {1: 0, 2: 0}
    total_frames = 0
    
    try:
        # Đọc file status để kiểm tra phân tích đã hoàn tất
        if os.path.exists("output_data/analysis_status.txt"):
            with open("output_data/analysis_status.txt", "r") as f:
                status = f.read().strip()
                if status == "completed":
                    # Đọc thông tin kiểm soát bóng từ main output
                    if os.path.exists("output_data/ball_possession.txt"):
                        with open("output_data/ball_possession.txt", "r") as f:
                            lines = f.readlines()
                            for line in lines:
                                parts = line.strip().split(":")
                                if len(parts) == 2 and parts[0].strip() in ["1", "2"]:
                                    team_id = int(parts[0].strip())
                                    possession = float(parts[1].strip().replace("%", "")) / 100
                                    team_possession[team_id] = possession
    except Exception as e:
        print(f"Lỗi khi đọc dữ liệu kiểm soát bóng: {str(e)}")
        # Sử dụng giá trị mặc định
        team_possession = {1: 0.5, 2: 0.5}
    
    if not player_stats.empty:
        for team in player_stats['team'].unique():
            team_key = str(team)  # Chuyển đổi team thành string để đảm bảo key đúng định dạng
            team_data = player_stats[player_stats['team'] == team]
            
            # Xử lý dữ liệu tốc độ nếu có
            max_speed = 0
            try:
                # Tạo file tạm thời để ghi dữ liệu tốc độ tối đa
                temp_speed_file = "output_data/team_speeds.txt"
                if os.path.exists(temp_speed_file):
                    with open(temp_speed_file, "r") as f:
                        for line in f:
                            if line.startswith(f"Team {team}:"):
                                speed_str = line.split(":")[1].strip()
                                max_speed = float(speed_str)
                                break
            except Exception:
                # Nếu không đọc được, sử dụng giá trị mặc định từ khoảng cách
                max_speed = float(team_data['total_distance_meters'].max())
            
            # Tính toán thống kê đội
            team_stats[team_key] = {
                'total_distance': float(team_data['total_distance_meters'].sum()),
                'avg_distance': float(team_data['total_distance_meters'].mean()),
                'player_count': int(len(team_data)),
                'max_speed': max_speed,
                'total_possession': team_possession.get(int(team), 0.5)  # Sử dụng dữ liệu kiểm soát bóng thực tế
            }
            
            # Đảm bảo tốc độ và quãng đường có giá trị hợp lý
            if team_stats[team_key]['max_speed'] <= 0:
                team_stats[team_key]['max_speed'] = 10.0  # Giá trị mặc định hợp lý
            
            # Chuyển đổi any NumPy data types to Python types để tránh lỗi JSON
            for key, value in team_stats[team_key].items():
                if hasattr(value, 'item'):  # NumPy scalar
                    team_stats[team_key][key] = value.item()
    
    # Thay đổi: Đọc thông tin quãng đường trung bình từ file team_distances.txt
    try:
        if os.path.exists("output_data/team_distances.txt"):
            with open("output_data/team_distances.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "Team 1 avg distance:" in line:
                        avg_distance = float(line.split(":")[1].strip().replace(" m", ""))
                        if "1" in team_stats:
                            team_stats["1"]["avg_distance"] = avg_distance
                    elif "Team 2 avg distance:" in line:
                        avg_distance = float(line.split(":")[1].strip().replace(" m", ""))
                        if "2" in team_stats:
                            team_stats["2"]["avg_distance"] = avg_distance
    except Exception as e:
        print(f"Lỗi khi đọc thông tin quãng đường trung bình: {str(e)}")
    
    # Nếu không có dữ liệu kiểm soát bóng, đảm bảo tổng là 100%
    if len(team_stats) >= 2:
        team_keys = list(team_stats.keys())
        total_possession = sum(team_stats[key]['total_possession'] for key in team_keys)
        if total_possession == 0:
            # Nếu tổng kiểm soát bóng là 0, phân bổ đều
            for key in team_keys:
                team_stats[key]['total_possession'] = 1.0 / len(team_keys)
        elif total_possession != 1.0:
            # Chuẩn hóa để tổng bằng 1.0
            for key in team_keys:
                team_stats[key]['total_possession'] = team_stats[key]['total_possession'] / total_possession
    
    return templates.TemplateResponse(
        "compare.html", 
        {
            "request": request, 
            "team_stats": team_stats
        }
    )

@app.get("/debug")
async def debug_info(request: Request):
    """Endpoint for debugging file access issues"""
    debug_info = {
        "base_dir": BASE_DIR,
        "input_video_dir": os.path.abspath("input_video"),
        "output_video_dir": os.path.abspath("output_video"),
        "output_data_dir": os.path.abspath("output_data"),
        "input_videos": [],
        "analysis_status": analysis_status
    }
    
    # Check input videos
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(glob.glob(os.path.join("input_video", ext)))
    
    for video in video_files:
        debug_info["input_videos"].append({
            "path": video,
            "size_mb": round(os.path.getsize(video) / (1024 * 1024), 2),
            "last_modified": os.path.getmtime(video),
            "permissions": oct(os.stat(video).st_mode)[-3:]
        })
    
    # Check if get_latest_video function works
    try:
        from main import get_latest_video
        latest_video = get_latest_video()
        debug_info["latest_video_from_function"] = latest_video
    except Exception as e:
        debug_info["latest_video_error"] = str(e)
    
    return debug_info

@app.get("/video/{video_name}")
async def serve_video(video_name: str):
    """Trả về video với MIME type phù hợp"""
    video_path = os.path.join(static_path, video_name)
    
    if not os.path.exists(video_path):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Video không tồn tại"}
        )
    
    # Xác định MIME type phù hợp
    mime_type = "video/mp4"
    if video_name.endswith(".avi"):
        mime_type = "video/x-msvideo"
    
    # Tạo generator để trả về file từng phần
    def iterfile():
        with open(video_path, "rb") as f:
            yield from f
    
    # Trả về video với headers phù hợp
    return StreamingResponse(
        iterfile(),
        media_type=mime_type,
        headers={
            "Content-Disposition": f"inline; filename={video_name}",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache"
        }
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True) 