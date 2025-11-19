import os
import shutil
import time
import threading
import json
import glob
import subprocess 
import sys
import socket # ⭐️ (เพิ่ม) ไว้หา IP Address
from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
import numpy as np
import cv2

try:
    import dobot_drawing_logic as ddl 
    from dobot_drawing_logic import (
        DOBOT_SPEED, DOBOT_ACCELERATION, 
        PEN_DOWN_Z, PEN_UP_Z, TEST_PARAMS
    )
except ImportError:
    print("❌ ไม่พบไฟล์ dobot_drawing_logic.py")
    exit()

app = Flask(__name__) 
CORS(app) 

UPLOAD_FOLDER = 'static/mobile_uploads'
OUTPUT_FOLDER = 'static/processed' 
# ⭐️ โฟลเดอร์เก็บรูปจากมือถือ
RAW_UPLOAD_FOLDER = 'static/mobile_uploads' 

ddl.OUTPUT_DIR_BASE = OUTPUT_FOLDER 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(RAW_UPLOAD_FOLDER, exist_ok=True) 

bot = None
drawing_thread = None
drawing_state = {
    "status": "idle", 
    "message": "Disconnected",
    "progress": 0,
    "progress_image_url": "",
    "stop_flag": False
}
processed_data = {
    "current_run_dir": None,
    "all_steps_dir": None,
    "base_bgr_image": None,
    "bw_image_path": None, 
    "img_gray_resized": None,
    "filtered_contours": None,
    "processed_paths": None,
    "contour_lengths": None,
    "total_contours": 0,
    "original_image_name": None 
}

DFCALL_SCRIPT_PATH = '/Users/student/Desktop/DobotDrawRESEARCH/dfcall/512_use/dfcall.py'
DFCALL_OUTPUT_IMAGE_PATH = '/Users/student/Desktop/DobotDrawRESEARCH/dfcall/512_use/stitched_cartoon_512x512.jpg'
DFCALL_DIR = os.path.dirname(DFCALL_SCRIPT_PATH)

# --- ⭐️ ฟังก์ชันหา IP Address ของเครื่อง (เพื่อสร้าง QR) ⭐️ ---
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # ไม่ได้เชื่อมต่อจริง แต่ใช้เพื่อดูว่าจะใช้ IP ไหนออกเน็ต
        s.connect(('8.8.8.8', 80))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

MY_IP = get_ip()
PORT = 5001
# --------------------------------------------------------

@app.route('/')
def index():
    # ส่ง IP ไปให้หน้าเว็บเพื่อสร้าง QR Code
    return render_template('index.html', server_ip=MY_IP, server_port=PORT)

# --- ⭐️ API รับไฟล์จากมือถือ (รองรับหลายไฟล์) ⭐️ ---
@app.route('/quick_upload', methods=['POST'])
def quick_upload():
    if 'images' not in request.files:
        return jsonify({"status": "error", "message": "No images part"}), 400
    
    files = request.files.getlist('images') # รับเป็น list
    if not files or files[0].filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    saved_files = []
    try:
        timestamp = int(time.time())
        for i, file in enumerate(files):
            if file:
                safe_filename = f"mobile_{timestamp}_{i}_{file.filename}"
                save_path = os.path.join(RAW_UPLOAD_FOLDER, safe_filename)
                file.save(save_path)
                saved_files.append(safe_filename)
                print(f"📱 Saved: {save_path}")
        
        return jsonify({
            "status": "success", 
            "message": f"Saved {len(saved_files)} images to PC!",
            "filenames": saved_files
        })
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
# ----------------------------------------------------

@app.route('/connect', methods=['POST'])
def connect_dobot():
    global bot
    if bot: return jsonify({"status": "success", "message": "Already connected", "port": bot.port, "model": "Dobot Magician"}), 200
    port = ddl.find_dobot_port()
    if not port:
        drawing_state["message"] = "Dobot not found"
        return jsonify({"status": "error", "message": "Dobot not found. Check connection."}), 404
    try:
        bot = ddl.Dobot(port=port, verbose=False)
        bot.speed(DOBOT_SPEED, DOBOT_ACCELERATION)
        drawing_state["status"] = "idle"
        drawing_state["message"] = "Connected"
        print(f"✅ Dobot connected at {port}")
        return jsonify({"status": "success", "message": "Connected", "port": port, "model": "Dobot Magician"}), 200
    except Exception as e:
        drawing_state["message"] = f"Connection failed: {e}"
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/disconnect', methods=['POST'])
def disconnect_dobot():
    global bot
    if bot:
        try: bot.close()
        except Exception as e: print(f"⚠️ Error closing dobot: {e}")
        bot = None
    drawing_state["status"] = "idle"
    drawing_state["message"] = "Disconnected"
    return jsonify({"status": "success", "message": "Disconnected"})

@app.route('/get_position', methods=['GET'])
def get_position():
    if not bot: return jsonify({"status": "error", "message": "Dobot not connected"}), 400
    try:
        pose = bot.pose()
        return jsonify({"status": "success", "x": round(pose[0], 2), "y": round(pose[1], 2)})
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/set_paper_corners', methods=['POST'])
def set_paper_corners():
    data = request.json
    corners_dict = data.get('corners')
    if not corners_dict: return jsonify({"status": "error", "message": "No corners data provided"}), 400
    try:
        corners_list = [corners_dict['tl'], corners_dict['tr'], corners_dict['br'], corners_dict['bl']] 
        with open(ddl.CALIBRATION_FILE, 'w') as f: json.dump(corners_list, f, indent=4)
        ddl.PAPER_CORNERS = np.float32(corners_list)
        print(f"✅ New calibration saved: {corners_list}")
        return jsonify({"status": "success", "message": "Corners set and saved"})
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/process_image', methods=['POST'])
def process_image():
    global processed_data
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    drawing_state["status"] = "processing"
    drawing_state["message"] = "Starting DFCall..."

    try:
        if os.path.exists(DFCALL_OUTPUT_IMAGE_PATH):
            print(f"🧹 Clearing old output file: {DFCALL_OUTPUT_IMAGE_PATH}")
            os.remove(DFCALL_OUTPUT_IMAGE_PATH)
            
        run_dir = ddl.get_next_experiment_dir() 
        processed_data["current_run_dir"] = run_dir
        processed_data["all_steps_dir"] = os.path.join(run_dir, 'all_steps')
        os.makedirs(processed_data["all_steps_dir"], exist_ok=True)
        
        original_image_name = f"original_{file.filename}"
        original_image_path = os.path.join(UPLOAD_FOLDER, original_image_name)
        file.save(original_image_path)
        print(f"🖼️  Saved original image to {original_image_path}")
        
        processed_data["original_image_name"] = original_image_name 

        python_executable = sys.executable 
        input_image_full_path = os.path.abspath(original_image_path)
        command = [python_executable, DFCALL_SCRIPT_PATH, input_image_full_path]
        
        print(f"✅ Subprocess: กำลังรันสคริปต์ (Detached)...")
        
        subprocess.Popen(
            command,
            cwd=DFCALL_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True 
        )
        
        print(f"--- [app.py] DFCall is running in background. ---")
        
        drawing_state["message"] = "Processing DFCall... (Polling)"
        return jsonify({
            "status": "processing_started",
            "message": "DFCall started. Polling for result..."
        })

    except Exception as e:
        drawing_state["status"] = "idle"
        drawing_state["message"] = f"Error: {e}"
        print(f"❌ /process_image Error (Pre-run): {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/check_processing', methods=['GET'])
def check_processing():
    global processed_data
    
    try:
        if not os.path.exists(DFCALL_OUTPUT_IMAGE_PATH):
            return jsonify({"status": "processing", "message": "DFCall is running..."})

        print(f"✅ DFCall Success: พบไฟล์ผลลัพธ์!")
        drawing_state["message"] = "DFCall complete. Processing comparison..."
        
        run_dir = processed_data["current_run_dir"]
        run_dir_basename = os.path.basename(run_dir)
        original_image_name = processed_data["original_image_name"]

        bw_image_path = os.path.join(run_dir, "processed_bw_image.jpg")
        shutil.copy(DFCALL_OUTPUT_IMAGE_PATH, bw_image_path)
        os.remove(DFCALL_OUTPUT_IMAGE_PATH) 
        
        print(f"⬛️⬜️ คัดลอกภาพ (dfcall) มาที่: {bw_image_path}")
        
        img_color = cv2.imread(bw_image_path)
        if img_color is None:
            raise Exception(f"Could not read B&W image at {bw_image_path}")
            
        original_h, original_w = img_color.shape[:2]
        scale_factor = ddl.IMAGE_MAX_SIZE / max(original_h, original_w)
        target_w = int(original_w * scale_factor)
        target_h = int(original_h * scale_factor)
        img_color_resized = cv2.resize(img_color, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img_gray_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2GRAY)
        processed_data["img_gray_resized"] = img_gray_resized.copy()
        processed_data["base_bgr_image"] = cv2.cvtColor(img_gray_resized, cv2.COLOR_GRAY2BGR)

        drawing_state["message"] = "Generating parameter comparison..."
        comparison_image_path = ddl.visualize_parameters(
            img_color_resized, 
            img_gray_resized.copy(), 
            ddl.TEST_PARAMS, 
            run_dir 
        )
        print(f"📊 Saved comparison sheet to {comparison_image_path}")

        drawing_state["status"] = "idle"
        drawing_state["message"] = "Ready for parameter selection"
        
        return jsonify({
            "status": "success",
            "message": "Processing complete. Please select parameters.",
            "original_url": f"{UPLOAD_FOLDER}/{original_image_name}".replace(os.path.sep, '/'),
            "bw_image_url": f"{OUTPUT_FOLDER}/{run_dir_basename}/processed_bw_image.jpg".replace(os.path.sep, '/'), 
            "comparison_url": f"{OUTPUT_FOLDER}/{run_dir_basename}/parameter_comparison.jpg".replace(os.path.sep, '/')
        })

    except Exception as e:
        drawing_state["status"] = "idle"
        drawing_state["message"] = f"Error: {e}"
        print(f"❌ /check_processing Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/select_parameters', methods=['POST'])
def select_parameters():
    global processed_data
    data = request.json
    choice_index = data.get('choice_index') 
    if choice_index is None or not (0 <= choice_index < len(TEST_PARAMS)):
        return jsonify({"status": "error", "message": "Invalid parameter choice"}), 400
    if processed_data["img_gray_resized"] is None:
        return jsonify({"status": "error", "message": "No image processed yet. Please /process_image first."}), 400
    drawing_state["status"] = "processing"
    drawing_state["message"] = "Generating contours..."
    try:
        selected_params = TEST_PARAMS[choice_index]
        name, blur, block, c, eps, min_area = selected_params
        print(f"⚙️ User selected parameters: {name}")
        preview_img_bgr, filtered_contours, total_length_mm = ddl.process_and_draw_contours(
            processed_data["img_gray_resized"].copy(),
            blur_ksize=blur, thresh_blocksize=block, thresh_c=c, 
            epsilon_factor=eps, min_contour_area=min_area
        )
        if not filtered_contours:
            raise Exception("No contours found with these parameters.")
        
        run_dir_basename = os.path.basename(processed_data["current_run_dir"])
        lineart_path = os.path.join(processed_data["current_run_dir"], "final_lineart.jpg")
        cv2.imwrite(lineart_path, preview_img_bgr)
        print(f"✒️ Saved final lineart to {lineart_path}")

        drawing_state["message"] = "Generating all_steps previews..."
        base_bgr = processed_data["base_bgr_image"]
        all_steps_dir = processed_data["all_steps_dir"]
        current_run_dir = processed_data["current_run_dir"]
        for ci in range(1, len(filtered_contours) + 1):
            ddl.create_progress_image(
                base_bgr, filtered_contours, ci, is_final=False,
                output_all_steps_path=all_steps_dir,
                output_current_run_path=current_run_dir
            )
        ddl.create_progress_image(
            base_bgr, filtered_contours, len(filtered_contours) + 1, is_final=True,
            output_all_steps_path=all_steps_dir,
            output_current_run_path=current_run_dir
        )
        print(f"✅ Generated {len(filtered_contours)} all_steps images in {all_steps_dir}")
        img_h, img_w = processed_data["img_gray_resized"].shape 
        img_corners = np.float32([[0, 0], [img_w-1, 0], [img_w-1, img_h-1], [0, img_h-1]])
        current_paper_corners = ddl.load_calibration() 
        M = cv2.getPerspectiveTransform(img_corners, current_paper_corners) 
        processed_paths = []
        contour_lengths = []
        for cnt in filtered_contours:
            pts = np.array(cnt, dtype=np.float32).reshape(-1, 1, 2)
            pts_transformed = cv2.perspectiveTransform(pts, M) 
            processed_paths.append(pts_transformed)
            length = np.sum(np.sqrt(np.sum(np.diff(pts_transformed.reshape(-1, 2), axis=0)**2, axis=1)))
            contour_lengths.append(length)
        processed_data["filtered_contours"] = filtered_contours
        processed_data["processed_paths"] = processed_paths
        processed_data["contour_lengths"] = contour_lengths
        processed_data["total_contours"] = len(filtered_contours)
        drawing_state["status"] = "idle"
        drawing_state["message"] = "Ready to select start contour"
        
        lineart_url = f"{OUTPUT_FOLDER}/{run_dir_basename}/final_lineart.jpg"
        return jsonify({
            "status": "success",
            "message": "Paths generated. Ready to draw.",
            "lineart_url": lineart_url.replace(os.path.sep, '/'),
            "total_contours": len(filtered_contours)
        })
    except Exception as e:
        drawing_state["status"] = "idle"
        drawing_state["message"] = f"Error: {e}"
        print(f"❌ /select_parameters Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def drawing_thread_task(start_contour_index, pen_down_z, pen_up_z, home_x, home_y):
    global drawing_state, processed_data
    try:
        drawing_state["status"] = "drawing"
        drawing_state["message"] = "Initializing..."
        drawing_state["progress"] = 0
        drawing_state["stop_flag"] = False
        base_bgr = processed_data["base_bgr_image"]
        current_run_dir = processed_data["current_run_dir"]
        run_dir_basename = os.path.basename(current_run_dir)
        
        contours_to_draw = list(processed_data["filtered_contours"])
        paths_to_draw = list(processed_data["processed_paths"])
        lengths_to_draw = list(processed_data["contour_lengths"])
        total_contours = len(paths_to_draw)
        total_length_to_draw = sum(lengths_to_draw)
        start_index = start_contour_index - 1
        if start_index != 0:
            print(f"🔄 Re-ordering drawing. Starting at {start_contour_index}")
            contours_to_draw = contours_to_draw[start_index:] + contours_to_draw[:start_index]
            paths_to_draw = paths_to_draw[start_index:] + paths_to_draw[:start_index]
            lengths_to_draw = lengths_to_draw[start_index:] + lengths_to_draw[:start_index]
            original_indices = list(range(start_index, total_contours)) + list(range(0, start_index))
        else:
            original_indices = list(range(total_contours))
        ddl.safe_move(bot, home_x, home_y, pen_up_z, wait=True)
        start_time = time.time()
        current_length_drawn = 0
        
        progress_img_path = os.path.join(current_run_dir, "current_progress_drawing.jpg")
        progress_img_url_base = f"{OUTPUT_FOLDER}/{run_dir_basename}/current_progress_drawing.jpg".replace(os.path.sep, '/')

        print(f"🚀 Start Drawing: {total_contours} contours, Total Length: {total_length_to_draw:.2f} mm")

        for i in range(total_contours):
            if drawing_state["stop_flag"]:
                drawing_state["message"] = "Drawing stopped"
                print("🛑 Drawing interrupted by User.")
                break
            while drawing_state["status"] == "paused":
                if drawing_state["stop_flag"]:
                    break
                time.sleep(0.2) 
            if drawing_state["stop_flag"]: 
                drawing_state["message"] = "Drawing stopped"
                break
            pts_transformed = paths_to_draw[i]
            if pts_transformed is None or len(pts_transformed) < 2:
                continue
            ci_original = original_indices[i] + 1
            ci_loop = i + 1
            
            ddl.update_current_progress_image(
                base_bgr, contours_to_draw, ci_loop, is_final=False,
                output_filename=progress_img_path
            )
            
            drawing_state["progress_image_url"] = f"{progress_img_url_base}?t={time.time()}"
            
            current_length_drawn += lengths_to_draw[i]
            percent_done = (current_length_drawn / total_length_to_draw) * 100 if total_length_to_draw > 0 else 0
            drawing_state["progress"] = round(percent_done, 1)
            eta_display = ddl.get_eta_display(start_time, current_length_drawn, total_length_to_draw)
            drawing_state["message"] = f"Drawing {ci_loop}/{total_contours} (Orig #{ci_original}) | {eta_display}"
            
            elapsed_now = time.time() - start_time
            print(f"🖌️ [{elapsed_now:.1f}s] Drawing Contour {ci_loop}/{total_contours} (Len: {lengths_to_draw[i]:.1f}mm) | Total: {percent_done:.1f}% | {eta_display}")

            sx, sy = pts_transformed[0][0]
            ddl.safe_move(bot, sx, sy, pen_up_z, wait=False)
            ddl.safe_move(bot, sx, sy, pen_down_z, wait=True) 
            x_last, y_last = sx, sy 
            for p in pts_transformed[1:]:
                x_last, y_last = p[0] 
                ddl.safe_move(bot, x_last, y_last, pen_down_z, wait=False) 
            ddl.safe_move(bot, x_last, y_last, pen_down_z, wait=True)
            ddl.safe_move(bot, x_last, y_last, pen_up_z, wait=False) 
        
        if drawing_state["stop_flag"]:
            print("🛑 Drawing stopped by user.")
            drawing_state["message"] = "Drawing stopped"
            bot.stop_queue()
            bot.clear_queue()
            time.sleep(0.5)
            pose = bot.pose()
            ddl.safe_move(bot, pose[0], pose[1], pose[2] + 20, wait=True)
        else:
            end_time = time.time()
            total_seconds = end_time - start_time
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            
            time_str = ""
            if hours > 0: time_str += f"{hours}h "
            if minutes > 0: time_str += f"{minutes}m "
            time_str += f"{seconds}s"
            
            print("\n" + "="*50)
            print(f"🎉 Drawing Finished Successfully!")
            print(f"⏱️  Total Drawing Time: {time_str} ({total_seconds:.2f} seconds)")
            print("="*50 + "\n")
            drawing_state["message"] = "Drawing complete!"
            drawing_state["progress"] = 100
            ddl.update_current_progress_image(
                base_bgr, contours_to_draw, total_contours + 1, is_final=True,
                output_filename=progress_img_path
            )
            drawing_state["progress_image_url"] = f"{progress_img_url_base}?t={time.time()}"
        ddl.safe_move(bot, home_x, home_y, pen_up_z, wait=True) 
    except Exception as e:
        print(f"❌ ERROR in drawing thread: {e}")
        drawing_state["status"] = "error"
        drawing_state["message"] = f"Error: {e}"
    finally:
        if drawing_state["status"] != "error":
            drawing_state["status"] = "idle"
        drawing_state["stop_flag"] = False

@app.route('/start_drawing', methods=['POST'])
def start_drawing():
    global drawing_thread
    if not bot:
        return jsonify({"status": "error", "message": "Dobot not connected"}), 400
    if drawing_thread and drawing_thread.is_alive():
        return jsonify({"status": "error", "message": "Already drawing"}), 400
    if not processed_data.get("processed_paths"):
        return jsonify({"status": "error", "message": "No paths generated. Please select parameters first."}), 400
    data = request.json
    try:
        start_contour = int(data.get('start_contour', 1))
        speed_percent = float(data.get('speed', 50))
        pen_offset = float(data.get('pen_offset', 0))
        safety_height = float(data.get('safety_height', 10))
        if not (1 <= start_contour <= processed_data["total_contours"]):
            raise ValueError(f"Start contour must be between 1 and {processed_data['total_contours']}")
        dobot_speed_val = (speed_percent / 100.0) * DOBOT_SPEED
        dobot_accel_val = (speed_percent / 100.0) * DOBOT_ACCELERATION
        dobot_speed_val = max(100, min(dobot_speed_val, DOBOT_SPEED))
        dobot_accel_val = max(100, min(dobot_accel_val, DOBOT_ACCELERATION))
        bot.speed(dobot_speed_val, dobot_accel_val)
        
        # ใช้ค่าจาก ddl.PEN_DOWN_Z (ล่าสุด)
        pen_down_z = ddl.PEN_DOWN_Z + pen_offset 
        if safety_height is None: safety_height = 20.0
        pen_up_z = safety_height 
        
        current_paper_corners = ddl.load_calibration()
        home_x, home_y = current_paper_corners[0] 
        print(f"▶️ Starting drawing... Speed: {speed_percent}% ({dobot_speed_val:.0f}), Start: #{start_contour}")
        
        print(f"--- Z-HEIGHT DEBUG ---")
        print(f"  Base PEN_DOWN_Z (from Logic): {ddl.PEN_DOWN_Z}")
        print(f"  Offset (from Web): {pen_offset}")
        print(f"  Final Z: {pen_down_z}")
        print(f"----------------------")

        drawing_thread = threading.Thread(
            target=drawing_thread_task, 
            args=(start_contour, pen_down_z, pen_up_z, home_x, home_y)
        )
        drawing_thread.start()
        return jsonify({"status": "success", "message": "Drawing started..."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/progress', methods=['GET'])
def get_progress():
    return jsonify(drawing_state)

@app.route('/pause', methods=['POST'])
def pause_drawing():
    if drawing_thread and drawing_thread.is_alive() and drawing_state["status"] == "drawing":
        drawing_state["status"] = "paused"
        drawing_state["message"] = "Paused"
        # bot.pause(True)
        print("⏸️ Drawing paused")
    return jsonify({"status": "success", "message": "Paused"})

@app.route('/resume', methods=['POST'])
def resume_drawing():
    if drawing_thread and drawing_thread.is_alive() and drawing_state["status"] == "paused":
        drawing_state["status"] = "drawing"
        drawing_state["message"] = "Resuming..."
        # bot.pause(False)
        print("▶️ Drawing resumed")
    return jsonify({"status": "success", "message": "Resumed"})

@app.route('/stop', methods=['POST'])
def stop_drawing():
    drawing_state["stop_flag"] = True 
    if drawing_state["status"] == "paused":
        drawing_state["status"] = "drawing" 
    print("🛑 Stop signal sent.")
    return jsonify({"status": "success", "message": "Stop signal sent"})

if __name__ == '__main__':
    print("======================================================")
    print(" 🤖 Dobot Drawing Web Server 🤖")
    print(f" 🌍 Local: http://127.0.0.1:{PORT}") 
    print(f" 📲 Mobile: http://{MY_IP}:{PORT}") 
    print("======================================================")
    app.run(host='0.0.0.0', port=PORT, debug=False)