# --- ⭐️ บังคับใช้โหมด Agg (Non-GUI) เพื่อป้องกันเซิร์ฟเวอร์แครช ⭐️ ---
import matplotlib
matplotlib.use('Agg')
# --------------------------------------------------

import cv2
import numpy as np
try:
    from pydobot import Dobot
except ImportError:
    Dobot = None

import time
import os
import matplotlib.pyplot as plt
import json 
import glob
import sys
import shutil
import math

# ================== ตั้งค่า (CONFIG) ==================
OUTPUT_DIR_BASE = 'static/processed' 
EXP_PREFIX = 'exp_' 
# ---------------------------------------------

IMAGE_MAX_SIZE = 1000

# ระดับปากกา
PEN_DOWN_Z = -57  
PEN_UP_Z = -35    

RETRY_ATTEMPTS = 3
DOBOT_SPEED = 3200
DOBOT_ACCELERATION = 2000

# ระยะห่างที่จะดูดเส้นเข้าหากัน
MERGE_DISTANCE_THRESHOLD = 20 

#ความหนาแน่นของการถมดำ (หน่วยพิกเซล)
# ค่า 1 = ถมละเอียดสุด (ดำปึด), ค่า 2 = เร็วขึ้นแต่จางลงนิดหน่อย
FILL_DENSITY = 1 

# ชุดพารามิเตอร์สำหรับทดสอบ
TEST_PARAMS = [
    ("Solid Eyes (Concentric)", 5, 11, 7, 0.0010, 10), # ⭐️ แนะนำอันนี้
    ("Default (Fine)", 5, 11, 7, 0.0015, 1),      
    ("High Detail", 3, 9, 5, 0.00075, 3),         
    ("Smooth Lines", 9, 15, 10, 0.002, 5),        
    ("Aggressive", 5, 11, 2, 0.0005, 1)           
]

CALIBRATION_FILE = 'dobot_calibration.json'

PAPER_CORNERS_DEFAULT = np.float32([
    [1.69, 96.04],      # มุมบนซ้าย
    [134.10, 215.25],   # มุมบนขวา
    [264.16, 28.42],    # มุมล่างขวา
    [106.29, -51.89]    # มุมล่างซ้าย
])

# ----------------- ฟังก์ชันช่วยเหลือทั่วไป -----------------

def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                corners_list = json.load(f)
                if len(corners_list) == 4 and all(len(c) == 2 for c in corners_list):
                    print(f"✅ โหลดค่า Calibration จาก {CALIBRATION_FILE}")
                    return np.float32(corners_list)
        except Exception:
            pass
    return PAPER_CORNERS_DEFAULT

PAPER_CORNERS = load_calibration()

def find_dobot_port():
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    print("\n🔍 DEBUG: รายชื่อ Port ที่เจอ:")
    for p in ports:
        print(f"   - Device: {p.device}, Desc: {p.description}")
        if not hasattr(p, 'description') or not hasattr(p, 'device'): continue
        is_dobot = "USB" in p.description.upper() or \
                   "SERIAL" in p.description.upper() or \
                   "CH340" in p.description.upper() or \
                   "CP210" in p.description.upper() or \
                   "USB" in p.device.upper()
        if is_dobot:
            print(f"✅ เลือกใช้ Port: {p.device}")
            return p.device
    print("❌ ไม่พบ Port ที่เข้าข่าย")
    return None

def safe_move(bot, x, y, z, r=0, wait=True):
    for i in range(RETRY_ATTEMPTS):
        try:
            bot.move_to(x, y, z, r, wait=wait)
            return True
        except Exception:
            time.sleep(0.1)
    return False

def get_next_experiment_dir():
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    existing_dirs = glob.glob(os.path.join(OUTPUT_DIR_BASE, f'{EXP_PREFIX}[0-9]*'))
    max_num = 0
    for dir_path in existing_dirs:
        try:
            num_str = os.path.basename(dir_path).replace(EXP_PREFIX, '')
            max_num = max(max_num, int(num_str))
        except ValueError:
            continue
    next_num = max_num + 1
    new_exp_dir = os.path.join(OUTPUT_DIR_BASE, f'{EXP_PREFIX}{next_num}')
    
    os.makedirs(os.path.join(new_exp_dir, 'all_steps'), exist_ok=True)
    os.makedirs(os.path.join(new_exp_dir, 'current_run'), exist_ok=True)
    
    print(f"✅ สร้างโฟลเดอร์งานใหม่: {new_exp_dir}/")
    return new_exp_dir 

def create_progress_image(base_img_bgr, filtered_contours, current_contour_index, is_final, 
                          output_all_steps_path, output_current_run_path):
    preview = base_img_bgr.copy()
    if current_contour_index > 1:
        cv2.drawContours(preview, filtered_contours[:current_contour_index-1], -1, (255, 0, 0), 1) 
    if not is_final and current_contour_index <= len(filtered_contours):
        cv2.drawContours(preview, [filtered_contours[current_contour_index-1]], -1, (0, 255, 0), 2)
    
    if not is_final:
        filename_all = os.path.join(output_all_steps_path, f"step_{current_contour_index:04d}_drawing.jpg")
        cv2.imwrite(filename_all, preview)
        
    filename_current = os.path.join(output_current_run_path, f"current_progress_{'done' if is_final else 'drawing'}.jpg")
    cv2.imwrite(filename_current, preview)

def update_current_progress_image(base_img_bgr, filtered_contours, current_contour_index, is_final, output_filename):
    preview = base_img_bgr.copy()
    if current_contour_index > 1:
        cv2.drawContours(preview, filtered_contours[:current_contour_index-1], -1, (255, 0, 0), 1) 
    if not is_final and current_contour_index <= len(filtered_contours):
        cv2.drawContours(preview, [filtered_contours[current_contour_index-1]], -1, (0, 255, 0), 2)
    cv2.imwrite(output_filename, preview)

# --- ⭐️ Skeletonize ⭐️ ---
def skeletonize(img):
    img = img.copy()
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0: break
    return skel

# --- ⭐️ RDP Simplification ⭐️ ---
def simplify_path_rdp(path, epsilon=2.0):
    if len(path) < 3: return path
    simplified = cv2.approxPolyDP(path, epsilon, False) 
    return simplified

# --- ⭐️ Optimization ⭐️ ---
def sort_and_merge_contours(contours, threshold=MERGE_DISTANCE_THRESHOLD):
    if not contours: return []
    unvisited = [c for c in contours]
    ordered_paths = []
    current_path = unvisited.pop(0)
    
    while True:
        current_end_point = current_path[-1][0]
        best_dist = float('inf')
        best_idx = -1
        should_reverse = False
        
        for i, p in enumerate(unvisited):
            start_p = p[0][0]
            end_p = p[-1][0]
            dist_start = np.linalg.norm(current_end_point - start_p)
            dist_end = np.linalg.norm(current_end_point - end_p)
            
            if dist_start < best_dist:
                best_dist = dist_start
                best_idx = i
                should_reverse = False
            if dist_end < best_dist:
                best_dist = dist_end
                best_idx = i
                should_reverse = True
        
        if best_idx != -1:
            next_path = unvisited[best_idx]
            if best_dist < threshold:
                if should_reverse: next_path = next_path[::-1]
                current_path = np.vstack((current_path, next_path))
                unvisited.pop(best_idx)
            else:
                current_path = simplify_path_rdp(current_path, epsilon=2.0)
                ordered_paths.append(current_path)
                current_path = unvisited.pop(best_idx)
                if should_reverse: current_path = current_path[::-1]
        else:
            current_path = simplify_path_rdp(current_path, epsilon=2.0)
            ordered_paths.append(current_path)
            if unvisited: current_path = unvisited.pop(0)
            else: break
    return ordered_paths

# --- ⭐️ NEW FEATURE: Concentric Fill (ถมดำแบบวนเข้าใน) ⭐️ ---
def generate_concentric_fill(binary_mask, step_size=FILL_DENSITY):
    """
    สร้างเส้นวนเข้าใน (Inward Spiraling) เพื่อถมดำสนิท
    binary_mask: ภาพขาวดำพื้นที่ที่จะถม
    step_size: จำนวนพิกเซลที่จะหดลงในแต่ละรอบ (1 = ละเอียดสุด)
    """
    fill_contours = []
    temp_mask = binary_mask.copy()
    
    # Kernel สำหรับการหดพื้นที่ (Erosion)
    # ใช้ Cross shape เพื่อให้หดตัวสม่ำเสมอ
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    loop_count = 0
    max_loops = 200 # กัน Loop ไม่รู้จบ
    
    while True:
        if loop_count >= max_loops: break
        
        # หาเส้นขอบของพื้นที่ปัจจุบัน
        contours, _ = cv2.findContours(temp_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: break
        
        added_any = False
        for cnt in contours:
            if cv2.contourArea(cnt) > 5: # กรองจุดเล็กๆ ทิ้ง
                # ทำให้เส้นเนียนขึ้นนิดหน่อยก่อนเก็บ
                approx = cv2.approxPolyDP(cnt, 0.5, False)
                fill_contours.append(approx)
                added_any = True
        
        if not added_any: break
        
        # หดพื้นที่ลง (Erode) เพื่อทำรอบถัดไป
        # ทำซ้ำ step_size รอบ (เช่นถ้า step=2 ก็ erode 2 ที)
        for _ in range(step_size):
            temp_mask = cv2.erode(temp_mask, kernel)
            
        # ถ้าไม่มีพื้นที่ขาวเหลือแล้ว ก็จบ
        if cv2.countNonZero(temp_mask) == 0:
            break
            
        loop_count += 1
            
    return fill_contours

# --- ⭐️ LOGIC หลัก ⭐️ ---
def process_and_draw_contours(img_gray, blur_ksize, thresh_blocksize, thresh_c, epsilon_factor, min_contour_area):
    if blur_ksize % 2 == 0: blur_ksize += 1
    
    # 1. Blur ภาพ
    img_blurred = cv2.GaussianBlur(img_gray, (blur_ksize, blur_ksize), 0)
    
    # ============ A. ส่วนของเส้นขอบ (Outline) ============
    if thresh_blocksize % 2 == 0: thresh_blocksize += 1
    if thresh_blocksize < 3: thresh_blocksize = 3
     
    thresh = cv2.adaptiveThreshold(
         img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
         cv2.THRESH_BINARY_INV, thresh_blocksize, thresh_c
    )
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh_dilated = cv2.dilate(thresh, kernel_dilate, iterations=1)
    thresh_dilated = cv2.morphologyEx(thresh_dilated, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

    thinned = skeletonize(thresh_dilated)
    contours_outline, _ = cv2.findContours(thinned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    final_contours = []
    for cnt in contours_outline:
        if cv2.contourArea(cnt) < min_contour_area: continue
        if cv2.arcLength(cnt, False) < 10: continue
        approx = cv2.approxPolyDP(cnt, 0.0005 * cv2.arcLength(cnt, False), False)
        if len(approx) >= 2:
            final_contours.append(approx)
            
    # ============ B. ⭐️ ส่วนของการถมดำสนิท (Solid Fill) ============
    # ใช้ค่า Threshold ที่เข้มข้นขึ้น (ต่ำกว่า 80) เพื่อเลือกเฉพาะส่วนที่ดำจริงๆ
    _, mask_fill = cv2.threshold(img_blurred, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Clean Noise
    mask_fill = cv2.morphologyEx(mask_fill, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    # กรองเฉพาะก้อนที่มีขนาดเหมาะสม (ตาดำ/คิ้ว)
    fill_contours_raw, _ = cv2.findContours(mask_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_fill_filtered = np.zeros_like(mask_fill)
    
    for cnt in fill_contours_raw:
        area = cv2.contourArea(cnt)
        # ⭐️ ปรับขนาดกรอง: เล็กสุด 15 px (จุด), ใหญ่สุด 6000 px (กันถมผมทั้งหัว)
        if 15 < area < 6000: 
            cv2.drawContours(mask_fill_filtered, [cnt], -1, 255, -1) 
            
    # ⭐️ เรียกใช้ฟังก์ชันถมดำแบบวนเข้าใน (Concentric)
    solid_fill_lines = generate_concentric_fill(mask_fill_filtered, step_size=FILL_DENSITY)
    
    print(f"🧩 Found {len(solid_fill_lines)} concentric fill paths.")
    
    # รวมเส้นเข้าด้วยกัน
    final_contours.extend(solid_fill_lines)

    # ============ C. Optimize ============
    optimized_contours = sort_and_merge_contours(final_contours, threshold=MERGE_DISTANCE_THRESHOLD)
    
    # Preview
    preview_img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview_img_bgr, optimized_contours, -1, (0, 0, 255), 1)
    
    # ไฮไลท์ส่วนที่ถมดำใน Preview
    mask_vis = cv2.cvtColor(mask_fill_filtered, cv2.COLOR_GRAY2BGR)
    preview_img_bgr = cv2.addWeighted(preview_img_bgr, 1.0, mask_vis, 0.4, 0)
    
    return preview_img_bgr, optimized_contours, 0

def visualize_parameters(original_img_color, original_img_gray, test_params, output_dir):
    fig, axs = plt.subplots(3, 2, figsize=(8.27, 11.69)) 
    axs = axs.flatten()
    axs[0].imshow(cv2.cvtColor(original_img_color, cv2.COLOR_BGR2RGB))
    axs[0].set_title("1. Original Image (BGR)", fontsize=10, fontweight='bold')
    axs[0].axis("off")
    
    all_test_params = TEST_PARAMS
    
    for i, (name, blur, block, c, eps, min_area) in enumerate(all_test_params, start=1):
        if i >= len(axs): break
            
        processed_img_bgr, _, _ = process_and_draw_contours(
            original_img_gray.copy(), 
            blur_ksize=blur, 
            thresh_blocksize=block, 
            thresh_c=c, 
            epsilon_factor=eps, 
            min_contour_area=min_area
        )
        
        axs[i].imshow(cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"{i+1}. {name}", fontsize=8)
        axs[i].axis("off")
        
    for i in range(len(all_test_params) + 1, len(axs)):
        fig.delaxes(axs[i])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.suptitle("Dobot Drawing Parameter Comparison", fontsize=16, fontweight='bold')
    
    output_filename = os.path.join(output_dir, "parameter_comparison.jpg")
    plt.savefig(output_filename, dpi=200) 
    plt.close(fig) 
    print(f"✅ บันทึกภาพเปรียบเทียบที่: {output_filename}")
    
    return output_filename 

def get_eta_display(start_time, current_length_drawn, total_length_to_draw):
    elapsed_time = time.time() - start_time
    eta_display = "ETA: Calculating..."
    if elapsed_time > 5 and current_length_drawn > 10 and current_length_drawn < total_length_to_draw: 
        try:
            avg_speed_mm_per_sec = current_length_drawn / elapsed_time 
            remaining_length = total_length_to_draw - current_length_drawn
            eta_seconds = remaining_length / avg_speed_mm_per_sec
            eta_minutes = eta_seconds / 60
            eta_display = f"ETA: {eta_minutes:.1f} min"
        except ZeroDivisionError:
            eta_display = "ETA: Error"
    elif current_length_drawn >= total_length_to_draw:
        eta_display = "ETA: Done"
    return eta_display

print("✅ dobot_drawing_logic.py loaded.")