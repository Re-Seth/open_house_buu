import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
import serial.tools.list_ports
from pydobot import Dobot
import time
import os
import matplotlib.pyplot as plt # (import นี้ต้องอยู่ *หลัง* .use('Agg'))
import shutil 
import glob 
import sys 
import subprocess 
import math 
import json 

# ================== CONFIG ==================
OUTPUT_DIR_BASE = 'static/processed' 
EXP_PREFIX = 'exp_' 
# ---------------------------------------------

IMAGE_MAX_SIZE = 1000
PEN_DOWN_Z = -57
PEN_UP_Z = -50
RETRY_ATTEMPTS = 3

DOBOT_SPEED = 3200
DOBOT_ACCELERATION = 2000
EPSILON = 0.0015
MIN_CONTOUR_AREA = 1

TEST_PARAMS = [
    # (Name, Blur, ThreshBlock, ThreshC, Epsilon, MinArea)
    ("Default (Fine)", 5, 11, 7, 0.0015, 1),
    ("High Detail (Slower)", 3, 9, 5, 0.00075, 3),
    ("Smooth Lines", 9, 15, 10, 0.002, 5),
    ("Coarse Detail", 5, 21, 5, 0.0002, 10),
    ("Aggressive Thresh", 5, 11, 2, 0.0005, 1)
]

CALIBRATION_FILE = 'dobot_calibration.json'

PAPER_CORNERS_DEFAULT = np.float32([
    [1.69, 96.04],      # top-left
    [134.10, 215.25],   # top-right
    [264.16, 28.42],    # bottom-right
    [106.29, -51.89]    # bottom-left
])
# =============================================

# ----------------- Utility Functions -----------------

def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                corners_list = json.load(f)
                if len(corners_list) == 4 and all(len(c) == 2 for c in corners_list):
                    print(f"✅ โหลดค่า Calibration ล่าสุดจาก {CALIBRATION_FILE}")
                    return np.float32(corners_list)
                else:
                    print(f"⚠️ ไฟล์ {CALIBRATION_FILE} มีรูปแบบไม่ถูกต้อง, ใช้ค่า Default")
                    return PAPER_CORNERS_DEFAULT
        except Exception as e:
            print(f"⚠️ ไม่สามารถโหลด {CALIBRATION_FILE}: {e}. ใช้ค่า Default แทน")
            return PAPER_CORNERS_DEFAULT
    else:
        print(f"ℹ️ ไม่พบไฟล์ {CALIBRATION_FILE}, ใช้ค่า Default ในโค้ด")
        return PAPER_CORNERS_DEFAULT

PAPER_CORNERS = load_calibration()


def find_dobot_port():
    ports = serial.tools.list_ports.comports()
    dobot_port = None
    for p in ports:
        if not hasattr(p, 'description') or not hasattr(p, 'device'):
            continue
        is_dobot = "USB" in p.description.upper() or \
                   "SERIAL" in p.description.upper() or \
                   "CH340" in p.description.upper() or \
                   "CP210" in p.description.upper()
        is_dobot = is_dobot or \
                   "MODEM" in p.device.upper() or \
                   "USB" in p.device.upper() or \
                   "WCHUSB" in p.device.upper()
        if is_dobot:
            print(f"✅ พบพอร์ตที่น่าจะเป็น Dobot: {p.device} ({p.description})")
            dobot_port = p.device
            break
    if not dobot_port:
        print("\n⚠️ ไม่พบ Dobot โดยอัตโนมัติ")
        all_ports = [f"  - {p.device} ({getattr(p, 'description', 'N/A')})" for p in ports if hasattr(p, 'device')]
        if all_ports:
            print("\n".join(all_ports))
        else:
            print("❌ ไม่พบพอร์ต Serial ใด ๆ เลย")
        return None  
    return dobot_port

def safe_move(bot, x, y, z, r=0, wait=True):
    for i in range(RETRY_ATTEMPTS):
        try:
            bot.move_to(x, y, z, r, wait=wait)
            return True
        except Exception as e:
            if i < RETRY_ATTEMPTS - 1:
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
    
    print(f"✅ สร้างโฟลเดอร์ทดลองใหม่: {new_exp_dir}/")
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

def update_current_progress_image(base_img_bgr, filtered_contours, current_contour_index, is_final,
                                  output_filename):
    preview = base_img_bgr.copy()
    if current_contour_index > 1:
        cv2.drawContours(preview, filtered_contours[:current_contour_index-1], -1, (255, 0, 0), 1) 
    if not is_final and current_contour_index <= len(filtered_contours):
        cv2.drawContours(preview, [filtered_contours[current_contour_index-1]], -1, (0, 255, 0), 2)
    
    cv2.imwrite(output_filename, preview)


def process_and_draw_contours(img_gray, blur_ksize, thresh_blocksize, thresh_c, epsilon_factor, min_contour_area):
    if blur_ksize % 2 == 0: blur_ksize += 1
    if blur_ksize < 3: blur_ksize = 3
    img = cv2.GaussianBlur(img_gray, (blur_ksize, blur_ksize), 0)
    
    if thresh_blocksize % 2 == 0: thresh_blocksize += 1
    if thresh_blocksize < 3: thresh_blocksize = 3
    thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, thresh_blocksize, thresh_c
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    total_length_mm = 0.0
    
    current_paper_corners = load_calibration()
    
    img_h, img_w = img_gray.shape
    img_corners = np.float32([[0, 0], [img_w-1, 0], [img_w-1, img_h-1], [0, img_h-1]])
    M = cv2.getPerspectiveTransform(img_corners, current_paper_corners) 

    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area:
            continue
        arc_length = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_factor * arc_length, True)
        if len(approx) >= 2:
            filtered_contours.append(approx)
            pts = np.array(approx, dtype=np.float32).reshape(-1, 1, 2)
            pts_transformed = cv2.perspectiveTransform(pts, M)
            length = np.sum(np.sqrt(np.sum(np.diff(pts_transformed.reshape(-1, 2), axis=0)**2, axis=1)))
            total_length_mm += length
            
    preview_img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview_img_bgr, filtered_contours, -1, (0, 0, 255), 1) 
    return preview_img_bgr, filtered_contours, total_length_mm


def visualize_parameters(original_img_color, original_img_gray, test_params, output_dir):
    # (โค้ดนี้จะรอดแล้ว เพราะ matplotlib.use('Agg') ถูกเรียกไปแล้ว)
    fig, axs = plt.subplots(3, 2, figsize=(8.27, 11.69)) 
    axs = axs.flatten()
    axs[0].imshow(cv2.cvtColor(original_img_color, cv2.COLOR_BGR2RGB))
    axs[0].set_title("1. Original Image (BGR)", fontsize=10, fontweight='bold')
    axs[0].axis("off")
    
    all_test_params = TEST_PARAMS
    
    for i, (name, blur, block, c, eps, min_area) in enumerate(all_test_params, start=1):
        if i >= len(axs): break
            
        processed_img_bgr, _, length_mm = process_and_draw_contours(
            original_img_gray.copy(), 
            blur_ksize=blur, 
            thresh_blocksize=block, 
            thresh_c=c, 
            epsilon_factor=eps, 
            min_contour_area=min_area
        )
        
        axs[i].imshow(cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB))
        params_text = f"B={blur}, T={block}, C={c}, E={eps*1000:.2f}e-3, MinA={min_area}"
        axs[i].set_title(
            f"{i+1}. {name}\n({params_text})", 
            fontsize=8
        )
        axs[i].axis("off")
        
    for i in range(len(all_test_params) + 1, len(axs)):
        fig.delaxes(axs[i])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.suptitle("Dobot Drawing Parameter Comparison (2x3 Grid)", fontsize=16, fontweight='bold')
    
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