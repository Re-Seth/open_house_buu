import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
import os
import sys

# นำเข้า Logic การวาดของเรา
try:
    import dobot_drawing_logic as ddl
except ImportError:
    print("❌ ไม่พบไฟล์ dobot_drawing_logic.py กรุณาวางไฟล์นี้ไว้ที่เดียวกัน")
    sys.exit(1)

# ================= ตั้งค่า =================
# ⚠️ ใส่ Path รูปที่คุณต้องการทดสอบที่นี่
INPUT_IMAGE_PATH = "/Users/student/Desktop/dobot_web_app/static/mobile_uploads/original_original_IMG_0339.JPG" 
# (หรือเปลี่ยนเป็นชื่อไฟล์รูปอื่นที่คุณมีในเครื่อง)

OUTPUT_PDF_NAME = "Dobot_100_Random_Tests.pdf"
TOTAL_SAMPLES = 100
SAMPLES_PER_PAGE = 5 # (1 Original + 5 Random per page)
# =========================================

def generate_random_params():
    """สุ่มค่าพารามิเตอร์สำหรับการวาด"""
    
    # 1. Blur (ต้องเป็นเลขคี่): 1, 3, 5, 7, 9
    blur = random.choice([1, 3, 5, 7, 9])
    
    # 2. Threshold Block Size (ต้องเป็นเลขคี่ > 1): 3 ถึง 51
    # ค่ามาก = สนใจพื้นที่กว้าง (เส้นสะอาด), ค่าน้อย = สนใจพื้นที่แคบ (เก็บรายละเอียด)
    block = random.randrange(3, 51, 2)
    
    # 3. Threshold C (ค่าคงที่ลบออก): 0 ถึง 20
    # ค่ามาก = เส้นน้อย/ขาด, ค่าน้อย = เส้นเยอะ/ขยะ
    c = random.randint(1, 20)
    
    # 4. Epsilon Factor (ความละเอียดเส้น): 0.0001 ถึง 0.005
    # ค่ามาก = เส้นเหลี่ยม/หยาบ, ค่าน้อย = เส้นยึกยือตามรอยเดิม
    epsilon = round(random.uniform(0.0001, 0.0030), 5)
    
    # 5. Min Area (กรองจุดเล็ก): 1 ถึง 50
    min_area = random.randint(1, 50)

    return (blur, block, c, epsilon, min_area)

def main():
    # 1. ตรวจสอบรูปภาพ
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"❌ ไม่พบไฟล์รูปภาพที่: {INPUT_IMAGE_PATH}")
        print("👉 กรุณาแก้ไขตัวแปร INPUT_IMAGE_PATH ในโค้ดบรรทัดที่ 18")
        return

    print(f"⏳ กำลังโหลดรูปภาพ: {INPUT_IMAGE_PATH}...")
    img_color = cv2.imread(INPUT_IMAGE_PATH)
    
    # Resize ภาพให้พอดีกับการประมวลผล (เหมือนใน app.py)
    original_h, original_w = img_color.shape[:2]
    scale_factor = ddl.IMAGE_MAX_SIZE / max(original_h, original_w)
    target_w = int(original_w * scale_factor)
    target_h = int(original_h * scale_factor)
    img_resized = cv2.resize(img_color, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    print(f"🚀 กำลังเริ่มสร้าง PDF: {OUTPUT_PDF_NAME}")
    print(f"📊 จำนวนแบบทดสอบทั้งหมด: {TOTAL_SAMPLES} แบบ")

    # คำนวณจำนวนหน้า (1 หน้ามี 5 แบบ)
    total_pages = math.ceil(TOTAL_SAMPLES / SAMPLES_PER_PAGE)

    with PdfPages(OUTPUT_PDF_NAME) as pdf:
        sample_count = 0
        
        for page in range(total_pages):
            print(f"   ...กำลังทำหน้า {page + 1}/{total_pages}")
            
            # สร้าง Layout 3x2 (6 ช่อง)
            fig, axs = plt.subplots(3, 2, figsize=(8.27, 11.69)) # ขนาด A4
            axs = axs.flatten()
            
            # ช่องที่ 1: รูปต้นฉบับ (เสมอ)
            axs[0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
            axs[0].set_title("Original Image", fontweight='bold')
            axs[0].axis("off")
            
            # ช่องที่ 2-6: รูปสุ่ม
            for i in range(1, 6):
                sample_count += 1
                if sample_count > TOTAL_SAMPLES:
                    axs[i].axis("off") # ซ่อนถ้าเกินจำนวน
                    continue

                # สุ่มค่า
                blur, block, c, eps, min_area = generate_random_params()
                
                # ประมวลผลภาพด้วยค่าที่สุ่มได้
                processed_img, _, _ = ddl.process_and_draw_contours(
                    img_gray.copy(),
                    blur_ksize=blur,
                    thresh_blocksize=block,
                    thresh_c=c,
                    epsilon_factor=eps,
                    min_contour_area=min_area
                )
                
                # แสดงผล
                axs[i].imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                
                # เขียนค่าพารามิเตอร์ใต้ภาพ
                title_text = f"#{sample_count} | B={blur}, Blk={block}, C={c}\nEps={eps}, MinA={min_area}"
                axs[i].set_title(title_text, fontsize=9, color='blue')
                axs[i].axis("off")
            
            # จัดระยะห่างและบันทึกลง PDF
            plt.tight_layout()
            plt.suptitle(f"Dobot Parameter Random Test (Page {page+1}/{total_pages})", fontsize=16)
            pdf.savefig(fig)
            plt.close(fig)
            
    print("\n" + "="*50)
    print(f"✅ สร้างไฟล์ PDF เสร็จสมบูรณ์!")
    print(f"📄 ไฟล์อยู่ที่: {os.path.abspath(OUTPUT_PDF_NAME)}")
    print("="*50)

if __name__ == "__main__":
    import math
    main()





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