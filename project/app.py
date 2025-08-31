import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
st.set_page_config(page_title="🍎 Apple Detector Pro", layout="wide")

# =========================
# CUSTOM CSS
st.markdown("""
<style>
body, .stApp { background-color: #0f111a; color: #ffffff; font-family: 'Arial', sans-serif; }
.hero { position: relative; background-image: url('demo_chart.jpg'); background-size: cover; background-position: center;
       height: 50vh; display: flex; flex-direction: column; justify-content: center; align-items: center; color: white;
       text-align: center; border-radius: 15px; margin-bottom: 40px; box-shadow: 0 0 20px rgba(0,0,0,0.5);}
.hero h1 { font-size: 3rem; font-weight: bold; text-shadow: 2px 2px 8px #000; }
.hero p { font-size: 1.2rem; margin-top: 1rem; text-shadow: 1px 1px 5px #000; }
.section { background-color: #1a1a2e; padding: 30px 20px; border-radius: 20px; margin-bottom: 30px; box-shadow: 0 0 20px rgba(0,0,0,0.4); }
.stButton>button { border-radius: 12px; padding: 8px 20px; font-weight: bold; background: linear-gradient(90deg,#00c6ff,#0072ff); color: #fff; border: none; transition: 0.3s; }
.stButton>button:hover { opacity: 0.8; cursor: pointer; }
.badge { display:inline-block; margin:5px 8px; padding:5px 12px; border-radius:12px; font-weight:bold; font-size:0.9rem; }
.ripe { background-color:#2ecc71; color:#fff; }
.unripe { background-color:#f1c40f; color:#fff; }
.overripe { background-color:#e74c3c; color:#fff; }
.stImage img { transition: all 0.3s ease-in-out; }
.stPlotlyChart { transition: all 0.3s ease-in-out; }
</style>
""", unsafe_allow_html=True)

# =========================
# HERO SECTION
st.markdown("""
<div class="hero">
    <h1>🍎 Apple Detector Pro</h1>
    <p>Detect apples and their ripeness in images or live camera feed reliably.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# EXAMPLE IMAGES
st.markdown('<div class="section"><h2>🍏 Example Apple Images</h2></div>', unsafe_allow_html=True)
example_images = ["demo_apple.jpg", "demo_chart.jpg", "demo_frame.jpg", "demo_query.jpg"]
cols = st.columns(4)
for i, col in enumerate(cols):
    if i < len(example_images):
        col.image(example_images[i], caption=f"Example {i+1}", use_container_width=True)

# =========================
# LOAD YOLO MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov8n.pt")  # nano model

# =========================
# HELPER FUNCTIONS
def predict_ripeness(img_crop):
    if img_crop.size == 0: return "Unknown"
    lab = cv2.cvtColor(img_crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    mean_a = np.mean(a)
    if mean_a > 150: return "Ripe"
    elif mean_a > 130: return "Overripe"
    else: return "Unripe"

def is_new_box(box, existing_boxes, threshold=0.3):
    x1, y1, x2, y2 = box
    for ex in existing_boxes:
        ex_x1, ex_y1, ex_x2, ex_y2 = ex
        ix1 = max(x1, ex_x1); iy1 = max(y1, ex_y1)
        ix2 = min(x2, ex_x2); iy2 = min(y2, ex_y2)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter_area = iw * ih
        union_area = (x2-x1)*(y2-y1) + (ex_x2-ex_x1)*(ex_y2-ex_y1) - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        if iou > threshold: return False
    return True

# =========================
# SESSION STATE
if "camera_running" not in st.session_state: st.session_state.camera_running = False
if "cumulative_counts" not in st.session_state: st.session_state.cumulative_counts = {"Ripe":0, "Unripe":0, "Overripe":0}
if "cumulative_boxes" not in st.session_state: st.session_state.cumulative_boxes = []

# =========================
# IMAGE UPLOAD
st.markdown('<div class="section"><h2>📷 Upload Image</h2></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an apple image", type=["jpg","png","jpeg"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    results = yolo_model.predict(img, device=device)
    annotated_img = img.copy()
    detected_boxes = []
    counts = {"Ripe":0, "Unripe":0, "Overripe":0}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if results[0].names[cls_id].lower() == "apple":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            apple_crop = img[y1:y2, x1:x2]
            ripeness = predict_ripeness(apple_crop)
            color = (0,255,0) if ripeness=="Ripe" else (0,165,255) if ripeness=="Unripe" else (0,0,255)
            (w,h), _ = cv2.getTextSize(ripeness, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(annotated_img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(annotated_img, ripeness, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.rectangle(annotated_img, (x1, y1),(x2, y2), color, 2)
            if is_new_box((x1,y1,x2,y2), detected_boxes): detected_boxes.append((x1,y1,x2,y2))
            counts[ripeness] +=1
    st.image(annotated_img, channels="BGR", use_container_width=True)
    st.markdown("<div>", unsafe_allow_html=True)
    for k,v in counts.items(): st.markdown(f"<span class='badge {k.lower()}'>{k}: {v}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# AUTO DETECT AVAILABLE CAMERAS
def get_available_cameras(max_tested=5):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened(): available.append(i); cap.release()
    return available

available_cameras = get_available_cameras()
st.markdown('<div class="section"><h2>🎥 Live Camera Detection</h2></div>', unsafe_allow_html=True)
cam_index = 1 if 1 in available_cameras else (available_cameras[0] if available_cameras else None)
if cam_index is None: st.error("❌ No camera found. Please connect a USB camera or enable your webcam.")

# =========================
# CAMERA CONTROL
col1, col2 = st.columns(2)
with col1: start_btn = st.button("Start Camera")
with col2: stop_btn = st.button("Stop Camera")
if start_btn: st.session_state.camera_running = True
if stop_btn: st.session_state.camera_running = False

# =========================
# PLACEHOLDERS
camera_placeholder = st.image([], width=480)
bar_placeholder = st.empty()
pie_placeholder = st.empty()

bar_colors = {'Ripe':'#2ecc71', 'Unripe':'#f1c40f', 'Overripe':'#e74c3c'}

# =========================
# LIVE CAMERA LOOP
if st.session_state.camera_running and cam_index is not None:
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret: break

        results = yolo_model.predict(frame, device=device)
        annotated_frame = frame.copy()

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if results[0].names[cls_id].lower() == "apple":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                apple_crop = frame[y1:y2, x1:x2]
                ripeness = predict_ripeness(apple_crop)
                if is_new_box((x1,y1,x2,y2), st.session_state.cumulative_boxes):
                    st.session_state.cumulative_counts[ripeness] += 1
                    st.session_state.cumulative_boxes.append((x1,y1,x2,y2))
                color = (0,255,0) if ripeness=="Ripe" else (0,165,255) if ripeness=="Unripe" else (0,0,255)
                cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),color,2)
                (w,h), _ = cv2.getTextSize(ripeness, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(annotated_frame, ripeness, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        camera_placeholder.image(annotated_frame, channels="BGR")

        # -------------------------
        # Update Bar Chart
        bar_fig, bar_ax = plt.subplots(figsize=(3,1.2), dpi=80, facecolor='none')
        bar_ax.bar(st.session_state.cumulative_counts.keys(), st.session_state.cumulative_counts.values(),
                   color=[bar_colors[k] for k in st.session_state.cumulative_counts.keys()])
        bar_ax.set_facecolor('none')
        bar_ax.tick_params(axis='x', colors='white')
        bar_ax.tick_params(axis='y', colors='white')
        bar_ax.spines['bottom'].set_color('white')
        bar_ax.spines['left'].set_color('white')
        bar_ax.spines['top'].set_color('none')
        bar_ax.spines['right'].set_color('none')
        plt.tight_layout()
        bar_placeholder.pyplot(bar_fig, bbox_inches='tight', use_container_width=True)

        # -------------------------
        # Update Pie Chart
        total_counts = st.session_state.cumulative_counts
        if sum(total_counts.values()) > 0:
            pie_fig, pie_ax = plt.subplots(figsize=(2,2), dpi=80, facecolor='none')
            pie_ax.pie(total_counts.values(), labels=[k.lower() for k in total_counts.keys()],
                       autopct='%1.0f%%', colors=[bar_colors[k] for k in total_counts.keys()],
                       startangle=90, textprops={'fontsize':8, 'color':'white'}, radius=0.7)
            pie_ax.axis('equal')
            plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
            pie_placeholder.pyplot(pie_fig, bbox_inches='tight', use_container_width=False)

    cap.release()
