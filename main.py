import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO
import re
from datetime import datetime
import os


# Enhanced detection configuration
model = YOLO("my_yolo_model.pt")  # Your trained model path
PREDICTION_CONF = 0.15  # Lower confidence threshold
PREDICTION_IOU = 0.3  # Lower IoU for fewer merges
PREDICTION_IMGSZ = 1280  # Higher resolution
MAX_DETECTIONS = 100  # Maximum detections per frame

# Session state initialization
if 'sessions' not in st.session_state:
    st.session_state.sessions = []

vegetable_prices = {
    "onion": 40, "potato": 30, "capcicum": 80, "tomato": 25,
    "carrot": 35, "broccoli": 60, "spinach": 20, "ginger": 100,
    "garlic": 150, "cucumber": 15, "eggplant": 45, "cauliflower": 55
}


def extract_weight(label):
    # Enhanced regex for various label formats
    match = re.match(r'([a-zA-Z]+)[_\s-]?(\d+)gms?', label, re.IGNORECASE)
    return match.groups() if match else ("unknown", 0)


def create_session(mode):
    return {
        "id": len(st.session_state.sessions) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "detections": {},
        "total_weight": 0,
        "frames_processed": 0
    }


def update_session(session, results):
    for result in results:
        for obj in result.boxes:
            label = model.names[int(obj.cls)]
            veg, weight = extract_weight(label)
            veg = veg.lower()
            session["detections"][veg] = session["detections"].get(veg, 0) + int(weight)
    session["total_weight"] = sum(session["detections"].values())
    session["frames_processed"] += 1


def process_frame(frame, session):
    results = model.predict(
        frame,
        conf=PREDICTION_CONF,
        iou=PREDICTION_IOU,
        imgsz=PREDICTION_IMGSZ,
        max_det=MAX_DETECTIONS,
        augment=True
    )
    update_session(session, results)
    return results[0].plot()


def dashboard():
    st.header("📊 Detection Sessions Dashboard")

    if st.button("🧹 Clear All Sessions"):
        st.session_state.sessions = []

    if not st.session_state.sessions:
        st.info("No detection sessions recorded yet")
        return

    for idx, session in enumerate(reversed(st.session_state.sessions)):
        with st.expander(f"Session #{session['id']} - {session['mode']} - {session['timestamp']}"):
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.metric("Total Weight", f"{session['total_weight']}g")
                st.write(f"**Mode:** {session['mode']}")
                st.write(f"**Frames:** {session['frames_processed']}")

            with col2:
                if session['detections']:
                    st.write("**Detected Items:**")
                    for veg, weight in session['detections'].items():
                        price = vegetable_prices.get(veg, 0) * weight / 1000
                        st.write(f"- {veg.capitalize()}: {weight}g (₹{price:.2f})")
                else:
                    st.warning("No vegetables detected in this session")

            # Add this print button section
            with col3:
                report_text = f"Session Report\n{'=' * 30}\n"
                report_text += f"Session ID: {session['id']}\n"
                report_text += f"Timestamp: {session['timestamp']}\n"
                report_text += f"Detection Mode: {session['mode']}\n"
                report_text += f"Total Weight: {session['total_weight']}g\n\n"
                report_text += "Detected Items:\n"

                for veg, weight in session['detections'].items():
                    price = vegetable_prices.get(veg, 0) * weight / 1000
                    report_text += f"{veg.capitalize()}: {weight}g (₹{price:.2f})\n"

                st.download_button(
                    label="📄 Print Report",
                    data=report_text,
                    file_name=f"session_{session['id']}_report.txt",
                    mime="text/plain",
                    key=f"print_{session['id']}"
                )


def main():
    st.set_page_config(page_title="Vegetable Analyst", layout="wide")
    st.title("🥦 Multi-Session Vegetable Detection System 🥕")

    mode = st.sidebar.radio("Select Mode:", [
        "Image Upload", "Real-Time", "Video Upload", "Photo Capture", "Dashboard"
    ])

    if mode == "Dashboard":
        dashboard()
        return

    if mode == "Image Upload":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            session = create_session("Image Upload")
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            processed_frame = process_frame(frame, session)
            st.session_state.sessions.append(session)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image")
            with col2:
                st.image(processed_frame[:, :, ::-1], caption="Detected Vegetables")

    elif mode == "Real-Time":
        st.header("Live Camera Detection")
        run = st.checkbox("Start Detection")

        session = create_session("Real-Time")
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while run and cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.error("Camera Error")
                break

            processed_frame = process_frame(frame, session)
            frame_placeholder.image(processed_frame[:, :, ::-1], channels="BGR")

        if session["frames_processed"] > 0:
            st.session_state.sessions.append(session)
        cap.release()

    elif mode == "Video Upload":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            session = create_session("Video Upload")
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            frame_placeholder = st.empty()

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                processed_frame = process_frame(frame, session)
                frame_placeholder.image(processed_frame[:, :, ::-1], channels="BGR")

            if session["frames_processed"] > 0:
                st.session_state.sessions.append(session)
            cap.release()

    elif mode == "Photo Capture":
        if st.button("📸 Capture Photo"):
            session = create_session("Photo Capture")
            cap = cv2.VideoCapture(0)
            success, frame = cap.read()
            cap.release()

            if success:
                processed_frame = process_frame(frame, session)
                st.session_state.sessions.append(session)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(frame[:, :, ::-1], caption="Captured Photo")
                with col2:
                    st.image(processed_frame[:, :, ::-1], caption="Detected Vegetables")


if __name__ == "__main__":
    main()