import streamlit as st
import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

st.markdown("<h1 style='line-height:1.2;margin-bottom:0;'>SignSync</h1><h3 style='margin-top:0;'>: 동영상 업로드, 손·얼굴 표정 인식 프레임 추출 데모</h3>", unsafe_allow_html=True)
st.markdown("MP4 동영상 파일을 업로드하면 프레임을 추출하고 MediaPipe로 손과 얼굴 표정을 인식한 결과 프레임을 보여줍니다.")

uploaded_file = st.file_uploader("MP4 파일을 업로드하세요.", type=['mp4'])

def extract_and_annotate_frames(video_path, max_frames=16):
    mp_holistic = mp.solutions.holistic
    drawing = mp.solutions.drawing_utils
    frames = []
    total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > max_frames:
        idxs = set([int(i * total_frames / max_frames) for i in range(max_frames)])
    else:
        idxs = set(range(total_frames))
    idx = 0
    frame_count = 0
    cap = cv2.VideoCapture(video_path)
    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5) as holistic:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in idxs:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)
                annotated = rgb.copy()
                # 손, 얼굴, 포즈 랜드마크 모두 그림
                if result.face_landmarks:
                    drawing.draw_landmarks(
                        annotated, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                if result.left_hand_landmarks:
                    drawing.draw_landmarks(
                        annotated, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if result.right_hand_landmarks:
                    drawing.draw_landmarks(
                        annotated, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                frames.append(Image.fromarray(annotated))
                frame_count += 1
            idx += 1
    cap.release()
    return frames

if uploaded_file is not None:
    temp_file_path = os.path.join("/tmp", uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("파일 업로드 완료")

    with st.spinner("프레임 추출 및 손·얼굴 표정 인식 중..."):
        annotated_frames = extract_and_annotate_frames(temp_file_path, max_frames=8)
    st.success(f"총 {len(annotated_frames)}개 프레임에서 손·얼굴 표정 인식 완료")

    st.markdown("### 손·얼굴 표정 인식 결과 프레임")
    # 4개씩 한 줄에 출력
    cols = st.columns(4)
    for i, img in enumerate(annotated_frames):
        with cols[i % 4]:
            st.image(img, caption=f"Frame {i+1}", use_container_width=True)