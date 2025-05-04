import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import subprocess
import torch
import requests  # DeepL API 사용을 위해 추가

# ====== 타이틀 및 설명 ======
st.markdown("<h1 style='line-height:1.2;margin-bottom:0;'>SignSync</h1><h3 style='margin-top:0;'>: 영어 수어(ASL) 동영상 업로드, VideoMAE 기반 번역 및 자막 생성 데모</h3>", unsafe_allow_html=True)
st.markdown("ASL(영어 수어) 동영상 파일을 업로드하면 프레임을 추출하고, VideoMAE 기반 모델로 번역해 자막을 생성하고 영상에 입힙니다.")
            
DEEPL_API_KEY = "112a1dc2-2634-49ef-a189-eb674714781b:fx"  # 여기에 본인의 DeepL API 키를 입력하세요

def translate_to_korean(text):
    try:
        url = "https://api-free.deepl.com/v2/translate"
        params = {
            "auth_key": DEEPL_API_KEY,
            "text": text,
            "source_lang": "EN",
            "target_lang": "KO"
        }
        response = requests.post(url, data=params)
        result = response.json()
        return result["translations"][0]["text"]
    except Exception as e:
        st.error(f"DeepL 번역 오류: {e}")
        return ""

def extract_frames(video_path, max_frames=16):
    import cv2
    from PIL import Image
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > max_frames:
        idxs = set([int(i * total_frames / max_frames) for i in range(max_frames)])
    else:
        idxs = set(range(total_frames))
    idx = 0
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in idxs:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            frame_count += 1
        idx += 1
    cap.release()
    return frames

@st.cache_resource
def load_videomae_model():
    try:
        from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
        processor = VideoMAEImageProcessor.from_pretrained("ihsanahakiim/videomae-base-finetuned-signlanguage-last-3")
        model = VideoMAEForVideoClassification.from_pretrained("ihsanahakiim/videomae-base-finetuned-signlanguage-last-3")
        return processor, model
    except Exception as e:
        st.error(f"VideoMAE 모델 로딩 오류: {e}")
        return None, None

def videomae_translate(frames, processor, model):
    try:
        inputs = processor(frames, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # 라벨 리스트에서 숫자와 탭/공백을 제외하고 단어만 추출
            with open("class_list.txt") as f:
                idx2word = []
                for line in f:
                    word = line.strip()
                    if "\t" in word:
                        _, word = word.split("\t", 1)
                    elif " " in word:
                        _, word = word.split(" ", 1)
                    if word and not word.isdigit():
                        idx2word.append(word)
            pred_idx = logits.argmax(-1).item()
            label = idx2word[pred_idx]
        return label
    except Exception as e:
        st.error(f"VideoMAE 번역 오류: {e}")
        return "..."

def generate_srt(text, fps=30, duration_sec=5):
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    start = 0
    end = duration_sec
    srt = f"1\n{format_time(start)} --> {format_time(end)}\n{text}\n"
    return srt

# 파일 업로드 위젯 선언
uploaded_file = st.file_uploader("ASL MP4 파일을 업로드하세요.", type=['mp4'])

if uploaded_file is not None:
    temp_file_path = os.path.join("/tmp", uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("파일 업로드 완료")

    with st.spinner("프레임 추출 중..."):
        frames = extract_frames(temp_file_path, max_frames=16)
    st.success(f"총 {len(frames)}개 프레임 추출 완료")

    with st.spinner("VideoMAE 수어 번역 모델 로딩 및 예측 중..."):
        processor, model = load_videomae_model()
        if processor is not None and model is not None:
            translated_text = videomae_translate(frames, processor, model)
        else:
            translated_text = "..."
    st.success("수어 번역(자막) 생성 완료")
    st.write("번역 결과(영어):", translated_text)

    # DeepL API로 한글 번역 추가
    with st.spinner("영어 자막을 한글로 번역 중..."):
        translated_text_ko = translate_to_korean(translated_text)
    st.success("Deepl API 번역 완료")
    st.write(f"DeepL API 번역 결과 : {translated_text_ko}")

    cap = cv2.VideoCapture(temp_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = int(total_frames / fps) if fps > 0 else 5
    cap.release()
    srt_content = generate_srt(translated_text, fps=int(fps), duration_sec=duration_sec)
    srt_path = temp_file_path + ".srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    with st.spinner("자막이 입혀진 영상 생성 중..."):
        srt_path = os.path.abspath(srt_path)
        output_video_path = temp_file_path.replace(".mp4", "_sub.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", temp_file_path,
            "-vf", f"subtitles={srt_path}:force_style='FontName=Arial,FontSize=24'",
            output_video_path
        ]
        subprocess.run(cmd)
    st.success("자막이 입혀진 영상 생성 완료")

    if os.path.exists(output_video_path):
        st.markdown("### 자막이 입혀진 영상 미리보기 및 다운로드")
        st.video(output_video_path)
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
        with col4:
            st.download_button("자막(SRT) 다운로드", srt_content, file_name="output.srt")
        with col5:
            with open(output_video_path, "rb") as f:
                st.download_button("자막 입힌 영상 다운로드", f, file_name="output_with_sub.mp4")

st.markdown("---")
st.markdown("#### 참고: VideoMAE 수어 번역 모델")
st.markdown("- [ihsanahakiim/videomae-base-finetuned-signlanguage-last-3 (Hugging Face)](https://huggingface.co/ihsanahakiim/videomae-base-finetuned-signlanguage-last-3)")