import streamlit as st
import os

st.title("SignSync: 영어 수어(ASL) 영상 자막 생성기 (Beta)")
st.markdown("1분 이내의 MP4 영어 수어 영상을 업로드하면 자막을 생성해 줍니다.")

uploaded_file = st.file_uploader("MP4 파일을 업로드하세요.", type=['mp4'])

if uploaded_file is not None:
    temp_file_path = os.path.join("/tmp", uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"파일 '{uploaded_file.name}' 업로드 완료!")