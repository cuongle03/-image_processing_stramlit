import streamlit as st
import cv2
import numpy as np
from PIL import Image

L= 256
def FrequencyFiltering(imgin, H):
    M,N = imgin.shape
    f = imgin.astype(np.float64)

    # Bước 1: DFT
    F = np.fft.fft2(f)

    # Bước 2: Shift vào giữa
    F = np.fft.fftshift(F)

    # Bước 3: Nhân F với H
    G = F * H

    # Bước 4: Shift ngược lại
    G = np.fft.ifftshift(G)  # << Đúng là ifftshift, không phải fftshift

    # Bước 5: IDFT
    g = np.fft.ifft2(G)

    # Lấy phần thực, chuẩn hóa về uint8
    gR = np.real(g)
    gR = np.clip(gR, 0, L - 1)
    imgout = gR.astype(np.uint8)
    return imgout

def CreateMotionFilter(M,N):
    H= np.zeros((M,N), np.complex64)
    a= 0.1
    b=0.1
    T = 1.0
    phi_prev =0.0
    for u in range(0,M):
        for v in range(0,N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if abs(phi) < 1.0e-6:
                phi = phi_prev
            RE = T*np.sin(phi)/phi*np.cos(phi)
            IM = -T*np.sin(phi)/phi*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def CreateMotion(imgin) :
    M,N = imgin.shape
    H = CreateMotionFilter(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout     

def CreateMotionInverseFilter(M,N):
    H = np.zeros((M,N), np.complex64)
    T =1
    a= 0.1
    b=0.1
    phi_prev =0.0
    for u in range(0,M):
        for v in range(0,N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            mau_so= np.sin(phi)
            if abs(mau_so)< 1.0e-6:
                phi= phi_prev
            RE = phi/(T*np.sin(phi))* np.cos(phi)
            IM = phi/T
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def DeMotion(imgin) :
    M,N = imgin.shape
    H = CreateMotionInverseFilter(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout  
def DeMotionNoise(imgin) :
    imgout= cv2.medianBlur(imgin,5)
    return imgout
def show():
# Dictionary ánh xạ tên chức năng → hàm xử lý tương ứng và chế độ ảnh ("L" hay "RGB")
    func_map = {
    "Create Motion":(CreateMotion,"L"),
    "DeMotion":(DeMotion,"L"),
    "DeMotionNoise":(DeMotionNoise,"L")
    }
    uploaded_file = st.file_uploader("Chọn ảnh đầu vào", type=["jpg", "jpeg", "png", "tif"])
    if uploaded_file is None:
        st.warning("Vui lòng chọn một ảnh.")
        return

    current_func = st.session_state.current_page.split("_")[1]

    if current_func in func_map:
        process_func, mode = func_map[current_func]
        
        # Đọc ảnh gốc và chuyển sang chế độ cần thiết
        imgin = Image.open(uploaded_file).convert(mode)
        imgin = np.array(imgin)

        # Xử lý ảnh và lưu ảnh kết quả
        img_result = process_func(imgin)

        # Tạo 2 cột để hiển thị ảnh gốc và ảnh đã xử lý
        col1, col2 = st.columns(2)

        # Cột 1 - Ảnh gốc
        with col1:
            st.image(imgin, caption="Ảnh gốc", use_column_width=True)

        # Cột 2 - Ảnh đã xử lý
        with col2:
            st.image(img_result, caption="Ảnh sau xử lý", use_column_width=True)

    else:
        st.warning("Chức năng chưa được cài đặt.")