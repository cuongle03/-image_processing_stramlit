import streamlit as st
import cv2
import numpy as np
from PIL import Image

L= 256
def Spectrum(imgin):
    M,N = imgin.shape
    #buoc 1 and 2: tạo ảnh có kích thước PxQ
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = 1.0*imgin/(L-1)

    #bước 3 nhân fp với (-1)^x+y
    for x in range (0,M):
        for y in range(0,N):
            if (x+y) %2 ==1:
                fp[x,y] = -fp[x,y]

    #bước 4 tính DFT
    F = cv2.dft(fp, flags= cv2.DFT_COMPLEX_OUTPUT)
    FR = F[:,:,0]
    FI = F[:,:,1]
    #tính phổ
    S = np.sqrt(FR**2+FI**2)
    S = np.clip(S, 0, L-1)
    imgout = S.astype(np.uint8)
    return imgout


def CreateNotFilter(P, Q):
    H = np.ones((P,Q,2), np.float32)
    H[:,:,1] = 0.0

    u1= 45; v1 =58
    u2= 86; v2 =58
    u3= 41; v3 =119
    u4= 83; v4 =119

    u5 = P-u1; v5 = Q-v1
    u6 = P-u2; v6 = Q-v2
    u7 = P-u3; v7 = Q-v3
    u8 = P-u4; v8 = Q-v4
    D0=15
    for u in range(0,P):
        for v in range(0,Q):
            #u1,v1
            Duv = np.sqrt((1.0*u-u1)**2+ (1.0*v-v1)**2)
            if Duv<=D0:
                H[u,v,0]= 0.0

            #u2,v2
            Duv = np.sqrt((1.0*u-u2)**2+ (1.0*v-v2)**2)
            if Duv<=D0:
                H[u,v,0]= 0.0

            #u3,v3
            Duv = np.sqrt((1.0*u-u3)**2+ (1.0*v-v3)**2)
            if Duv<=D0:
                H[u,v,0]= 0.0

            #u4,v4
            Duv = np.sqrt((1.0*u-u4)**2+ (1.0*v-v4)**2)
            if Duv<=D0:
                H[u,v,0]= 0.0

            #u5,v5
            Duv = np.sqrt((1.0*u-u5)**2+ (1.0*v-v5)**2)
            if Duv<=D0:
                H[u,v,0]= 0.0

            #u6,v6
            Duv = np.sqrt((1.0*u-u6)**2+ (1.0*v-v6)**2)
            if Duv<=D0:
                H[u,v,0]= 0.0

            #u7,v7
            Duv = np.sqrt((1.0*u-u7)**2+ (1.0*v-v7)**2)
            if Duv<=D0:
                H[u,v,0]= 0.0

            #u8,v8
            Duv = np.sqrt((1.0*u-u8)**2+ (1.0*v-v8)**2)
            if Duv<=D0:
                H[u,v,0]= 0.0
    return H


def CreateNotPeriodFilter(P, Q):
    H = np.ones((P,Q,2), np.float32)
    H[:,:,1] = 0.0
    D0 = 10
    v0 = Q//2
    for u in range (0,P):
        for v in range(0,Q):
            if u not in range (P//2-30, P// 2+31+1):
                if abs(v - v0) <=D0:
                    H[u,v,0] = 0.0
    return H 



def DrawNotchFilter(imgin):
    M,N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H= CreateNotFilter(P,Q)

    HR = H[:,:,0]*(L-1)
    imgout = HR.astype(np.uint8)

    return imgout



def DrawNotchPeriodFilter(imgin):
    M,N = imgin.shape

    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H= CreateNotPeriodFilter(P,Q)
    HR = H[:,:,0]*(L-1)
    imgout = HR.astype(np.uint8)
    return imgout

def RemoveMoireSmiple(imgin):
    M,N = imgin.shape
    #buoc 1 and 2: tạo ảnh có kích thước PxQ
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = 1.0*imgin

    #bước 3 nhân fp với (-1)^x+y
    for x in range (0,M):
        for y in range(0,N):
            if (x+y) %2 ==1:
                fp[x,y] = -fp[x,y]

    #bước 4 tính DFT
    F = cv2.dft(fp, flags= cv2.DFT_COMPLEX_OUTPUT)
    
    #bước 5
    H = CreateNotFilter(P,Q)

    #bước 6 nhân G(u,v) =  F(u,v)* H(u,v)
    #Lọc trong miền tần số là bước 6
    G = cv2.mulSpectrums(F,H,flags=cv2.DFT_ROWS)

    # bước 7: IDFT
    g = cv2.idft(G, flags=cv2.DFT_SCALE)
    
    #bước 8 lấy lại kích thước ảnh  ban đầu MxN, lấy phần thực, nhân (-1)^(x+y)
    gR = g[:M,:N,0]
    for i in range(0,M):
        for i in range (0,N):
            if (x+y)%2 ==1:
                gR[x,y] = -g[x,y]
    gR = np.clip(gR, 0, L-1)
    imgout = gR.astype(np.uint8)
     


    return imgout

def RemovePeriodNoise(imgin):
    M,N = imgin.shape
    #buoc 1 and 2: tạo ảnh có kích thước PxQ
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = 1.0*imgin

    #bước 3 nhân fp với (-1)^x+y
    for x in range (0,M):
        for y in range(0,N):
            if (x+y) %2 ==1:
                fp[x,y] = -fp[x,y]

    #bước 4 tính DFT
    F = cv2.dft(fp, flags= cv2.DFT_COMPLEX_OUTPUT)
    
    #bước 5
    H = CreateNotPeriodFilter(P,Q)

    #bước 6 nhân G(u,v) =  F(u,v)* H(u,v)
    #Lọc trong miền tần số là bước 6
    G = cv2.mulSpectrums(F,H,flags=cv2.DFT_ROWS)

    # bước 7: IDFT
    g = cv2.idft(G, flags=cv2.DFT_SCALE)
    
    #bước 8 lấy lại kích thước ảnh  ban đầu MxN, lấy phần thực, nhân (-1)^(x+y)
    gR = g[:M,:N,0]
    for i in range(0,M):
        for i in range (0,N):
            if (x+y)%2 ==1:
                gR[x,y] = -g[x,y]
    gR = np.clip(gR, 0, L-1)
    imgout = gR.astype(np.uint8)
    return imgout
def show():
# Dictionary ánh xạ tên chức năng → hàm xử lý tương ứng và chế độ ảnh ("L" hay "RGB")
    func_map = {
    "Spectrum":      (Spectrum, "L"),
    "Draw Not Filter":(DrawNotchFilter,"L"),
    "Remove Moire Simple":(RemoveMoireSmiple,"L"),
    "Draw Period Noise Filter":(DrawNotchPeriodFilter,"L"),
    " Remove Period Noise":(RemovePeriodNoise,"L")
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