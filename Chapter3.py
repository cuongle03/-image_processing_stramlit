import streamlit as st
import cv2
import numpy as np
from PIL import Image


L = 256
#lam ama image
def Negative(imgin):
    #M la do cao, N la do rong cua anh, anh là ma trận M hàng, N cột
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8) + np.uint8(255)
    #quét ảnh
    for x in range (0,M):
        for y in range(0,N):
            r= imgin[x,y]
            s= L-1-r
            imgout[x,y] = np.uint8(s)
    return imgout


def NegativeColor(imgin):
    #C la channel: la 3 cho anh mau
    M, N, C = imgin.shape
    imgout = np.zeros((M,N,C), np.uint8) + np.uint8(255)
    for x in range (0,M):
        for y in range(0,N):
            # ảnh màu của opencv là BGR
            # ảnh màu của pillow là RBG
            # Pillow là thư viện ảnh của python

            b = imgin[x,y,0]
            b = L-1-b

            g = imgin[x,y,1]
            g = L-1-g

            r = imgin[x,y,2]
            r = L-1-r
            imgout[x,y,0]= np.uint8(b)
            imgout[x,y,1]= np.uint8(g)
            imgout[x,y,2]= np.uint8(r)
    return imgout


def Logarit(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    c = (L-1.0)/(np.log(L*1.0))
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            if r==0:
                r=1
            s = c*np.log(1.0 + r)
            imgout[x,y] = np.uint8(s)
    return imgout
 
def Power(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    gamma = 5.0
    c = np.power(L-1.0, 1- gamma)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            if r==0:
                r=1
            s= c * np.power(1.0*r, gamma)
            imgout[x,y] = np.uint8(s)
    return imgout

def PicewiseLine(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)

    rmin, rmax, _, _, = cv2.minMaxLoc(imgin)
    r1 = rmin
    s1 =0
    r2 =rmax
    s2= L-1

    for x in range(0,M):
        for y in range(0,N):
            r = imgin[x,y]
            #đoạn I
            if r <r1:
                s=((1.0*s1)/(r1))*r
            #đoạn II
            elif r<r2:
                s = 1.0*((s2-s1)/(r2-r1)*(r-r1)) +s1
            #đoạn III
            else:
                s= 1.0*((L-1-s2)/(L-1-r2)*(r-r1))+s1
            imgout[x,y] = np.uint8(s) 
    return imgout

def Histogram(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,L,3), np.uint8) + np.uint8(255)
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0,N):
            r = imgin[x,y]
            h[r] = h[r]+1
    p = 1.0*(h/(M*N))
    scale = 3000
    for r in range(0, L):
        cv2.line(imgout, (r,0), (r , M-1-np.int32(scale*p[r])), (255,0,0) )
    return imgout
        

def HistEqual(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8) 
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0,N):
            r = imgin[x,y]
            h[r] = h[r]+1
    p = 1.0*(h/(M*N))  
    s= np.zeros(L, np.float64)
    for k in range(0,L):
        for j in range(0, k+1):
            s[k] = s[k] + p[j]         
    
    for x in range(0, M):
        for y in range(0,N):
            r = imgin[x,y]
            imgout[x,y]= np.uint8((L-1)*s[r])

    return imgout

def LocalHistogram(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8) 
    m =3
    n= 3
    a = m//2
    b= n//2
    for x in range(a, M-a):
        for y in range(b, N-1):
            w= imgin[x-a:x+a+1, y-b:y+b+1]
            w= cv2.equalizeHist(w)
            imgout[x,y]= w[a,b]
    return imgout

def HistogramStat(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8) 
    mean,stddev = cv2.meanStdDev(imgin)
    mG = mean[0,0]
    sigmaG= stddev[0,0]
    #chia anh thanh tung vung nho
    m =3
    n= 3
    a = m//2
    b= n//2

    C = 22.8
    k0 = 0
    k1 = 0.1
    k2 = 0
    k3 = 0.1
    for x in range(a, M-a):
        for y in range(b, N-1):
            w= imgin[x-a:x+a+1, y-b:y+b+1]
            mean,stddev = cv2.meanStdDev(w)
            msxy = mean[0,0]
            sigmasxy = stddev[0,0]
            if (k0*mG <= msxy <= k1*mG) and (k2*sigmaG <= sigmasxy <= k3*sigmaG):
                imgout[x,y] = np.uint8(C*imgin[x,y])
            else:
                imgout[0,0]= imgin[0,0]

    return imgout

def Sharp(imgin):
    w = np.array([[1,1,1],[1,-8,1],[1,1,1]], np.float32)
    Laplacian= cv2.filter2D(imgin, cv2.CV_32FC1, w)
    imgout = imgin - Laplacian
    imgout =  np.clip(imgout, 0, L-1) # cat bot
    imgout = imgout.astype(np.uint8) #ep kieu du lieu
    return imgout

def Gadient(imgin): # tách biên
    Sobel_x = np.array(([[1,2,1], [0,0,0], [-1,-2,-1]]), np.float32)
    Sobel_y = np.array(([[-1,0,1], [-2,0,2], [-1,0,1]]), np.float32)
    gx = cv2.filter2D(imgin, cv2.CV_32FC1, Sobel_x)
    gy = cv2.filter2D(imgin, cv2.CV_32FC1, Sobel_y)
    imgout = abs(gx) + abs(gy)

    imgout =  np.clip(imgout, 0, L-1) # cat bot
    imgout = imgout.astype(np.uint8) #ep kieu du lieu

    return imgout

def SmoothBox(imgin):
    return cv2.blur(imgin, (5, 5))  # hoặc (21, 21)

def SmoothGaussian(imgin):
    return cv2.GaussianBlur(imgin, (43, 43), 7)

def MedianFilter(imgin):
    return cv2.medianBlur(imgin, 5)

# ---Hiểm thị
def show():
    # Dictionary ánh xạ tên chức năng → hàm xử lý tương ứng và chế độ ảnh ("L" hay "RGB")
    func_map = {
        "Negative":      (Negative, "L"),
        "NegativeColor": (NegativeColor, "RGB"),
        "Logarit":       (Logarit, "L"),
        "Gamma":         (Power, "L"),
        "PicewiseLine":  (PicewiseLine, "L"),
        "Histogram":     (Histogram, "L"),
        "Hist Equal":    (HistEqual, "L"),
        "Local Histogram": (LocalHistogram, "L"),
        "Histogram Static": (HistogramStat, "L"),
        "Smooth Box":    (SmoothBox, "L"), 
        "Smooth Gaussian": (SmoothGaussian, "L"), 
        "median filter": (MedianFilter, "L"),
        "Sharp":         (Sharp, "L"),
        "Gradient":      (Gadient, "L")
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
