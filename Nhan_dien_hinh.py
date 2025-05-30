import cv2
import numpy as np
import streamlit as st
from PIL import Image

def phan_nguong(imgin):
    imgin = cv2.imread(imgin, cv2.IMREAD_GRAYSCALE)
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r == 63:
                s = 255
            else:
                s = 0
            imgout[x, y] = np.uint8(s)
    imgout = cv2.medianBlur(imgout, 7)
    return imgout

def mnu_shape_predict(imgin):
    temp = phan_nguong(imgin)
    contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imgout = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)  # chuyển sang ảnh màu để ghi chữ

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        m = cv2.moments(contour)
        Hu = cv2.HuMoments(m)

        if 0.1 <= Hu[0, 0] <= 0.159183:
            s = 'HinhTron'
        elif 0.1600 <= Hu[0, 0] <= 0.167:
            s = 'HinhVuong'
        elif 0.169 <= Hu[0, 0] <= 0.19:
            s = 'HinhTamGiac'
        else:
            s = 'Unknow'
        cv2.putText(imgout, s, (10,30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    return imgout

def show():
    # st.title("Phân loại hình học với Hu Moments")
    uploaded_file = st.file_uploader("Chọn ảnh đầu vào", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        img_path = "temp_input_image.bmp"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        st.image(Image.open(img_path), caption="Ảnh gốc", use_column_width=True)

        imgout = mnu_shape_predict(img_path)

        st.image(imgout, caption="Ảnh đã nhận dạng (ghi nhãn)", channels="BGR", use_column_width=True)

if __name__ == "__main__":
    show() 