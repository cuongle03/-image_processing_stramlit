import streamlit as st
import cv2
import numpy as np
from PIL import Image

L= 256
def Erosion(imgin):
    w= cv2.getStructuringElement(cv2.MORPH_RECT, (45,45))
    imgout = cv2.erode(imgin,w)
    return imgout

def Dilation(imgin):
    w =  cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    imgout = cv2.dilate(imgin,w)
    return imgout

def Duality(imgin):
    w= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    temp= cv2.erode(imgin,w)
    imgout = imgin - temp
    return imgout

def Coutour(imgin):
    # note: dung cho anh nhi phan
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)

    coutours, _= cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coutour = coutours[0]
    n = len(coutour)
    #debug open Shape len xem 
    for i in range(0, n-1):
        x1 = coutour[i,0,0]
        y1= coutour[i,0,1]

        x2 = coutour[i+1,0,0]
        y2 = coutour[i+1,0,1]

        cv2.line(imgout,(x1,y1),(x2,y2),(0,0,255),2)
    x1 = coutour[n-1,0,0]
    y1= coutour[n-1,0,1]

    x2 = coutour[0,0,0]
    y2 = coutour[0,0,1]
    cv2.line(imgout,(x1,y1),(x2,y2),(0,0,255),2)

    return imgout

#dùng để bao cả phần bị khuyết 
def ConvexHull(imgin):
    # note: dung cho anh nhi phan
    # Tinh convexhull phai qua 2 buoc
    #b1: Tinh countour
    #b2: Tinh convex
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)

    coutours, _= cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coutour = coutours[0]
    hull= cv2.convexHull(coutour)
    n = len(hull)
    #debug open Shape len xem 
    for i in range(0, n-1):
        x1 = hull[i,0,0]
        y1= hull[i,0,1]

        x2 = hull[i+1,0,0]
        y2 = hull[i+1,0,1]

        cv2.line(imgout,(x1,y1),(x2,y2),(0,0,255),2)
    x1 = hull[n-1,0,0]
    y1= hull[n-1,0,1]

    x2 = hull[0,0,0]
    y2 = hull[0,0,1]
    cv2.line(imgout,(x1,y1),(x2,y2),(0,0,255),2)

    return imgout
# phá hiện khuyết = convexhull- coutour 
def DefecDetect(imgin):
    #qua 3 step, 1 tính coutour, 2 tính conver, 3 tính DefectDecte
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    #b1
    coutours, _= cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coutour = coutours[0]
    #b2, p là vị trí tren coutour [18,1]
    p = cv2.convexHull(coutour, returnPoints= False)
    n = len(p)
    for i in range(0,n-1):
        vi_tri1 = p[i,0]
        vi_tri2 = p[i+1,0]
        x1= coutour[vi_tri1,0,0]
        y1= coutour[vi_tri1,0,1]

        x2= coutour[vi_tri2,0,0]
        y2= coutour[vi_tri2,0,1]

        cv2.line(imgout, (x1,y1), (x2,y2), (0,0,255),2)
    vi_tri1 = p[n-1,0]
    vi_tri2 = p[0,0]
    x1= coutour[vi_tri1,0,0]
    y1= coutour[vi_tri1,0,1]

    x2= coutour[vi_tri2,0,0]
    y2= coutour[vi_tri2,0,1]

    cv2.line(imgout, (x1,y1), (x2,y2), (0,0,255),2)

    #b3 tính khuyết

    defects = cv2.convexityDefects(coutour, p)
    nguong_do_sau = np.max(defects[:,:,3])//2
    n = len (defects)
    for i in range (0,n):
        do_sau = defects[i,0,3]
        if do_sau>nguong_do_sau:
            vi_tri_khuyet = defects[i,0,2]
            x = coutour[vi_tri_khuyet,0,0]
            y = coutour[vi_tri_khuyet,0,1]
            cv2.circle(imgout,(x,y),5,(0,255,0),-1) 
    return imgout

def HoleFill(imgin):
    #ảnh màu của opencv là BGR
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    cv2.floodFill(imgout, None, (104,295),(0,0,255))
    return imgout

def ConnectedComponents(imgin):
    nguong = 200
    _,temp = cv2.threshold(imgin, nguong, L-1, cv2.THRESH_BINARY)
    imgout = cv2.medianBlur(temp, 7)
    dem, label = cv2.connectedComponents(imgout, None)
    a = np.zeros(dem, np.int32)
    M ,N = label.shape
    for x in range (0,M):
        for y in range(0,N):
            r = label[x,y]
            if r >0:
                a[r] = a[r]+1

    s = 'co %d thanh phan lien thong' % (dem-1)
    cv2.putText(imgout, s, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

    for r in range(1,dem):
        s ='%3d %5d' % (r, a[r])
        cv2.putText(imgout, s, (10, (r+1)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    return imgout

def RemoveSmallRice(imgin):
    # dùng biến đổi TOPHAT làm đậm bóng cho hạt gạo
    # 81 là kích thước lớn nhất của hạt gạo pixel
    w= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81,81))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    nguong = 120
    _, temp = cv2.threshold(temp, nguong,L-1, cv2.THRESH_BINARY| cv2.THRESH_OTSU )
    dem, label = cv2.connectedComponents(temp, None)
    a = np.zeros(dem, np.int32)
    M,N = label.shape
    for x in range(0,M):
        for y in range(0,N):
            r =label[x,y]
            if r>0:
                a[r] = a[r]+1
    max_value = np.max(a)
    imgout = np.zeros((M,N), np.uint8)
    for x in range(0,M):
        for y in range(0,N):
            r =label[x,y]
            if r>0:
                if a[r] > (0.7*max_value):
                    imgout[x,y] = L-1

    return imgout
def show():
    func_map = {
    "Erosion":(Erosion,"L"),
    "Dilation":(Dilation,"L"),
    "Duality":(Duality,"L"), 
    "Coutour":(Coutour,"L"),
    "ConvexHull":(ConvexHull,"L"),
    "Defect_Detect":(DefecDetect,"RGB"),
    "Hole_Fill":(HoleFill,"L"),
    "Connected_Components":(ConnectedComponents,"L"),
    "Remove_Small_Rice":(RemoveSmallRice,"L")
    
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