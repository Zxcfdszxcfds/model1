import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="图像颜色空间与插值平台", layout="wide")
st.title("🎨 图像颜色空间与插值实验（A1作业）")

# ---------------------- 通用上传 ----------------------
uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "png"], key="main_up")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    st.image(img_np, caption="原图", use_column_width=True)

    # ---------------------- 1. 颜色空间转换（RGB/HSV） ----------------------
    st.header("1. 颜色空间通道可视化")
    if st.button("显示RGB/HSV通道", key="color_btn"):
        with st.spinner("转换中..."):
            # RGB通道
            r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
            # HSV通道
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes[0,0].imshow(r, cmap="gray")
            axes[0,0].set_title("R通道")
            axes[0,0].axis("off")
            axes[0,1].imshow(g, cmap="gray")
            axes[0,1].set_title("G通道")
            axes[0,1].axis("off")
            axes[0,2].imshow(b, cmap="gray")
            axes[0,2].set_title("B通道")
            axes[0,2].axis("off")
            axes[1,0].imshow(h, cmap="hsv")
            axes[1,0].set_title("H通道")
            axes[1,0].axis("off")
            axes[1,1].imshow(s, cmap="gray")
            axes[1,1].set_title("S通道")
            axes[1,1].axis("off")
            axes[1,2].imshow(v, cmap="gray")
            axes[1,2].set_title("V通道")
            axes[1,2].axis("off")
            st.pyplot(fig)

    # ---------------------- 2. 图像插值算法 ----------------------
    st.header("2. 图像插值算法演示")
    scale = st.slider("缩放比例", 0.25, 4.0, 2.0, key="scale_slide")
    inter_method = st.selectbox("插值方法", ["最近邻", "双线性"], key="inter_select")

    if st.button("执行插值", key="inter_btn"):
        with st.spinner("插值中..."):
            inter_flag = cv2.INTER_NEAREST if inter_method == "最近邻" else cv2.INTER_LINEAR
            new_h, new_w = int(img_np.shape[0]*scale), int(img_np.shape[1]*scale)
            resized = cv2.resize(img_np, (new_w, new_h), interpolation=inter_flag)

            fig, axes = plt.subplots(1,2, figsize=(12,5))
            axes[0].imshow(img_np)
            axes[0].set_title("原图")
            axes[0].axis("off")
            axes[1].imshow(resized)
            axes[1].set_title(f"{inter_method}插值结果（缩放比例{scale}）")
            axes[1].axis("off")
            st.pyplot(fig)

    # ---------------------- 3. 图像旋转/拉伸变换 ----------------------
    st.header("3. 图像旋转与拉伸")
    angle = st.slider("旋转角度", 0, 360, 45, key="angle_slide")
    stretch = st.slider("水平拉伸比例", 0.5, 2.0, 1.0, key="stretch_slide")

    if st.button("执行变换", key="transform_btn"):
        with st.spinner("变换中..."):
            # 旋转
            rows, cols = img_np.shape[:2]
            center = (cols//2, rows//2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
            rotated = cv2.warpAffine(img_np, rot_mat, (cols, rows))

            # 拉伸
            stretched = cv2.resize(rotated, (int(cols*stretch), rows))

            fig, axes = plt.subplots(1,3, figsize=(15,5))
            axes[0].imshow(img_np)
            axes[0].set_title("原图")
            axes[0].axis("off")
            axes[1].imshow(rotated)
            axes[1].set_title(f"旋转{angle}°结果")
            axes[1].axis("off")
            axes[2].imshow(stretched)
            axes[2].set_title(f"旋转+水平拉伸结果")
            axes[2].axis("off")
            st.pyplot(fig)

st.markdown("---")
st.caption("模式识别与图像处理 - A1作业平台 | 基于Python+OpenCV")
