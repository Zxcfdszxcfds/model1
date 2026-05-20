import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="图像颜色空间与插值（无OpenCV版）", layout="wide")
st.title("🎨 图像颜色空间与插值实验（A1作业）")

# ---------------------- 通用上传 ----------------------
uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "png"], key="main_up")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    h, w = np.shape(img_np)[:2]
    st.image(img_np, caption="原图", use_column_width=True)

    # ---------------------- 1. 颜色空间通道可视化（RGB/HSV） ----------------------
    st.header("1. 颜色空间通道可视化")
    if st.button("显示RGB/HSV通道", key="color_btn"):
        with st.spinner("转换中..."):
            # RGB通道
            r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
            
            # 手动实现RGB→HSV转换
            r_n, g_n, b_n = r/255.0, g/255.0, b/255.0
            cmax = np.maximum(np.maximum(r_n, g_n), b_n)
            cmin = np.minimum(np.minimum(r_n, g_n), b_n)
            delta = cmax - cmin
            h = np.zeros_like(cmax)
            s = np.where(cmax == 0, 0, delta / cmax)
            v = cmax
            
            # 计算H通道
            mask = delta != 0
            h[mask & (cmax == r_n)] = 60 * (((g_n - b_n)/delta)[mask & (cmax == r_n)] % 6)
            h[mask & (cmax == g_n)] = 60 * (((b_n - r_n)/delta)[mask & (cmax == g_n)] + 2)
            h[mask & (cmax == b_n)] = 60 * (((r_n - g_n)/delta)[mask & (cmax == b_n)] + 4)
            
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

    # ---------------------- 2. 图像插值算法（最近邻/双线性） ----------------------
    st.header("2. 图像插值算法演示")
    scale = st.slider("缩放比例", 0.25, 4.0, 2.0, key="scale_slide")
    inter_method = st.selectbox("插值方法", ["最近邻", "双线性"], key="inter_select")

    if st.button("执行插值", key="inter_btn"):
        with st.spinner("插值中..."):
            new_h, new_w = int(h * scale), int(w * scale)
            resized = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            
            if inter_method == "最近邻":
                # 最近邻插值
                for i in range(new_h):
                    for j in range(new_w):
                        x = int(i / scale)
                        y = int(j / scale)
                        resized[i,j] = img_np[x, y]
            else:
                # 双线性插值
                for i in range(new_h):
                    for j in range(new_w):
                        x = i / scale
                        y = j / scale
                        x0, y0 = int(x), int(y)
                        x1, y1 = min(x0+1, h-1), min(y0+1, w-1)
                        dx, dy = x - x0, y - y0
                        for c in range(3):
                            resized[i,j,c] = (1-dx)*(1-dy)*img_np[x0,y0,c] + dx*(1-dy)*img_np[x1,y0,c] + (1-dx)*dy*img_np[x0,y1,c] + dx*dy*img_np[x1,y1,c]
            
            fig, axes = plt.subplots(1,2, figsize=(12,5))
            axes[0].imshow(img_np)
            axes[0].set_title("原图")
            axes[0].axis("off")
            axes[1].imshow(resized)
            axes[1].set_title(f"{inter_method}插值结果（缩放比例{scale}）")
            axes[1].axis("off")
            st.pyplot(fig)

    # ---------------------- 3. 图像旋转与拉伸变换 ----------------------
    st.header("3. 图像旋转与拉伸")
    angle = st.slider("旋转角度", 0, 360, 45, key="angle_slide")
    stretch = st.slider("水平拉伸比例", 0.5, 2.0, 1.0, key="stretch_slide")

    if st.button("执行变换", key="transform_btn"):
        with st.spinner("变换中..."):
            # 旋转（简化实现）
            theta = np.radians(angle)
            cos, sin = np.cos(theta), np.sin(theta)
            new_h_rot = int(abs(h*cos) + abs(w*sin))
            new_w_rot = int(abs(h*sin) + abs(w*cos))
            rotated = np.zeros((new_h_rot, new_w_rot, 3), dtype=np.uint8)
            cx, cy = w/2, h/2
            for i in range(new_h_rot):
                for j in range(new_w_rot):
                    x = (j - new_w_rot/2)*cos + (i - new_h_rot/2)*sin + cx
                    y = -(j - new_w_rot/2)*sin + (i - new_h_rot/2)*cos + cy
                    if 0 <= x < w and 0 <= y < h:
                        rotated[i,j] = img_np[int(y), int(x)]
            
            # 拉伸
            new_w_stretch = int(new_w_rot * stretch)
            stretched = np.zeros((new_h_rot, new_w_stretch, 3), dtype=np.uint8)
            for i in range(new_h_rot):
                for j in range(new_w_stretch):
                    stretched[i,j] = rotated[i, int(j/stretch)]
            
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
st.caption("模式识别与图像处理 - A1作业平台 | 无OpenCV依赖")
