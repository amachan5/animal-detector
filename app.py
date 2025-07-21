import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io

# モデルをロード
model = YOLO("yolov8n.pt")  # 小型のYOLOv8モデル（初回だけ自動DL）

st.title("🐾 物体検知アプリ")
st.write("画像をアップロードすると、物体を検出します。")

uploaded_file = st.file_uploader("画像を選んでください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 入力画像を読み込み
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="元画像", use_container_width=True)

    # 物体検出
    results = model.predict(image_np)
    result_image = results[0].plot()

    # 結果画像表示
    st.image(result_image, caption="検出結果", use_container_width=True)

    # --- 🔽 ダウンロードボタンを検出画像の下に配置 ---
    result_pil = Image.fromarray(result_image)
    img_buffer = io.BytesIO()
    result_pil.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    st.download_button(
        label="📥 この画像をダウンロード",
        data=img_buffer,
        file_name="detection_result.png",
        mime="image/png"
    )

    # 検出されたオブジェクト表示
    st.subheader("検出されたオブジェクト")
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        label = model.names[cls_id]
        st.write(f"- {label}（信頼度: {conf:.2f}）")
