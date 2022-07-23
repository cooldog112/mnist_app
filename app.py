import cv2
import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model

model = load_model('./model/mnist_model.h5')

col1, col2 = st.columns([1,1])
CANVAS_SIZE = 192

with col1 :
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=15,
        stroke_color='#FFFFFF',
        background_color='#000000',
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode='freedraw',
        key="canvas",
    )

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)
    col2.image(rescaled)

    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = model.predict(np.reshape(test_x, (1, 28 * 28)))
    st.success(np.argmax(res[0]))
    st.bar_chart(res[0])
