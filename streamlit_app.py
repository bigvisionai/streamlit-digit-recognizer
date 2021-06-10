import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas
import numpy as np

st.title("OpenCV Webinar Series")
st.header("Digit Recognizer using OpenCV and Streamlit")

st.write("Draw a digit over the canvas")

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#fff",
    background_color="#000",
    update_streamlit=1,
    height=250,
    width=250,
    key="canvas",
)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


@st.cache(allow_output_mutation=True)
def load_model():
    net = cv2.dnn.readNetFromONNX('model.onnx')
    return net


def predict(net,canvas_result):

    img = cv2.cvtColor(np.uint8(canvas_result.image_data), cv2.COLOR_RGBA2GRAY)

    # Create a 4D blob from image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (28, 28))

    # Run a model
    net.setInput(blob)
    out = net.forward()

    # Get a class with a highest score
    out = softmax(out.flatten())
    classId = np.argmax(out)
    confidence = out[classId]

    st.success("Predicted Class: {}, Confidence: {:.2f}%".format(classId, confidence*100.))

net = load_model()

predict(net,canvas_result)

