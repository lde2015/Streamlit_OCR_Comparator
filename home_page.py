import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#def app():
st.markdown("""
    <style>
        .main > div:first-of-type {
            padding: 1em 2em 1em 2em;
        }
    </style>
""", unsafe_allow_html=True)

st.image('ocr.png', width=900)
st.write("")
st.write("")
lib = "This app allows you to compare, from a given image, the results of different solutions:  \n*EasyOcr, PaddleOCR, MMOCR, Tesseract*"
st.markdown('''##### :blue-background[:orange[OCR]]:orange[, or Optical Character Recognition,] ''')
st.markdown('''is a computer vision task, \
which includes the detection of text areas, and the recognition of characters.''')
st.markdown(lib)

st.markdown('''Before evaluate OCR solutions, you can check if image quality is good enough for \
OCR task. If not, you could try to enhance it before, with some processing operations.


👈 Select the **About** page from the sidebar for information on how the app works''')
st.markdown("👈 or select the **Image processing** page to prepare and enhance your image")
st.markdown("👈 or directly select the **OCR Comparator** page")
