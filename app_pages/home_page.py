import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def app():
    st.image('ocr.png', width=900)

    #t.write("")

    st.markdown('''#### OCR, or Optical Character Recognition, is a computer vision task, \
    which includes the detection of text areas, and the recognition of characters.''')
    st.write("")
    st.write("")

    st.markdown("#####  This app allows you to compare, from a given image, the results of different solutions:")
    st.markdown("###### *EasyOcr, PaddleOCR, MMOCR, Tesseract*")
    st.write("")
#    st.write("")

    st.markdown("#####  Before evaluate OCR solutions, you can check if image quality is good enough for \
                OCR task. If not, you'll try to enhance it before.")
    st.write("")
    st.write("")

    st.markdown("👈 Select the **About** page from the sidebar for information on how the app works")

    st.markdown("👈 or select the **Image processing** page to prepare and enhance your image")

    st.markdown("👈 or directly select the **OCR App** page")