import streamlit as st

def app():
    st.image('ocr.png')

    st.write("")

    st.markdown('''#### OCR, or Optical Character Recognition, is a computer vision task, \
    which includes the detection of text areas, and the recognition of characters.''')
    st.write("")
    st.write("")

    st.markdown("#####  This app allows you to compare, from a given image, the results of different solutions:")
    st.markdown("##### *EasyOcr, PaddleOCR, MMOCR, Tesseract*")
    st.write("")
    st.write("")
    st.markdown("👈 Select the **About** page from the sidebar for information on how the app works")

    st.markdown("👈 or directly select the **App** page")