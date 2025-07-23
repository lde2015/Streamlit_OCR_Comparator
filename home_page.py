import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 1rem;
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.image('ocr.png', width=900)
st.write("")
st.write("")
lib = "This app allows you to **compare**, from a given image, the results of different solutions:  \n*EasyOcr, PaddleOCR, MMOCR, Tesseract*"
st.markdown('''##### :blue-background[:orange[OCR]]:orange[, or Optical Character Recognition,] ''')
st.markdown('''is a computer vision task, \
which includes the detection of text areas, and the recognition of characters.''')
st.markdown(lib)

st.markdown('''**Before** evaluate OCR solutions, you can check if image quality is good enough for \
OCR task. If not, you could try to **enhance** it before, with some processing operations.


👈 Select the :blue-background[:orange[**About**]] page from the sidebar for information on how the app works''')
st.markdown("👈 or select the :blue-background[:orange[**Image processing**]] page to check and enhance your image")
st.markdown("👈 or directly select the :blue-background[:orange[**OCR Comparator**]] page")


#import mim
#
#mim.install(['mmengine>=0.7.1,<1.1.0'])
#mim.install(['mmcv>=2.0.0rc4,<2.1.0'])
#mim.install(['mmdet>=3.0.rc5,<3.2.0'])
#mim.install(['mmocr'])

