import streamlit as st
#from multipage import MultiPage
#from app_pages import home, about, ocr_comparator, enhance
#from mmocr.utils.ocr import MMOCR
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(
    page_title='OCR Comparator', layout ="wide",
    initial_sidebar_state="expanded",
)

# https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded&selected=Material+Symbols+Rounded:info:FILL@0;wght@400;GRAD@0;opsz@24&icon.query=info&icon.size=24&icon.color=%235f6368

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

page1 = st.Page("home_page.py", title="Home", icon=":material/home:")
page2 = st.Page("about.py", title="About", icon=":material/info:")
page3 = st.Page("enhance.py", title="Image processing", icon=":material/edit_square:")
page4 = st.Page("ocr_comparator.py", title="OCR Comparator", icon=":material/smart_display:")

pg = st.navigation([page1, page2, page3, page4])

pg.run()