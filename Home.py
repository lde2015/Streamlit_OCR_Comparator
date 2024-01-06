import streamlit as st
from multipage import MultiPage
from app_pages import home, about, ocr_comparator

app = MultiPage()
st.set_page_config(
    page_title='OCR Comparator', layout ="wide",
    initial_sidebar_state="expanded",
)

# Add all your application here
app.add_page("Home", "house", home.app)
app.add_page("About", "info-circle", about.app)
app.add_page("App", "cast", ocr_comparator.app)

# The main app
app.run()