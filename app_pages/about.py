import streamlit as st

def app():
    st.title("OCR solutions comparator")

    st.write("")
    st.write("")
    st.write("")

    st.markdown("#####  This app allows you to compare, from a given picture, the results of different solutions:")
    st.markdown("##### *EasyOcr, PaddleOCR, MMOCR, Tesseract*")
    st.write("")
    st.write("")

    st.markdown(''' The 1st step is to choose the language for the text recognition (not all solutions \
    support the same languages), and then choose the picture to consider. It is possible to upload a file, \
    to take a picture, or to use a demo file. \
    It is then possible to change the default values for the text area detection process, \
    before launching the detection task for each solution.''')
    st.write("")

    st.markdown(''' The different results are then presented. The 2nd step is to choose one of these \
    detection results, in order to carry out the text recognition process there. It is also possible to change \
    the default settings for each solution.''')
    st.write("")

    st.markdown("###### The recognition results appear in 2 formats:")
    st.markdown(''' - a visual format resumes the initial image, replacing the detected areas with \
    the recognized text. The background is + or - strongly colored in green according to the \
    confidence level of the recognition.
        A slider allows you to change the font size, another \
    allows you to modify the confidence threshold above which the text color changes: if it is at \
    70% for example, then all the texts with a confidence threshold higher or equal to 70 will appear \
    in white, in black otherwise.''')

    st.markdown(" - a detailed format presents the results in a table, for each text box detected. \
    It is possible to download this results in a local csv file.")