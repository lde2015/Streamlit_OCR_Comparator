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

st.markdown('''#### :orange[OCR solutions comparator]''')
st.write("")
lib = "This application's tab allows you to compare, from a given image, the results of different solutions:  \n*EasyOcr, PaddleOCR, MMOCR, Tesseract*"
st.markdown(lib)

with st.expander("See details:"):
    st.markdown(''' The 1st step is to choose the language for the text recognition (not all solutions \
    support the same languages), and then choose the picture to consider. It is possible to upload a file, \
    to take a picture, or to use a demo file. \
    It is then possible to change the default values for the text area detection process, \
    before launching the detection task for each solution.\n
    The different results are then presented.

    The 2nd step is to choose one of these \
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


st.markdown('-----')


lib = "But, :orange[before evaluating OCR solutions], you may want to verify that the image is \
       of sufficient quality for an OCR task. And if it isn't, you might want to be able to \
       improve its quality.\n This is what the application's tab 'Image processing' allows you to do."
st.markdown(lib)

st.markdown('''#### :orange[Image quality verification and improvement]''')
with st.expander("See details:", expanded=True):
    st.write('''Here, you can run a recognition test with basic PPOCR. This displays the text areas \
             detected in the image, along with their values and probabilities.''')
    st.markdown('The **Image Processing** tab offers several operations: resize, rotate, filtering, \
             morphological transformations, and thresholding.')
    st.write('Special care has been taken to document the various operations, including links to \
             documentation and explanatory pop-ups for the various parameters:')
    st.image("doc.png", width=500)
    st.markdown('For each operation, you can choose whether or not to apply the transformation \
                to the image being processed using the **"Apply"** toggle.')
    st.markdown('When the result is satisfactory, you can view a detailed list of the various \
                operations performed on the original image using the **"List of operations"** button.')
    st.markdown('Finally, you can download the resulting image.')