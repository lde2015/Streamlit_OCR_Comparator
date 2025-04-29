import streamlit as st
import cv2
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageColor
import PIL
import os
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def app():
    ###################################################################################################
    ##   INITIALISATIONS
    ###################################################################################################
    ###
    @st.cache_data(show_spinner=False)
    def initializations():

        out_dict_lang_ppocr = {'Abaza': 'abq', 'Adyghe': 'ady', 'Afrikaans': 'af', 'Albanian': 'sq', \
        'Angika': 'ang', 'Arabic': 'ar', 'Avar': 'ava', 'Azerbaijani': 'az', 'Belarusian': 'be', \
        'Bhojpuri': 'bho','Bihari': 'bh','Bosnian': 'bs','Bulgarian': 'bg','Chinese & English': 'ch', \
        'Chinese Traditional': 'chinese_cht', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', \
        'Dargwa': 'dar', 'Dutch': 'nl', 'English': 'en', 'Estonian': 'et', 'French': 'fr', \
        'German': 'german','Goan Konkani': 'gom','Hindi': 'hi','Hungarian': 'hu','Icelandic': 'is', \
        'Indonesian': 'id', 'Ingush': 'inh', 'Irish': 'ga', 'Italian': 'it', 'Japan': 'japan', \
        'Kabardian': 'kbd', 'Korean': 'korean', 'Kurdish': 'ku', 'Lak': 'lbe', 'Latvian': 'lv', \
        'Lezghian': 'lez', 'Lithuanian': 'lt', 'Magahi': 'mah', 'Maithili': 'mai', 'Malay': 'ms', \
        'Maltese': 'mt', 'Maori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn', 'Nagpur': 'sck', \
        'Nepali': 'ne', 'Newari': 'new', 'Norwegian': 'no', 'Occitan': 'oc', 'Persian': 'fa', \
        'Polish': 'pl', 'Portuguese': 'pt', 'Romanian': 'ro', 'Russia': 'ru', 'Saudi Arabia': 'sa', \
        'Serbian(cyrillic)': 'rs_cyrillic', 'Serbian(latin)': 'rs_latin', 'Slovak': 'sk', \
        'Slovenian': 'sl', 'Spanish': 'es', 'Swahili': 'sw', 'Swedish': 'sv', 'Tabassaran': 'tab', \
        'Tagalog': 'tl', 'Tamil': 'ta', 'Telugu': 'te', 'Turkish': 'tr', 'Ukranian': 'uk', \
        'Urdu': 'ur', 'Uyghur': 'ug', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy'}
        return out_dict_lang_ppocr

    ###################################################################################################
    ##   FONTIONS
    ###################################################################################################

    ###
    def load_image(in_image_file):
        """Load input file and open it

        Args:
            in_image_file (string or Streamlit UploadedFile): image to consider

        Returns:
            matrix      : input file opened with Opencv
        """

        #if isinstance(in_image_file, str):
        #    out_image_path = "img."+in_image_file.split('.')[-1]
        #else:
        #    out_image_path = "img."+in_image_file.name.split('.')[-1]

        if isinstance(in_image_file, str):
            out_image_path = "tmp_"+in_image_file
        else:
            out_image_path = "tmp_"+in_image_file.name

        img = Image.open(in_image_file)
        img_saved = img.save(out_image_path)

        # Read image
#        out_image_orig = Image.open(out_image_path)
        out_image_cv2 = cv2.cvtColor(cv2.imread(out_image_path), cv2.COLOR_BGR2RGB)

        return out_image_cv2, out_image_path, out_image_cv2




    ###################################################################################################
    ##   STREAMLIT APP
    ###################################################################################################

    st.title("Image check and enhance for OCR task")

    st.write("")
    st.write("")
    st.write("")

    dict_lang_ppocr = initializations()

    st.markdown("#### Choose picture:")
    cols = st.columns([1, 2])
    img_typ = cols[0].radio("", ['Upload file', 'Take a picture', 'Use a demo file'], \
                                index=0) #, on_change=raz)

    if img_typ == 'Upload file':
        image_file = cols[1].file_uploader("Upload a file:", type=["jpg","jpeg"]) #, on_change=raz)
    """
    if img_typ == 'Take a picture':
        image_file = cols_pict[1].camera_input("Take a picture:", on_change=raz)
    if img_typ == 'Use a demo file':
        with st.expander('Choose a demo file:', expanded=True):
            demo_used = st.radio('', ['File 1', 'File 2'], index=0, \
                                horizontal=True, on_change=raz)
            cols_demo = st.columns([1, 2])
            cols_demo[0].markdown('###### File 1')
            cols_demo[0].image(img_demo_1, width=150)
            cols_demo[1].markdown('###### File 2')
            cols_demo[1].image(img_demo_2, width=300)
            if demo_used == 'File 1':
                image_file = 'img_demo_1.jpg'
            else:
                image_file = 'img_demo_2.jpg'
    """
    ##----------- Process input image -----------------------------------------------------------------
    if image_file is not None:
        img_cv2, image_path, img_wrk = load_image(image_file)

        col1, col2, col3 = st.columns([0.25, 0.25, 0.5]) #gap="medium")

        col1.markdown('#### Original image')
        col1.image(img_cv2, use_column_width=True)
        col1.text('Shape : ' + str(img_cv2.shape))

        cnt_img_wrk = col2.container(height=700, border=False)
        cnt_img_wrk.markdown('#### Processed image')

        col3.markdown('#### Check & enhance')

        with col3.expander("Quick overview of OCR recognition (with PPOCR)", expanded=True):
            with st.form("form1"):
                key_ppocr_lang = st.selectbox("Choose language: :", dict_lang_ppocr.keys(), 20)
                res = st.empty()
                submit_detect = st.form_submit_button("Launch overview")

        with col3.expander("Resize", expanded=False):
            scaling_factor = st.slider("Scaling factor :", 0.1, 10., 1., 0.1)
            img_wrk = cv2.resize(img_cv2, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
            cnt_img_wrk.empty()
            cnt_img_wrk.image(img_wrk, use_column_width=False)
            cnt_img_wrk.text('Shape : ' + str(img_wrk.shape))

        ##----------- Process text detection --------------------------------------------------------------
        if submit_detect:
            ocr = PaddleOCR(lang=dict_lang_ppocr[key_ppocr_lang])

            result = ocr.ocr(img_wrk)
            # draw result
            result = result[0]
            image = img_wrk.copy()
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/french.ttf')
            im_show = Image.fromarray(im_show)
            res.image(im_show)#, width=400)