import streamlit as st
import cv2
import imutils
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import PIL
import os
from time import sleep
import numpy as np
import ast
import operator
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.markdown("""
    <style>
        .main > div:first-of-type {
            padding: 1em 2em 2em 2em;
        }
    </style>
""", unsafe_allow_html=True)

###################################################################################################
##   INITIALISATIONS
###################################################################################################
###
@st.cache_data(show_spinner=True)
def initializations():
    print("Initializations ...")
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

    out_dict_interpolation = {"INTER_LINEAR": cv2.INTER_LINEAR,
                              "INTER_NEAREST": cv2.INTER_NEAREST,
                              "INTER_LINEAR_EXACT": cv2.INTER_LINEAR_EXACT,
                              "INTER_AREA": cv2.INTER_AREA,
                              "INTER_CUBIC": cv2.INTER_CUBIC,
                              "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
                              "INTER_NEAREST_EXACT": cv2.INTER_NEAREST_EXACT,
                              "INTER_MAX": cv2.INTER_MAX,
                              "WARP_FILL_OUTLIERS": cv2.WARP_FILL_OUTLIERS,
                              "WARP_INVERSE_MAP": cv2.WARP_INVERSE_MAP,
                             }

    out_dict_thresholding_type = {"THRESH_BINARY": cv2.THRESH_BINARY,
                                  "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
                                  "THRESH_TRUNC": cv2.THRESH_TRUNC,
                                  "THRESH_TOZERO": cv2.THRESH_TOZERO,
                                 }

    out_dict_adaptative_method = {"ADAPTIVE_THRESH_MEAN_C": cv2.ADAPTIVE_THRESH_MEAN_C,
                                  "ADAPTIVE_THRESH_GAUSSIAN_C": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}

    return out_dict_lang_ppocr, out_dict_interpolation, out_dict_thresholding_type, out_dict_adaptative_method

###################################################################################################
##   FONTIONS
###################################################################################################
###
@st.cache_data(show_spinner=False)
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
#    out_image_orig = Image.open(out_image_path)
    out_image_cv2 = cv2.cvtColor(cv2.imread(out_image_path), cv2.COLOR_BGR2RGB)
    return out_image_cv2, out_image_path
###
def eval_expr(expr):
    """Eval numeric expression
    Args:
        expr (string): numeric expression
    Returns:
       float: eval result
    """
    # Dictionnary of authorized operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):  # nombre
            return node.value
        elif isinstance(node, ast.BinOp):  # opérations binaires
            return operators[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):  # opérations unaires (-n)
            return operators[type(node.op)](_eval(node.operand))
        else:
            raise TypeError(node)
    parsed = ast.parse(expr, mode='eval')
    return _eval(parsed.body)
###
def text_kernel_to_latex(text_eval):
    """Try to parse a kernel text description like: 1/6 * [[1,1],[1,1]]
    Args:
        text_eval (string): the string with the kernel expression
    Returns:
       string: left part of input string before *
       list: right part of input string after *
       string: latex expression corresponding to the text kernel input
    """
    text_coeff = text_eval.split('*')[0].strip()
    text_kernel = text_eval.split('*')[-1].strip()
    list_kernel = ast.literal_eval(text_kernel)
    latex = "\\begin{bmatrix}\n"
    for row in list_kernel:
        latex += " & ".join(map(str, row)) + " \\\\\n"
    latex += "\\end{bmatrix}"
    return text_coeff, list_kernel, text_coeff + ' ' + latex
###
def get_img_fig(img):
    """Plot image with matplotlib, in order to have image size
    Args:
        img (Image): Image to show
    Returns:
        Matplotlib figure
    """
    fig = plt.figure()
    if len(img.shape) == 3:
        plt.imshow(img, cmap=None)
    else:
        plt.imshow(img, cmap='gray')
    return fig

###################################################################################################
##   STREAMLIT APP
###################################################################################################
st.title(''':orange[Image check and enhance for OCR task]''')
st.write("")
st.write("")
st.write("")

dict_lang_ppocr, dict_interpolation, dict_thresholding_type, dict_adaptative_method = initializations()

cols = st.columns([0.25, 0.25, 0.5])
cols[0].markdown("#### :orange[Choose picture:]")
img_typ = cols[0].radio("#### :orange[Choose picture type:]", ['Upload file', 'Take a picture', 'Use a demo file'], \
                            index=0) #, on_change=raz)
if img_typ == 'Upload file':
    image_file = cols[1].file_uploader("Upload a file:", type=["png","jpg","jpeg"]) #, on_change=raz)
_ = """
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
    img_cv2, image_path = load_image(image_file)

    cols[2].markdown('#### :orange[Original image]')
    cnt_img_ori = cols[2].container(height=300, border=False)
    #cnt_img_ori.image(img_cv2) #, use_container_width=True)
    cnt_img_ori.pyplot(get_img_fig(img_cv2))
    col1, col2 = st.columns([0.5, 0.5]) #gap="medium")

    col1.markdown('#### :orange[Processed image]')
    list_op = []

    if col1.checkbox("GrayScale"):
        try:
            img_first = cv2.cvtColor(img_cv2.copy(), cv2.COLOR_BGR2GRAY)
            list_op.append("Grayscale")
        except Exception as e:
            st.exception(e)
    else:
        img_first = img_cv2.copy()

    if col1.checkbox("Bit-wise inversion"):
        try:
            img_first = cv2.bitwise_not(img_first)
            list_op.append("Bit-wise inversion")
        except Exception as e:
            st.exception(e)

    img_wrk = img_first.copy()

    cnt_img_wrk = col1.container(height=500, border=False)
    img_processed = cnt_img_wrk.empty()
    img_processed.pyplot(get_img_fig(img_wrk))

    col2.markdown('#### :orange[Check & enhance]')

    with col2.expander(":blue[Image processing]", expanded=False):
        tab1, tab2, tab3, tab4, tab5 = \
                st.tabs(["Resize", "Rotate", "Filtering",
                          "Morphologie", "Thresholding"])

        with tab1: # Resize
            scaling_factor = st.slider("Scaling factor :", 0.1, 10., 1., 0.1)
            cols_tab1 = st.columns([0.1, 0.9], gap="medium", vertical_alignment="center")
            cols_tab1[0].markdown("💬", help="""An interpolation function’s goal is
    to examine neighborhoods of pixels and use these neighborhoods to optically increase or decrease
    the size of the image without introducing distortions (or at least as few distortions
    as possible).\n
    ```cv2.INTER_LINEAR``` This option uses the bilinear interpolation algorithm. Unlike INTER_NEAREST,
    this does the interpolation in two dimensions and predicts the function used to calculate the color
    of a pixel. This algorithm is effective in handling visual distortions while zooming or
    enlarging an image.\n
    ```cv2.INTER_NEAREST``` This option uses the nearest neighbor interpolation algorithm. It retains
    the sharpness of the edges though the overall image may be blurred.\n
    ```cv2.INTER_LINEAR_EXACT```is a modification of ```INTER_LINEAR``` and both uses bilinear
    interpolation algorithm. The only difference is that the calculations in ```INTER_LINEAR_EXACT```
    are accurate to a bit.\n
    ```cv2.INTER_AREA``` option uses resampling using pixel area relation technique. While enlarging
    images, INTER_AREA work same as INTER_NEAREST. In other cases, ```INTER_AREA works``` better in
    image decimation and avoiding false inference patterns in images (moire pattern).\n
    ```cv2.INTER_CUBIC``` option uses bicubic interpolation technique. This is an extension of cubic
    interpolation technique and is used for 2 dimension regular grid patterns.\n
    ```cv2.INTER_LANCZOS4``` option uses Lanczos interpolation over 8 x 8 pixel neighborhood technique.
    It uses Fourier series and Chebyshev polynomials and is suited for images with large number of
    small size details.\n
    ```cv2.INTER_NEAREST_EXACT ``` is a modification of INTER_NEAREST with bit level accuracy.\n
    ```cv2.INTER_MAX ``` option uses mask for interpolation codes.\n
    ```cv2.WARP_FILL_OUTLIERS ``` interpolation technique skips the outliers during interpolation calculations.\n
    ```cv2.WARP_INVERSE_MAP ``` option uses inverse transformation technique for interpolation.\n""")
            cols_tab1[0].link_button("📚", "https://iq.opengenus.org/different-interpolation-methods-in-opencv/")
            interpolation = cols_tab1[1].selectbox("Interpolation method:", list(dict_interpolation.keys()))
            if st.toggle("Apply", key=1):
                try:
                    img_wrk = cv2.resize(img_wrk, None, fx=scaling_factor, fy=scaling_factor,
                                        interpolation=dict_interpolation[interpolation])
                    img_processed.pyplot(get_img_fig(img_wrk))
                    list_op.append("Resize - fx="+str(scaling_factor)+", fy="+str(scaling_factor)+ \
                                ", interpolation="+interpolation)
                except Exception as e:
                    st.exception(e)

        with tab2: # Rotate
            angle = st.slider("Angle :", 0, 360, 0, step=10)
            if angle != 0:
                try:
                    img_wrk = imutils.rotate(img_wrk, angle=angle)
                    img_processed.pyplot(get_img_fig(img_wrk))
                    list_op.append("Rotate - angle="+str(angle))
                except Exception as e:
                    st.exception(e)

        with tab3: # Filtering
            st.write("📚 :blue[*More about image filtering*]  👉  \
                        [here](https://learnopencv.com/image-filtering-using-convolution-in-opencv/)")
            selection = st.segmented_control("Filtering type",
                                            ["Custom 2D Convolution", "Blurring"],
                                            selection_mode="single")
            match selection:
                case "Custom 2D Convolution":
                        st.write("📚 :blue[*More about convolution matrix*]  👉  \
                                    [here](https://en.wikipedia.org/wiki/Kernel_(image_processing))")
                        text_convol = st.text_input("Write your custom kernel here (example : 1/9 * [[1,1,1], [1,1,1], [1,1,1]]):",
                                                    value=None)
                        if text_convol is not None:
                            try:
                                text_coeff, list_kernel, latex_code = text_kernel_to_latex(text_convol)
                                coeff = eval_expr(text_coeff)
                                kernel = coeff * np.array(list_kernel)
                                st.latex(latex_code)
                            except Exception as e:
                                st.exception(e)
                                text_convol = None
                            try:
                                img_wrk = cv2.filter2D(src=img_wrk, ddepth=-1, kernel=kernel)
                                img_processed.pyplot(get_img_fig(img_wrk))
                                list_op.append("Filtering - Custom 2D Convolution - kernel="+ text_convol)
                            except Exception as e:
                                st.exception(e)
                        else:
                            text_coeff, list_kernel, latex_code = \
                                        text_kernel_to_latex("1/9 * [[1,1,1], [1,1,1], [1,1,1]]")
                            st.latex(latex_code)

                case "Blurring":
                    st.write("📚 :blue[*More about blurring techniques*]  👉  \
                                [here](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)")
                    typ_blurring = st.segmented_control("Bluring type",
                                        ["Averaging", "Gaussian Blurring", "Median Blurring",
                                        "Bilateral Filtering"],
                                        selection_mode="single")
                    match typ_blurring:
                        case "Averaging":
                                st.markdown("💬 :green[Averaging?]",
                                            help="This is done by convolving an image with a normalized box filter.\
                                        It simply takes the average of all the pixels under the kernel \
                                        area and replaces the central element."
                                           )
                                kernel_width = st.slider("Kernel size width:", 2, 20, None, 1)
                                kernel_height = st.slider("Kernel size height:", 2, 20, None, 1)
                                if st.toggle("Apply", key=2):
                                    kernel_size = (kernel_width, kernel_height)
                                    try:
                                        img_wrk = cv2.blur(src=img_wrk, ksize=kernel_size)
                                        img_processed.pyplot(get_img_fig(img_wrk))
                                        list_op.append("Filtering - Averaging - kernel_size="+str(kernel_size))
                                    except Exception as e:
                                        st.exception(e)

                        case "Gaussian Blurring":
                            st.markdown("💬 :green[Gaussian Blurringing?]",
                                            help="In this method, instead of a box filter, a Gaussian kernel is used. \
We should specify the width and height of the kernel which should be positive and odd. \
We also should specify the standard deviation in the X and Y directions, `sigmaX` and `sigmaY` respectively. \
If only `sigmaX` is specified, `sigmaY` is taken as the same as sigmaX. If both are given as zeros, they are \
calculated from the kernel size.\n \
Gaussian blurring is highly effective in removing Gaussian noise from an image.")
                            kernel_width = st.slider("Kernel size width:", 2, 20, None, 1,)
                            kernel_height = st.slider("Kernel size height:", 2, 20, None, 1)
                            st.markdown("Standard deviations of the Gaussian kernel:",
                                        help="""The parameters `sigmaX` and `sigmaY` represent the standard deviations
                                                of the Gaussian kernel in the horizontal (X) and vertical (Y) directions,
                                                respectively. These values control the extent of blurring applied to the image.​\n
**Typical Values for sigmaX and sigmaY:**
- Low values (e.g., 1–3): Apply a mild blur, useful for slight noise reduction while preserving image details.​
- Moderate values (e.g., 5–10): Produce a more noticeable blur, helpful for reducing more significant noise or smoothing out textures.
- High values (e.g., >10): Result in a strong blur, which can be used for artistic effects or to obscure details.​
It's common practice to set sigmaX and sigmaY to 0. In this case, OpenCV calculates the standard deviations based on the kernel size (ksize).
If only sigmaX is specified and sigmaY is set to 0, OpenCV uses the same value for both directions. ​\n
**Recommendations:**
- Specify sigmaX and sigmaY explicitly: For precise control over the blurring effect, define both parameters based on the desired outcome.​
- Use sigmaX = 0 and sigmaY = 0: To allow OpenCV to compute the standard deviations automatically from the kernel size.​
- Choose an appropriate kernel size: The ksize parameter should be a tuple of positive odd integers (e.g., (3, 3), (5, 5)).
                    """)
                            sigmaX = st.slider("sigmaX:", 0, 20, 0, 1)
                            sigmaY = st.slider("sigmaY:", 0, 20, 0, 1)
                            if st.toggle("Apply", key=3):
                                kernel_size = (kernel_width, kernel_height)
                                try:
                                    img_wrk = cv2.GaussianBlur(src=img_wrk, ksize=kernel_size, \
                                                               sigmaX=sigmaX, sigmaY=sigmaY)
                                    img_processed.pyplot(get_img_fig(img_wrk))
                                    list_op.append("Filtering - Gaussian Blurring - ksize="+ \
                                                   str(kernel_size)+", sigmaX="+str(sigmaX)+ \
                                                    ", sigmaY="+str(sigmaY))
                                except Exception as e:
                                    st.exception(e)

                        case "Median Blurring":
                            st.markdown("💬 :green[Median Blurring?]",
                                        help="It takes the median of all the pixels under the \
kernel area and the central element is replaced with this median value.  Interestingly, in the above \
filters, the central element is a newly calculated value which may be a pixel value in the image or a new value. \
But in median blurring, the central element is always replaced by some pixel value in the image. \
It reduces the noise effectively. Its kernel size should be a positive odd integer.\n \
Median blurring is highly effective against salt-and-pepper noise in an image.")
                            kernel_size = st.slider("Kernel size:", 3, 15, None, 2, key=101)
                            if st.toggle("Apply", key=4):
                                try:
                                    img_wrk = cv2.medianBlur(img_wrk, kernel_size)
                                    img_processed.pyplot(get_img_fig(img_wrk))
                                    list_op.append("Filtering - Median Blurring - kernel_size="+ \
                                                   str(kernel_size))
                                except Exception as e:
                                    st.exception(e)

                        case "Bilateral Filtering":
                            st.markdown("💬 :green[Bilateral Filtering?]",
                                        help="It is highly effective in noise removal while \
keeping edges sharp. But the operation is slower compared to other filters. We already saw that a \
Gaussian filter takes the neighbourhood around the pixel and finds its Gaussian weighted average. \
This Gaussian filter is a function of space alone, that is, nearby pixels are considered while \
filtering. It doesn't consider whether pixels have almost the same intensity. It doesn't consider \
whether a pixel is an edge pixel or not. So it blurs the edges also, which we don't want to do.\n \
Bilateral filtering also takes a Gaussian filter in space, but one more Gaussian filter which is \
a function of pixel difference. \
The Gaussian function of space makes sure that only nearby pixels are considered for blurring, \
while the Gaussian function of intensity difference makes sure that only those pixels with similar \
intensities to the central pixel are considered for blurring. \
So it preserves the edges since pixels at edges will have large intensity variation.")
                            st.markdown("Diameter of each pixel neighborhood that is used during filtering:",
                                        help=""" **Effect:**\n
A larger `d` value means that more neighboring pixels are considered in the filtering process, leading to a more pronounced
blurring effect. Conversely, a smaller `d` focuses the filter on a tighter area, preserving more details.​
**Automatic Calculation:**\n
If `d` is set to a non-positive value (e.g., 0 or negative), OpenCV automatically calculates it based on the sigmaSpace parameter.
Specifically, the radius is computed as `radius = cvRound(sigmaSpace * 1.5)`, and then `d = radius * 2 + 1` to ensure it's an odd
number. This ensures that the kernel has a central pixel. ​
**Typical Values for `d`:**\n
The choice of d depends on the desired balance between noise reduction and edge preservation:​
- Small d (e.g., 5 to 9): Suitable for subtle smoothing while maintaining edge sharpness.​
- Medium d (e.g., 9 to 15): Offers a balance between noise reduction and detail preservation.​
- Large d (e.g., 15 and above): Provides stronger blurring, which may be useful for artistic effects but can lead to loss of
fine details.​
**Recommendations:**\n
- Large filters (d > 5) are very slow, so it is recommended to use `d=5` for real-time applications, and perhaps
`d=9` for offline applications that need heavy noise filtering.
- Start with Moderate Values: Begin with `d=9`, `sigmaColor=75`, and `sigmaSpace=75` as a baseline. Adjust these values based on
the specific requirements of your application.​
- Consider Image Size: For larger images, you might need to increase `d` to achieve a noticeable effect. Conversely,
for smaller images, a smaller `d` might suffice.​
- Balance with `sigmaColor` and `sigmaSpace`: Ensure that `d` is appropriately balanced with `sigmaColor` and
`sigmaSpace`. An excessively large `sigmaSpace` with a small `d` might not utilize the full potential of the spatial filtering.
                                            """)
                            d_value = st.slider("d:", 3, 15, None, 2)
                            st.markdown("`sigmaColor` and `sigmaSpace`:", help="""
`sigmaColor`: This parameter defines the filter sigma in the color space. A larger value means that pixels with more significant
color differences will be mixed together, resulting in areas of semi-equal color.​
`sigmaSpace`: This parameter defines the filter sigma in the coordinate space. A larger value means that pixels farther apart
will influence each other as long as their colors are close enough.​\n
These parameters work together to ensure that the filter smooths the image while preserving edges.​
**Typical Values for `sigmaColor` and `sigmaSpace`:**\n
The choice of `sigmaColor` and `sigmaSpace` depends on the specific application and the desired effect.
However, some commonly used values are:​
- `sigmaColor`: Values around 75 are often used for general smoothing while preserving edges.​
- `sigmaSpace`: Similarly, values around 75 are typical for maintaining edge sharpness while reducing noise.​
For example, applying the bilateral filter with `d=9`, `sigmaColor=75`, and `sigmaSpace=75` is a common practice.
**Recommendations:**`\n
- Start with Equal Values: Setting `sigmaColor` and `sigmaSpace` to the same value (e.g., 75) is a good starting point.​
- Adjust Based on Results: If the image appears too blurred, reduce the values. If noise is still present, increase them.​
- Consider Image Characteristics: For images with high noise, higher values may be necessary. For images where edge preservation
is critical, lower values are preferable.""")
                            sigma_color = st.slider("sigmaColor", 1, 255, None, 1)
                            sigma_space = st.slider("sigmaSpace", 1, 255, None, 1)
                            if st.toggle("Apply", key=5):
                                try:
                                    img_wrk = cv.bilateralFilter(img_wrk, d, sigma_color, sigma_space)
                                    img_processed.pyplot(get_img_fig(img_wrk))
                                    list_op.append("Filtering - Bilateral Filtering - d="+ \
                                                  str(d)+", sigma_color="+str(sigma_color)+ \
                                                  ", sigma_space="+str(sigma_space))
                                except Exception as e:
                                    st.exception(e)

        with tab4: # Morphologie
            list_select = st.segmented_control("Morphological operation:",
                                              ["Erosion", 'Dilation'],
                                              selection_mode="multi")
            if "Erosion" in list_select:
                st.markdown("💬 :green[Erosion?]",
                                help="The basic idea of erosion is just like soil erosion only, it erodes \
away the boundaries of foreground object (Always try to keep foreground in white). \
So what it does? The kernel slides through the image (as in 2D convolution). A pixel in the \
original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel is 1, \
otherwise it is eroded (made to zero). \n \
So what happends is that, all the pixels near boundary will be discarded depending upon the \
size of kernel. So the thickness or size of the foreground object decreases or simply white region \
decreases in the image. \n\
It is useful for removing small white noises, detach two connected objects etc. \n \
:orange[**Best practice :** convert to grayscale before apply erosion.]​")
                kernel_size_ero = st.slider("Kernel size:", 3, 21, 3, 2, key=102)
                nb_iter = st.slider('Iterations number:', 1, 7, 1, 1, key=201)
                kernel = np.ones((kernel_size_ero,kernel_size_ero),np.uint8)
                erosion = cv2.erode(img_wrk, kernel, iterations=nb_iter)
                if st.toggle("Apply", key=8):
                    try:
                        img_wrk = cv2.erode(img_wrk, kernel, iterations=nb_iter)
                        img_processed.pyplot(get_img_fig(img_wrk))
                        list_op.append("Erosion - kernel_size="+str(kernel_size_ero)+ \
                                    ", iterations="+str(nb_iter))
                    except Exception as e:
                        st.exception(e)

            if "Dilation" in list_select:
                st.markdown("💬 :green[Dilation?]",
                                help="The opposite of an erosion is a dilation. Just like an \
erosion will eat away at the foreground pixels, a dilation will grow the foreground pixels. \
Dilations increase the size of foreground objects and are especially useful for joining broken \
parts of an image together. Dilations, just as an erosion, also utilize structuring elements \
— a center pixel p of the structuring element is set to white if ANY pixel in the structuring \
element is > 0. \n \
:orange[**Best practice :** convert to grayscale before apply dilation.]​")
                kernel_size_dil = st.slider("Kernel size:", 3, 21, 3, 2, key=103)
                nb_iter = st.slider('Iterations number:', 1, 7, 1, 1, key=202)
                kernel = np.ones((kernel_size_dil,kernel_size_dil),np.uint8)
                erosion = cv2.dilate(img_wrk, kernel, iterations=nb_iter)
                if st.toggle("Apply", key=9):
                    try:
                        img_wrk = cv2.erode(img_wrk, kernel, iterations=nb_iter)
                        img_processed.pyplot(get_img_fig(img_wrk))
                        list_op.append("Erosion - kernel_size="+str(kernel_size_dil)+ \
                                       ", iterations="+str(nb_iter))
                    except Exception as e:
                        st.exception(e)

        with tab5: # Thresholding
            selection = st.segmented_control("Type:", ["Binarization", "Adaptative thresholding"])
            match selection:
                case "Binarization":
                    st.markdown("💬 :green[What is thresholding?]",
                                help='''Thresholding is the binarization of an image. In general, we seek to
                                        convert a grayscale image to a binary image, where the pixels are either
                                        0 or 255.
                                        A simple thresholding example would be selecting a threshold value T,
                                        and then setting all pixel intensities less than T to 0, and all pixel
                                        values greater than T to 255. In this way, we are able to create a binary
                                        representation of the image.''')
                    st.markdown("*:orange[⚠ Image must be in gray scale]*")
                    cols_tab1 = st.columns([0.1, 0.9], gap="medium", vertical_alignment="center")
                    with cols_tab1[1]:
                        thresholding_type = cols_tab1[1].selectbox("Thresholding type:",
                                                                list(dict_thresholding_type.keys()))
                        with cols_tab1[0].popover(":material/info:", help="Help on thresholding type",
                                                use_container_width=False):
                            st.link_button("📚:blue[cf. OpenCV documentation :]",
                                        "https://docs.opencv.org/3.0-beta/modules/imgproc/doc/miscellaneous_transformations.html#threshold")

                    thresh = st.slider("Thresh :", 0, 255, 255, 1)
                    if thresholding_type in ["cv.THRESH_BINARY", "cv.THRESH_BINARY_INV"]:
                        value = st.slider("Value :", 0, 255, 255, 1)
                    else:
                        value = 255

                    cols_tab3 = st.columns(2, gap="medium", vertical_alignment="center")
                    otsu = cols_tab3[0].checkbox("Optimum Global Thresholding using Otsu’s Method?",
                                                help='''Otsu’s method tries to find a threshold value
                                                    which minimizes the weighted within-class variance.
                                                    Since Variance is the spread of the distribution
                                                        about the mean. Thus, minimizing the within-class
                                                        variance will tend to make the classes compact.''')
                    cols_tab3[1].link_button("📚:blue[Documentation]",
                                            "https://theailearner.com/2019/07/19/optimum-global-thresholding-using-otsus-method/")

                    thresh_typ = dict_thresholding_type[thresholding_type]
                    if st.toggle("Apply", key=6):
                        if otsu:
                            thresh_typ = thresh_typ+cv2.THRESH_OTSU
                        try:
                            ret, img_wrk = cv2.threshold(img_wrk, thresh, value, \
                                                        dict_thresholding_type[thresholding_type])
                            img_processed.pyplot(get_img_fig(img_wrk))
                            list_op.append("Thresholding - thresh="+str(tresh)+", maxval="+str(value)+ \
                                        ", type="+thresh_typ+", otsu="+str(otsu))
                        except Exception as e:
                            st.exception(e)

                case "Adaptative thresholding":
                    st.markdown("💬 :green[What is adaptative thresholding?]",
                                        help='''This is a usefull technique when dealing with images having non-uniform illumination.
                                        In this, the threshold value is calculated separately for each pixel using
                                        some statistics obtained from its neighborhood. This way we will get different thresholds
                                        for different image regions and thus tackles the problem of varying illumination.''')
                    st.markdown("*:orange[⚠ Image must be in gray scale]*")
                    thresholding_type = st.selectbox("Thresholding type:",
                                                    list(dict_thresholding_type.keys())[:2])
                    max_value = st.slider("Max value :", 0, 255, 255, 1,
                                        help="""This is the value assigned to the pixels after thresholding.
                                            This depends on the thresholding type. If the type is cv2.THRESH_BINARY,
                                            all the pixels greater than the threshold are assigned this maxValue.""")
                    adaptative_method = st.selectbox("Adaptative method:",
                                                    list(dict_adaptative_method.keys()),
                                                    help="""This tells us how the threshold is calculated from the pixel neighborhood.
            This currently supports two methods:
            - cv2.ADAPTIVE_THRESH_MEAN_C: In this, the threshold value is the mean of the neighborhood area.\n
            - cv2.ADAPTIVE_THRESH_GAUSSIAN_C: In this, the threshold value is the weighted sum of the
            neighborhood area. This uses Gaussian weights computed using getGaussiankernel() method.""")
                    block_size = st.slider("Block size:", 3, 21, 3, 2,
                                            help='''**🔍 What is blockSize?**\n
            In adaptive thresholding, the threshold for each pixel is determined based on a local neighborhood around it.
            The blockSize parameter specifies the size of this neighborhood.
            Specifically, it defines the dimensions of the square region (of size blockSize × blockSize) centered on the pixel being processed.
            The threshold is then calculated based on the pixel values within this region.​\n
            **✅ Acceptable Values for blockSize**\n
            Must be an odd integer greater than 1: This ensures that the neighborhood has a central pixel.​
            Common choices: 3, 5, 7, 9, 11, 13, 15, etc.​
            Even numbers are invalid: Using an even blockSize (e.g., 2, 4, 6) would result in an error because
            there would be no central pixel in the neighborhood.​\n
            **🎯 Impact of blockSize on Thresholding**\n
            Smaller blockSize (e.g., 3 or 5):​\n
            - Captures fine details and small variations in illumination.​
            - May be more sensitive to noise.​\n
            Larger blockSize (e.g., 15 or 21):​\n
            - Provides smoother thresholding, reducing the effect of noise.​
            - Might overlook small features or details.

            Choosing the appropriate blockSize depends on the specific characteristics of your image and the details you wish to preserve or suppress.''')

                    const = st.slider("C:", -10, 20, 0, 1,
                                            help='''The parameter C serves as a constant subtracted from the computed mean or weighted mean of the
                                                neighborhood pixels. This subtraction fine-tunes the thresholding process, allowing for better control
                                                over the binarization outcome.
            **🎯 Typical Values for C**
            The optimal value for C varies depending on the image's characteristics, such as lighting conditions and noise levels. Commonly used values include:​
            - 2 to 10: These values are often effective for standard images with moderate lighting variations.​
            - Higher values (e.g., 15 or 20): Useful for images with significant noise or when a more aggressive thresholding is needed.​
            - Negative values: Occasionally used to make the thresholding more lenient, capturing lighter details that might otherwise be missed.​

            It's advisable to experiment with different C values to determine the most suitable one for your specific application. ''')

                    if st.toggle("Apply", key=7):
                        try:
                            img_wrk = cv2.adaptiveThreshold(img_wrk, max_value, dict_adaptative_method[adaptative_method],
                                                dict_thresholding_type[thresholding_type], block_size, const)
                            img_processed.pyplot(get_img_fig(img_wrk))
                            maxValue, adaptiveMethod, thresholdType, blockSize, C
                            list_op.append("Adaptative thresholding - maxValue="+str(max_value)+ \
                                        ", adaptiveMethod="+adaptative_method+", thresholdType"+ \
                                        ", thresholding_type="+thresholding_type+", blockSize="+ \
                                        str(block_size)+", C="+str(const) )
                        except Exception as e:
                            st.exception(e)

    st.write(list_op)

    with col2.expander(":blue[Quick overview of OCR recognition (with PPOCR)]", expanded=True):
        with st.form("form1"):
            key_ppocr_lang = st.selectbox("Choose language: :", dict_lang_ppocr.keys(), 20)
            res_cnt = st.empty()
            submit_detect = st.form_submit_button("Launch overview")

    ##----------- Process OCR --------------------------------------------------------------
    if submit_detect:
        with res_cnt, st.spinner("PPOCR initialization ..."):
            ocr = PaddleOCR(lang=dict_lang_ppocr[key_ppocr_lang], show_log=False)
        with res_cnt, st.spinner("OCR process ..."):
            result = ocr.ocr(img_wrk)
        # draw result
        result = result[0]
        if len(img_wrk.shape) == 3:
            image = img_wrk.copy()
        else:
            image = cv2.cvtColor(img_wrk, cv2.COLOR_GRAY2RGB)
        boxes = [line[0] for line in result]

        txts = [line[1][0] for line in result]
        print("txts:", txts)
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/french.ttf')
        im_show = Image.fromarray(im_show)
        res_cnt.image(im_show, use_container_width=True)