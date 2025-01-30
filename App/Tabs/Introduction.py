import streamlit as st
from streamlit_lottie import st_lottie
import json
import os


#--------------- Custom CSS Styling -----------------------------

# Custom CSS for text justification
st.markdown(
    """
    <style>
    .justified-text {
        text-align: justify;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------

# Url for downloading animations: https://lottiefiles.com/free-animations/free

# There are 2 ways to add animations using streamlit_lottie package:

# Offline method:
# Download the JSON file.

# Function:

# def offlinefile(filepath: str):
#     with open(filepath, "r") as f:
#         return json.load(f)

# animation = offlinefile("filepath")
# st_lottie(
#     animation,
#     speed = 1,
#     reverse = False,
#     loop = True,
#     quality = "low", # "medium", "high"
#     renderer = "svg", # "canvas"
#     height = None,
#     widrh = None,
#     key = xyz
# )
# ----------------------------------------------------------

# Online Method:

# import requests
# def onlinefile(url: str):
#     r = request.get(url)
#     if r.status_code != 200:
#         reuturn None
#     return r.json()

# animation = offlinefile("url")
# st_lottie(
#     animation,
#     speed = 1,
#     reverse = False,
#     loop = True,
#     quality = "low", # "medium", "high"
#     renderer = "svg", # "canvas"
#     height = None,
#     widrh = None,
#     key = xyz
# )
# --------------------------------------------------------------

# Implementing animation function
def animation_file(filepath: str):
    with open(filepath, "rb") as f:
        return json.load(f)

def Bank_Animation():   
    animation = st_lottie(
        animation_file(r"App\Tabs\Animations\Bank.json"),
        speed = 1,
        reverse = False,
        loop = True,
        quality = "high"
    )
    return animation

def Thinking_Animation():   
    animation = st_lottie(
        animation_file(r"App\Tabs\Animations\Thinking.json"),
        speed = 1,
        reverse = False,
        loop = True,
        quality = "high"
    )
    return animation

def UpTrend_Animation():   
    animation = st_lottie(
        animation_file(r"App\Tabs\Animations\UpTrend.json"),
        speed = 1,
        reverse = False,
        loop = True,
        quality = "high"
    )
    return animation

def DownTrend_Animation():   
    animation = st_lottie(
        animation_file(r"App\Tabs\Animations\DownTrend.json"),
        speed = 1,
        reverse = False,
        loop = True,
        quality = "high"
    )
    return animation

def Speedometer_Animation():   
    animation = st_lottie(
        animation_file(r"App\Tabs\Animations\Speedometer.json"),
        speed = 1,
        reverse = False,
        loop = True,
        height=400, 
        width=400
    )
    return animation


# --------------------------------------------------------------

# Title of the page.
st.title("What is this all about?")
  
# -----------------------------------------------------------
# Section 1

# Creating 2 columns to add animation and text side by side.

column1_1, column2_1 = st.columns(2, gap="large", vertical_alignment="center")

with column1_1:
    Bank_Animation()

with column2_1:
    st.header("What do you know about the term *US Fedral Rates?*")
    st.markdown('<p class="justified-text">Banks lend money to each other overnight to meet daily cash needs. The Fed sets a target rate for these loans, which influences all interest rates in the economy.</p>', unsafe_allow_html=True)
    
st.divider()

# -----------------------------------------------------------
# Section 2

# Creating 2 columns to add animation and text side by side.

column1_2, column2_2 = st.columns(2, gap="large", vertical_alignment="center")

with column1_2:
    Thinking_Animation()

with column2_2:
    st.header("Why does it matter?")
    st.markdown('<p class="justified-text">Fed Rate affects everything from credit card rates, home loans, business loans, to how much interest you earn on your savings.</p>', unsafe_allow_html=True)

# -----------------------------------------------------------
# Section 3

# Creating 2 columns to add animation and text side by side.

column1_3, column2_3 = st.columns(2, gap="large", vertical_alignment="center")

with column1_3:
    st.subheader("How does it impact you?")
    st.markdown("""
    #### **When rates go UP:**
    - Borrowing becomes expensive (higher loan & credit card rates).
    - Mortgage and car loan payments increase.
    - Saving money is more rewarding (higher bank interest rates).
    - Economy slows down to control inflation (rising prices).
    """)

with column2_3:
    UpTrend_Animation()

# -----------------------------------------------------------
# Section 4

# Creating 2 columns to add animation and text side by side.

column1_4, column2_4 = st.columns(2, gap="large", vertical_alignment="center")

with column1_4:
    st.markdown("""
    #### **When rates go DOWN:**
    - Borrowing becomes cheaper (lower loan & credit card rates).
    - People & businesses spend more.
    - Economy speeds up, helping job growth.
    """)

with column2_4:
    DownTrend_Animation()

st.divider()

# -----------------------------------------------------------
# Section 5

st.subheader("Why does the Fed change it?")
st.markdown("""
    - To fight inflation, they raise rates to slow down spending.
    - To boost the economy, they cut rates to encourage borrowing and investing.
    """)

st.divider()

# -----------------------------------------------------------
# Section 5

st.markdown('<h2 style="text-align: justify;"> Think of it like adjusting the speed of a car: the Fed uses interest rates to speed up or slow down the economy as needed!</h2>', unsafe_allow_html=True)

block1, block2, block3 = st.columns([2, 6, 1])

with block2:
    Speedometer_Animation()

st.divider()