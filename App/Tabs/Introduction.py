import streamlit as st
from streamlit_lottie import st_lottie
import json
import os


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
    with open(filepath, "r") as f:
        return json.load(f)

def Bank_Animation():   
    animation = st_lottie(
        animation_file("App\Tabs\Animations\Bank.json"),
        speed = 1,
        reverse = False,
        loop = True,
        quality = "medium"
    )
    return animation


# --------------------------------------------------------------

# Title of the page.
st.title("What is this all about?")
  
# -----------------------------------------------------------

# Creating 2 columns to add animation and text side by side.

column1, column2 = st.columns(2, gap="large", vertical_alignment="center")

with column1:
    Bank_Animation()

with column2:
    st.header("What do you know about the term *US Fedral Rates?*")
