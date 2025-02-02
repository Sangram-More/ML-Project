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
        animation_file(r"App/Tabs/Animations/Bank.json"),
        App/Tabs/Animations/Bank.json
        speed = 1,
        reverse = False,
        loop = True,
        quality = "high"
    )
    return animation

def Thinking_Animation():   
    animation = st_lottie(
        animation_file(r"App/Tabs/Animations/Thinking.json"),
        speed = 1,
        reverse = False,
        loop = True,
        quality = "high"
    )
    return animation

def UpTrend_Animation():   
    animation = st_lottie(
        animation_file(r"App/Tabs/Animations/UpTrend.json"),
        speed = 1,
        reverse = False,
        loop = True,
        quality = "high"
    )
    return animation

def DownTrend_Animation():   
    animation = st_lottie(
        animation_file(r"App/Tabs/Animations/DownTrend.json"),
        speed = 1,
        reverse = False,
        loop = True,
        quality = "high"
    )
    return animation

def Speedometer_Animation():   
    animation = st_lottie(
        animation_file(r"App/Tabs/Animations/Speedometer.json"),
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
# Section 6

st.markdown('<h2 style="text-align: justify;"> Think of it like adjusting the speed of a car: the Fed uses interest rates to speed up or slow down the economy as needed!</h2>', unsafe_allow_html=True)

block1, block2, block3 = st.columns([2, 6, 1])

with block2:
    Speedometer_Animation()

st.divider()

# -----------------------------------------------------------
# Section 7

st.subheader("Lets try to understand the Background of Fed Rates")
st.markdown("<p class='justified-text'>The US Federal Reserve, commonly known as the Fed, plays a crucial role in shaping the economy by setting interest rates. These rates influence everything from borrowing costs to inflation and economic growth. Historically, changes in the Fed's interest rate policies have had widespread effects on businesses, consumers, and financial markets. Given its impact, predicting these rate changes has been a major area of interest for economists, investors, and policymakers. Traditionally, financial experts relied on economic indicators, historical trends, and expert opinions to anticipate rate decisions. However, with the growing availability of data and advancements in computing, new approaches have emerged. Machine learning, a branch of artificial intelligence, has opened up possibilities to analyze complex economic patterns and predict rate changes more accurately. By leveraging data-driven insights, we can gain a deeper understanding of how various factors contribute to Fed rate decisions.</p>", unsafe_allow_html=True)

# -----------------------------------------------------------
# Section 8

st.subheader("Now lets try to understand the Basis of Fed Rates")
st.markdown("<p class='justified-text'>The foundation of interest rate decisions lies in macroeconomic factors such as inflation, employment, GDP growth, and global financial conditions. The Federal Reserve assesses these indicators to determine whether the economy needs higher interest rates to control inflation or lower rates to stimulate growth. Traditional financial models use statistical techniques to analyze past trends and make predictions. However, these models often struggle to adapt to rapidly changing economic conditions. Machine learning, on the other hand, can process vast amounts of real-time data, identify hidden patterns, and improve the accuracy of rate predictions. By combining historical financial data with modern computational techniques, researchers and analysts aim to enhance forecasting methods and develop better economic strategies.</p>", unsafe_allow_html=True)

# -----------------------------------------------------------
# Section 9

st.subheader("Why Fed Rates are important?")
st.markdown("<p class='justified-text'>Interest rate predictions are valuable not only for policymakers but also for businesses, investors, and individuals. Changes in Fed rates influence loan interest rates, mortgage payments, credit card costs, and overall investment returns. A well-informed prediction can help businesses plan their financial strategies, assist investors in making informed decisions, and guide policymakers in shaping effective economic policies. Machine learning models analyze past decisions, economic reports, and market behaviors to recognize potential patterns. Unlike traditional models, these AI-driven techniques learn from vast datasets, allowing them to adjust and refine their predictions as new information becomes available. The goal is to bridge the gap between historical economic theories and modern data-driven decision-making.</p>", unsafe_allow_html=True)

# -----------------------------------------------------------
# Section 10

st.subheader("Why use Machine Learning to predict Fed Rates?")
st.markdown("<p class='justified-text'>Predicting US Federal Reserve rate changes is a challenge that has significant implications for financial planning and economic stability. A more accurate forecasting approach can help businesses mitigate risks, enable investors to make better decisions, and assist policymakers in responding proactively to economic shifts. The rise of machine learning presents an opportunity to refine traditional forecasting methods by leveraging data-driven insights. By developing a predictive model, we aim to contribute to a deeper understanding of economic trends and decision-making processes. The motivation behind this project is to explore how artificial intelligence can enhance financial forecasting and provide valuable insights into one of the most influential economic decisions worldwide.</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------
# Section 11

st.subheader("What Questions are we trying to answere here?")
st.markdown(""" 
1. What are the historical trends in the US Federal Reserve interest rates?
2. What macroeconomic indicators (inflation, unemployment, GDP, etc.) are most correlated with the Fed rate changes?
3. Are there seasonal patterns or cycles in Fed rate changes?
4. Which economic factors have the strongest impact on predicting future Fed rate changes?
5. How does inflation (CPI) impact Fed rate decisions?
6. How do unemployment rates and labor market conditions affect Fed rate decisions?
7. Which machine learning model provides the most accurate prediction for Fed rate changes?
8. How well can historical data predict future Fed rate changes?
9. Can we forecast Fed rate changes during economic crises (e.g., 2008 financial crisis, COVID-19)?
10. How do Fed rate changes impact stock markets, bond yields, and the housing market?
""")
