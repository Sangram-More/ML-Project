import streamlit as st
from streamlit_lottie import st_lottie
import json

# --------------- Custom CSS Styling -----------------------------
st.markdown(
    """
    <style>
    .justified-text {
        text-align: justify;
        line-height: 1.8;
    }
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .skill-badge {
        display: inline-block;
        background: #e8f4f8;
        color: #1a1a2e;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.9em;
    }
    .project-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Animation Functions

def animation_file(filepath: str):
    with open(filepath, "rb") as f:
        return json.load(f)

def Bank_Animation():
    return st_lottie(
        animation_file(r"App/Tabs/Animations/Bank.json"),
        speed=1,
        reverse=False,
        loop=True,
        quality="high"
    )

def Thinking_Animation():
    return st_lottie(
        animation_file(r"App/Tabs/Animations/Thinking.json"),
        speed=1,
        reverse=False,
        loop=True,
        quality="high"
    )

def UpTrend_Animation():
    return st_lottie(
        animation_file(r"App/Tabs/Animations/UpTrend.json"),
        speed=1,
        reverse=False,
        loop=True,
        quality="high"
    )

def DownTrend_Animation():
    return st_lottie(
        animation_file(r"App/Tabs/Animations/DownTrend.json"),
        speed=1,
        reverse=False,
        loop=True,
        quality="high"
    )

def Speedometer_Animation():
    return st_lottie(
        animation_file(r"App/Tabs/Animations/Speedometer.json"),
        speed=1,
        reverse=False,
        loop=True,
        height=400,
        width=400
    )

# -----------------------------------------------------------------------------
# Hero Section - Project Overview

st.markdown("""
<div class="project-header">
    <h1 style="margin:0; font-size: 2.5em;">Predicting Federal Reserve Interest Rates</h1>
    <h3 style="margin-top: 0.5rem; font-weight: normal; opacity: 0.9;">Using Machine Learning to Forecast Economic Policy Decisions</h3>
    <p style="margin-top: 1rem; opacity: 0.8;">An end-to-end machine learning project demonstrating data collection, preprocessing, exploratory analysis, and predictive modeling with <strong>97% accuracy</strong></p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Key Metrics Row

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: #667eea; margin: 0;">97%</h2>
        <p style="margin: 0; color: #666;">Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: #667eea; margin: 0;">9</h2>
        <p style="margin: 0; color: #666;">ML Algorithms</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: #667eea; margin: 0;">70+</h2>
        <p style="margin: 0; color: #666;">Years of Data</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: #667eea; margin: 0;">8</h2>
        <p style="margin: 0; color: #666;">Economic Indicators</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("###")

# -----------------------------------------------------------------------------
# Skills & Technologies Section

with st.expander("**Technical Skills Demonstrated in This Project**", expanded=True):
    st.markdown("""
    #### Machine Learning & Data Science
    <span class="skill-badge">Supervised Learning</span>
    <span class="skill-badge">Unsupervised Learning</span>
    <span class="skill-badge">Random Forest</span>
    <span class="skill-badge">SVM</span>
    <span class="skill-badge">Decision Trees</span>
    <span class="skill-badge">Naive Bayes</span>
    <span class="skill-badge">Logistic Regression</span>
    <span class="skill-badge">K-Means Clustering</span>
    <span class="skill-badge">Hierarchical Clustering</span>
    <span class="skill-badge">DBSCAN</span>
    <span class="skill-badge">PCA</span>
    <span class="skill-badge">Association Rule Mining</span>

    #### Data Engineering & Analysis
    <span class="skill-badge">Python</span>
    <span class="skill-badge">Pandas</span>
    <span class="skill-badge">NumPy</span>
    <span class="skill-badge">Scikit-learn</span>
    <span class="skill-badge">API Integration</span>
    <span class="skill-badge">Data Cleaning</span>
    <span class="skill-badge">Feature Engineering</span>
    <span class="skill-badge">Time Series Analysis</span>
    <span class="skill-badge">Statistical Analysis</span>

    #### Visualization & Deployment
    <span class="skill-badge">Matplotlib</span>
    <span class="skill-badge">Seaborn</span>
    <span class="skill-badge">Plotly</span>
    <span class="skill-badge">Streamlit</span>
    <span class="skill-badge">Interactive Dashboards</span>
    <span class="skill-badge">Data Storytelling</span>
    """, unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 1: What are Fed Rates?

st.title("Understanding Federal Reserve Interest Rates")

column1_1, column2_1 = st.columns(2, gap="large", vertical_alignment="center")

with column1_1:
    Bank_Animation()

with column2_1:
    st.header("What are US Federal Rates?")
    st.markdown('<p class="justified-text">Banks lend money to each other overnight to meet daily cash needs. The Federal Reserve sets a target rate for these loans, which influences all interest rates in the economy. This benchmark rate is one of the most powerful tools for controlling economic growth and inflation.</p>', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 2: Why it matters

column1_2, column2_2 = st.columns(2, gap="large", vertical_alignment="center")

with column1_2:
    Thinking_Animation()

with column2_2:
    st.header("Why Does It Matter?")
    st.markdown('<p class="justified-text">Fed Rate affects everything from credit card rates, home loans, business loans, to how much interest you earn on your savings. Understanding and predicting these changes can help individuals and businesses make better financial decisions.</p>', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 3 & 4: Impact of rate changes

col_up, col_down = st.columns(2, gap="large")

with col_up:
    st.subheader("When Rates Go UP")
    UpTrend_Animation()
    st.markdown("""
    - Borrowing becomes expensive (higher loan & credit card rates)
    - Mortgage and car loan payments increase
    - Saving money is more rewarding (higher bank interest)
    - Economy slows down to control inflation
    """)

with col_down:
    st.subheader("When Rates Go DOWN")
    DownTrend_Animation()
    st.markdown("""
    - Borrowing becomes cheaper (lower loan rates)
    - People & businesses spend more
    - Economy speeds up, helping job growth
    - Investment activity increases
    """)

st.divider()

# -----------------------------------------------------------------------------
# Section 5: Why Fed changes rates

st.subheader("Why Does the Fed Change Rates?")
st.markdown("""
- **To fight inflation**: They raise rates to slow down spending
- **To boost the economy**: They cut rates to encourage borrowing and investing
- **To maintain stability**: Balance between growth and price stability
""")

st.divider()

# -----------------------------------------------------------------------------
# Section 6: Speedometer analogy

st.markdown('<h2 style="text-align: center;">Think of it like adjusting the speed of a car: the Fed uses interest rates to speed up or slow down the economy as needed!</h2>', unsafe_allow_html=True)

block1, block2, block3 = st.columns([2, 6, 1])
with block2:
    Speedometer_Animation()

st.divider()

# -----------------------------------------------------------------------------
# Section 7: Project Background

st.subheader("Project Background")
st.markdown("<p class='justified-text'>The US Federal Reserve, commonly known as the Fed, plays a crucial role in shaping the economy by setting interest rates. These rates influence everything from borrowing costs to inflation and economic growth. Historically, changes in the Fed's interest rate policies have had widespread effects on businesses, consumers, and financial markets. Given its impact, predicting these rate changes has been a major area of interest for economists, investors, and policymakers. Traditionally, financial experts relied on economic indicators, historical trends, and expert opinions to anticipate rate decisions. However, with the growing availability of data and advancements in computing, new approaches have emerged. Machine learning, a branch of artificial intelligence, has opened up possibilities to analyze complex economic patterns and predict rate changes more accurately.</p>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section 8: Economic Foundation

st.subheader("The Economic Foundation")
st.markdown("<p class='justified-text'>The foundation of interest rate decisions lies in macroeconomic factors such as inflation, employment, GDP growth, and global financial conditions. The Federal Reserve assesses these indicators to determine whether the economy needs higher interest rates to control inflation or lower rates to stimulate growth. Traditional financial models use statistical techniques to analyze past trends and make predictions. However, these models often struggle to adapt to rapidly changing economic conditions. Machine learning, on the other hand, can process vast amounts of real-time data, identify hidden patterns, and improve the accuracy of rate predictions.</p>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section 9: Importance

st.subheader("Why Accurate Predictions Matter")
st.markdown("<p class='justified-text'>Interest rate predictions are valuable not only for policymakers but also for businesses, investors, and individuals. Changes in Fed rates influence loan interest rates, mortgage payments, credit card costs, and overall investment returns. A well-informed prediction can help businesses plan their financial strategies, assist investors in making informed decisions, and guide policymakers in shaping effective economic policies. Machine learning models analyze past decisions, economic reports, and market behaviors to recognize potential patterns.</p>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section 10: ML Approach

st.subheader("The Machine Learning Approach")
st.markdown("<p class='justified-text'>Predicting US Federal Reserve rate changes is a challenge that has significant implications for financial planning and economic stability. A more accurate forecasting approach can help businesses mitigate risks, enable investors to make better decisions, and assist policymakers in responding proactively to economic shifts. The rise of machine learning presents an opportunity to refine traditional forecasting methods by leveraging data-driven insights. This project explores how artificial intelligence can enhance financial forecasting and provide valuable insights into one of the most influential economic decisions worldwide.</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 11: Global Impact

st.markdown("<p class='justified-text'>As the global economy becomes increasingly interconnected, the ripple effects of Fed rate changes are felt not only within the United States but across international markets as well. A single rate hike or cut can impact foreign exchange rates, international trade, and global investment flows. This makes understanding and anticipating Fed decisions even more essential for stakeholders around the world. The ability to forecast these changes with greater confidence is no longer just a financial exercise—it's a strategic tool that can influence economic stability and guide informed decision-making on a much broader scale.</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 12: Research Questions

st.subheader("Research Questions Explored")
st.markdown("""
1. **Can we predict future Federal Reserve interest rate decisions using economic data?**
2. **Which economic indicators are most influential in predicting Fed rate changes?**
3. **How effective are machine learning models in forecasting Fed rate movements?**
4. **What does clustering analysis reveal about economic conditions related to interest rates?**
5. **Can association rule mining uncover hidden relationships between economic factors?**
6. **How does PCA help in understanding and simplifying economic data?**
7. **What accuracy can we achieve in predicting Fed rate changes?**
8. **How does model performance vary across different ML algorithms?**
9. **Can these predictive models assist policymakers and investors?**
10. **What are the practical implications of this research for the general public?**
""")

st.divider()

# -----------------------------------------------------------------------------
# Navigation Guide

st.markdown("""
### Explore This Project

Use the sidebar navigation to explore different aspects of this machine learning project:

| Section | Description |
|---------|-------------|
| **Data Preparation** | Data collection from FRED API, cleaning, and preprocessing |
| **PCA** | Dimensionality reduction and variance analysis |
| **Clustering** | K-Means, Hierarchical, and DBSCAN clustering analysis |
| **Association Rules** | Pattern discovery using Apriori algorithm |
| **Naive Bayes** | Probabilistic classification approach |
| **Decision Tree** | Tree-based classification with interpretable rules |
| **Regression** | Linear and logistic regression analysis |
| **SVM** | Support Vector Machine classification |
| **Ensemble Learning** | Random Forest achieving 97% accuracy |
| **Conclusion** | Summary of findings and key insights |
""")

st.divider()
