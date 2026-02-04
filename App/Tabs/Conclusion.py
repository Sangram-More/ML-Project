import streamlit as st
import pandas as pd

# --------------- Custom CSS Styling -----------------------------
st.markdown(
    """
    <style>
    .justified-text {
        text-align: justify;
        line-height: 1.8;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .model-comparison {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .key-finding {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    .answer-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Header

st.title("Conclusion & Key Findings")
st.markdown("### Summary of Machine Learning Analysis for Federal Reserve Rate Prediction")

st.divider()

# -----------------------------------------------------------------------------
# Key Results Summary

st.markdown("## Project Results at a Glance")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="result-card">
        <h1 style="margin: 0; font-size: 3em;">97.1%</h1>
        <p style="margin: 0;">Best Model Accuracy<br>(Random Forest)</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="result-card">
        <h1 style="margin: 0; font-size: 3em;">9</h1>
        <p style="margin: 0;">ML Algorithms<br>Implemented</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="result-card">
        <h1 style="margin: 0; font-size: 3em;">3</h1>
        <p style="margin: 0;">Top Features<br>Identified</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Model Performance Comparison

st.markdown("## Model Performance Comparison")

performance_data = {
    "Model": ["Random Forest", "SVM (RBF Kernel)", "Decision Tree", "Logistic Regression", "Naive Bayes", "K-Means Clustering"],
    "Accuracy": ["97.1%", "94.3%", "91.2%", "87.5%", "82.1%", "N/A (Unsupervised)"],
    "Key Strength": [
        "Best overall performance, feature importance",
        "Excellent with high-dimensional data",
        "Highly interpretable rules",
        "Good baseline, probability outputs",
        "Fast training, handles categorical data",
        "Pattern discovery without labels"
    ]
}

df_performance = pd.DataFrame(performance_data)
st.dataframe(df_performance, use_container_width=True, hide_index=True)

st.markdown("""
<div class="key-finding">
<strong>Key Finding:</strong> Random Forest outperformed all other algorithms, achieving 97.1% accuracy with just 50-100 trees.
This demonstrates that ensemble methods are highly effective for financial time series prediction.
</div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Research Questions Answered

st.markdown("## Research Questions Answered")

questions_answers = [
    ("Can we predict future Federal Reserve interest rate decisions using economic data?",
     "Yes, by analyzing key economic indicators such as inflation, GDP, and unemployment rates, we can anticipate potential movements in Federal Reserve interest rate decisions with high accuracy."),

    ("Which economic indicators are most influential in predicting Fed rate changes?",
     "The most influential indicators identified were: (1) Inflation/Consumer Price Index, (2) GDP and Real GDP, and (3) Unemployment Rate. These align with the factors the Federal Reserve publicly considers in policy decisions."),

    ("How effective are machine learning models in forecasting Fed rate movements?",
     "Machine learning models, particularly Random Forest (97.1%) and SVM (94.3%), demonstrated high accuracy in predicting rate changes, significantly outperforming traditional statistical methods."),

    ("What does clustering analysis reveal about economic conditions related to interest rates?",
     "Clustering techniques (K-Means, Hierarchical, DBSCAN) grouped similar economic conditions, revealing distinct economic 'regimes' - high inflation periods, recession periods, and stable growth periods - each associated with different interest rate patterns."),

    ("Can association rule mining uncover hidden relationships between economic factors?",
     "Yes, association rule mining identified patterns such as: high inflation often coincides with higher interest rates, and periods of stable GDP with moderate unemployment tend to have predictable rate movements."),

    ("How does PCA help in understanding economic data?",
     "PCA reduced the 8-dimensional dataset to 3 principal components while retaining 89.78% of variance. The first component alone captured 55.8% of variance, primarily driven by GDP-related features."),

    ("What accuracy was achieved in predicting Fed rate changes?",
     "The Random Forest model achieved approximately 97.1% accuracy, with only 5 misclassifications out of 170 test samples. This indicates strong predictive capability based on the selected economic indicators."),

    ("How does model performance vary with different algorithms?",
     "Performance varied significantly: Random Forest (97.1%) > SVM (94.3%) > Decision Tree (91.2%) > Logistic Regression (87.5%) > Naive Bayes (82.1%). Ensemble methods consistently outperformed single models."),

    ("Can these models assist policymakers and investors?",
     "Absolutely. These models can serve as decision-support tools for policymakers to assess potential economic scenarios and for investors to make informed decisions based on anticipated interest rate movements."),

    ("What are the practical implications for the general public?",
     "Understanding and predicting interest rate changes can help individuals make better financial decisions, such as timing for mortgages, loans, or investments, by anticipating shifts in borrowing costs.")
]

for i, (question, answer) in enumerate(questions_answers, 1):
    with st.expander(f"**{i}. {question}**"):
        st.markdown(f'<div class="answer-box"><p class="justified-text">{answer}</p></div>', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Top Features Identified

st.markdown("## Most Important Economic Indicators")

st.markdown("""
The Random Forest model's feature importance analysis revealed the following ranking:
""")

feature_col1, feature_col2 = st.columns([1, 1])

with feature_col1:
    feature_data = {
        "Rank": ["1", "2", "3", "4", "5"],
        "Feature": ["Inflation (CPI)", "GDP", "Real GDP", "Real GDP Per Capita", "Unemployment Rate"],
        "Importance": ["High", "High", "Medium-High", "Medium", "Medium"]
    }
    st.dataframe(pd.DataFrame(feature_data), use_container_width=True, hide_index=True)

with feature_col2:
    st.image(r"App/Tabs/Images/rf_features.png", caption="Feature Importance from Random Forest Model")

st.divider()

# -----------------------------------------------------------------------------
# Final Results Visualization

st.markdown("## Final Model Results")

st.image(r"App/Tabs/Images/finalresult.png", caption="Comprehensive Model Comparison Results")

st.divider()

# -----------------------------------------------------------------------------
# Project Summary

st.markdown("## Project Summary")

st.markdown('''<p class='justified-text'>This comprehensive machine learning project successfully demonstrated that Federal Reserve interest rate decisions can be predicted with high accuracy using economic indicators. By implementing and comparing nine different machine learning algorithms—from unsupervised clustering to sophisticated ensemble methods—this analysis provided both predictive capabilities and valuable insights into the economic factors driving monetary policy decisions.

The Random Forest model emerged as the clear winner, achieving 97.1% accuracy while also revealing which economic indicators matter most. Inflation (measured by Consumer Price Index) stood out as the strongest predictor, followed by GDP metrics. These findings align with economic theory and the Federal Reserve's stated policy objectives, validating both the approach and the results.

Beyond prediction, this project demonstrated the full machine learning pipeline: data collection via API integration, preprocessing of time series data, exploratory data analysis, dimensionality reduction, clustering analysis, and supervised classification. Each technique contributed unique insights—PCA simplified the data while preserving information, clustering revealed economic regimes, and association rules uncovered hidden patterns.

The practical applications are significant. Financial institutions can use these models to anticipate rate changes and adjust strategies accordingly. Investors can make more informed decisions about interest-rate-sensitive investments. Even individuals can benefit by understanding when borrowing costs might rise or fall, helping with major financial decisions like home purchases or refinancing.</p>''', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Skills Demonstrated

st.markdown("## Technical Skills Demonstrated")

skill_col1, skill_col2, skill_col3 = st.columns(3)

with skill_col1:
    st.markdown("""
    **Data Engineering**
    - API Integration (FRED)
    - Data Cleaning & Preprocessing
    - Time Series Handling
    - Feature Engineering
    - Data Pipeline Development
    """)

with skill_col2:
    st.markdown("""
    **Machine Learning**
    - Supervised Learning
    - Unsupervised Learning
    - Model Selection & Tuning
    - Cross-Validation
    - Ensemble Methods
    - Dimensionality Reduction
    """)

with skill_col3:
    st.markdown("""
    **Tools & Technologies**
    - Python (Pandas, NumPy)
    - Scikit-learn
    - Matplotlib, Seaborn, Plotly
    - Streamlit
    - Jupyter Notebooks
    - Git/GitHub
    """)

st.divider()

# -----------------------------------------------------------------------------
# Call to Action

st.markdown("""
### Explore the Code

All code for this project is available on GitHub:

- **Jupyter Notebooks**: Detailed analysis with step-by-step explanations
- **Streamlit Application**: This interactive web application
- **Datasets**: Cleaned and processed data ready for analysis

[View Project on GitHub](https://github.com/Sangram-More/ML-Project)
""")

st.divider()
