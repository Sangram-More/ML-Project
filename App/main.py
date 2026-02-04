import streamlit as st

# ----- Page Configuration ------
st.set_page_config(
    page_title="Fed Rate Predictor | ML Project",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Custom CSS for Professional Look ------
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: #e8e8e8;
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Headers */
    h1, h2, h3 {
        color: #1a1a2e;
    }

    /* Links */
    a {
        color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# ----- Page Setup ------

Introduction_page = st.Page(
    page="Tabs/Introduction.py",
    title="Introduction",
    icon=":material/home:",
    default=True
)

Data_Prep_page = st.Page(
    page="Tabs/Data_Prep.py",
    title="Data Preparation",
    icon=":material/database:"
)

PCA_page = st.Page(
    page="Tabs/PCA.py",
    title="Principal Component Analysis",
    icon=":material/compress:"
)

Clustering_page = st.Page(
    page="Tabs/Clustering.py",
    title="Clustering",
    icon=":material/hub:"
)

ARM_page = st.Page(
    page="Tabs/ARM.py",
    title="Association Rule Mining",
    icon=":material/link:"
)

NaiveBayes_page = st.Page(
    page="Tabs/NaiveBayes.py",
    title="Naive Bayes",
    icon=":material/calculate:"
)

DecisionTree_page = st.Page(
    page="Tabs/DecisionTree.py",
    title="Decision Tree",
    icon=":material/account_tree:"
)

Regression_page = st.Page(
    page="Tabs/Regression.py",
    title="Regression",
    icon=":material/trending_up:"
)

SVM_page = st.Page(
    page="Tabs/SVM.py",
    title="Support Vector Machines",
    icon=":material/linear_scale:"
)

Ensembled_page = st.Page(
    page="Tabs/Ensembled.py",
    title="Ensemble Learning",
    icon=":material/forest:"
)

Conclusion_page = st.Page(
    page="Tabs/Conclusion.py",
    title="Conclusion",
    icon=":material/flag:"
)

# ------ Navigation Menu ----------
pg = st.navigation(
    {
        "Overview": [Introduction_page],
        "ML Techniques": [
            Data_Prep_page,
            PCA_page,
            Clustering_page,
            ARM_page,
            NaiveBayes_page,
            DecisionTree_page,
            Regression_page,
            SVM_page,
            Ensembled_page
        ],
        "Results": [Conclusion_page]
    }
)

# ----- Sidebar Footer ------
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.85em;'>
        <p><strong>Fed Rate Predictor</strong></p>
        <p>Machine Learning Project</p>
        <p>
            <a href="https://github.com/Sangram-More/ML-Project" target="_blank" style="color: #58a6ff;">
                View on GitHub
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ------- Run Navigation ------------
pg.run()