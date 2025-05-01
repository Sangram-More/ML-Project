import streamlit as st

# ----- Page Setup ------

Introduction_page = st.Page(
    page = "Tabs/Introduction.py",
    title = "Introduction",
    icon = ":material/home:",
    default = True
)

Methods_page = st.Page(
    page = "Tabs/Methods.py",
    title = "Machine Learning Methods",
    icon = ""
)

Data_Prep_page = st.Page(
    page = "Tabs/Data_Prep.py",
    title = "Data Preperation",
    icon = ""
)

Conclusion_page = st.Page(
    page = "Tabs/Conclusion.py",
    title = "Conclusion",
    icon = ""
)

PCA_page = st.Page(
    page = "Tabs/PCA.py",
    title = "Principal Component Analysis",
    icon = ""
)

Clustering_page = st.Page(
    page = "Tabs/Clustering.py",
    title = "Clustering",
    icon = ""
)

ARM_page = st.Page(
    page = "Tabs/ARM.py",
    title = "Association Rule Mining",
    icon = ""
)

NaiveBayes_page = st.Page(
    page = "Tabs/NaiveBayes.py",
    title = "Naive Bayes Algorithm",
    icon = ""
)

DecisionTree_page = st.Page(
    page = "Tabs/DecisionTree.py",
    title = "Decision Tree",
    icon = ""
)

Regression_page = st.Page(
    page = "Tabs/Regression.py",
    title = "Regression",
    icon = ""
)

SVM_page = st.Page(
    page = "Tabs/SVM.py",
    title = "SVM",
    icon = ""
)

Ensembled_page = st.Page(
    page = "Tabs/Ensembled.py",
    title = "Ensembled",
    icon = ""
)

# ------ Navigation Menu [Without Sections] ----------
# pg = st.navigation(pages = [Introduction_page, Methods_page, Conclusion_page])

# ------ Navigation Menu [Sections] ----------
pg = st.navigation(
    {
        "About": [Introduction_page],
        "Methodologies": [Data_Prep_page, Methods_page, PCA_page, Clustering_page, ARM_page, NaiveBayes_page, DecisionTree_page, Regression_page, SVM_page, Ensembled_page],
        "Conclusion": [Conclusion_page]
    }
)

# ------- Run Navigation ------------
pg.run()