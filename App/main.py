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

# ------ Navigation Menu [Without Sections] ----------
# pg = st.navigation(pages = [Introduction_page, Methods_page, Conclusion_page])

# ------ Navigation Menu [Sections] ----------
pg = st.navigation(
    {
        "About": [Introduction_page],
        "Methodologies": [Data_Prep_page, Methods_page],
        "Conclusion": [Conclusion_page]
    }
)

# ------- Run Navigation ------------
pg.run()