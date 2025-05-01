import streamlit as st
from streamlit_lottie import st_lottie
import json
import os
from statistics import correlation
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --------------- Data Analystics code --------------------------
# df_uncleaned = pd.read_csv(r"App/Tabs/Datasets/Merged_Data.csv")
df = pd.read_csv(r"App/Tabs/Datasets/finaldataset.csv")

df['date'] = pd.to_datetime(df['date'])

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
# Implementing animation function
def animation_file(filepath: str):
    with open(filepath, "rb") as f:
        return json.load(f)

def PCA_Animation():   
    animation = st_lottie(
        animation_file(r"App/Tabs/Animations/PCA.json"),
        speed = 1,
        reverse = False,
        loop = True,
        quality = "high"
    )
    return animation

# -----------------------------------------------------------------------------
# st.markdown("<p class='justified-text'></p>", unsafe_allow_html=True)

st.title("Ensembled Learning")
st.header("What if we combine 2 different ML models to increase the accuracy?")

# -----------------------------------------------------------------------------
# Section 1

st.markdown('''<p class='justified-text'>Ensemble learning is a powerful approach in machine learning that combines predictions from multiple models to achieve better overall performance. The idea is that instead of relying on a single model—which may have its own biases or weaknesses—we can blend several models together to make more accurate and stable predictions. Think of it like asking the opinion of a group of experts instead of just one; the collective insight tends to be more balanced and reliable. Techniques like bagging, boosting, and stacking are commonly used to build these ensemble models, each bringing a slightly different strategy to improve performance.

One of the most popular ensemble methods is Random Forest, which works by building a collection of decision trees, each trained on different subsets of the data. While a single decision tree might be prone to overfitting or misinterpreting noisy data, Random Forest averages out their results, making the final prediction more robust. This method often leads to higher accuracy and better generalization, especially on complex datasets where patterns may not be obvious. In my own project, Random Forest outperformed most other models and reached an impressive accuracy of around 97%, clearly showcasing the strength of ensemble learning.</p>''', unsafe_allow_html=True)

file_path = r"App/Tabs/Animations/ensembled1.gif"
st.image(file_path)

st.markdown('''<p class='justified-text'>What makes ensemble models so appealing is their flexibility and adaptability. Whether you're working with classification or regression problems, or even imbalanced data, ensembles often deliver strong results without requiring excessive fine-tuning. They’re like the safety net of machine learning—adding an extra layer of reliability, especially when individual models struggle.</p>''', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 2

st.markdown("### Lets talk about Random Forest, that we will be using on our dataset:")

st.markdown('''<p class='justified-text'>Random Forest is one of the most powerful and reliable machine learning models I used in my research on predicting Federal Reserve interest rate movements. It works by building multiple decision trees and combining their results, which helps improve accuracy and reduce the risk of overfitting. What I found particularly impressive about Random Forest is how well it handled the complexity of economic data. Even with different trends and patterns in variables like inflation, GDP, and unemployment, the model maintained high performance. In fact, among all the models I tested, Random Forest gave the highest accuracy—reaching around 97%.

Another thing I appreciated was its ability to rank the importance of each feature. This helped me better understand which economic indicators had the most influence on rate decisions. For example, variables like the Consumer Price Index and Real GDP stood out as strong predictors. Random Forest doesn’t require much parameter tuning, which makes it user-friendly while still being highly effective. Overall, it became clear to me that Random Forest strikes a great balance between accuracy, interpretability, and robustness.</p>''', unsafe_allow_html=True)

file_path = r"App/Tabs/Animations/randomfroest.gif"
st.image(file_path)

st.markdown('''<p class='justified-text'>In real-world terms, this model could be useful for financial institutions, analysts, or even everyday investors who want to anticipate changes in interest rates. By feeding it current economic data, we can generate well-informed predictions and prepare accordingly. Whether the goal is to adjust investment strategies or understand policy decisions, Random Forest proved to be a highly valuable tool in this project.</p>''', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 3

st.markdown("### Dataset, we used for implemening random forest model:")
st.markdown("#### Before")
st.write(df.iloc[89:99].head(10))

st.markdown("#### After")
st.write(df.iloc[50:90].head(10))

st.divider()

st.markdown("### Training and testing set:")

file_path = r"App/Tabs/Images/rf_train.png"
st.image(file_path)

file_path = r"App/Tabs/Images/rf_test.png"
st.image(file_path)

st.divider()
# -----------------------------------------------------------------------------
# Section 4

st.markdown("### Steps taken and Results:")

st.markdown('''<p class='justified-text'>In analysis of U.S. Federal Reserve interest rate predictions, the Random Forest model emerged as one of the most powerful and insightful tools. The process began with carefully cleaning and preprocessing the dataset to ensure it was suitable for supervised learning. This included removing null values, normalizing numeric features, and encoding the binary target variable, which labeled interest rates as either "High" or "Low." Using this cleaned dataset, I was able to split the data into training and testing sets and feed it into the Random Forest classifier.

To fine-tune the model, I experimented with various numbers of decision trees (estimators), ranging from 50 to 300. As shown in the line plot, the model performed best with 50 and 100 trees, yielding a remarkable accuracy and F1-score of approximately 97.1%. Interestingly, increasing the number of trees beyond 100 did not significantly improve performance—in fact, accuracy slightly declined. This finding underscores the importance of model tuning, showing that more complexity does not always translate to better results. A smaller, well-optimized forest can often achieve equal or even superior performance compared to larger models.

One of the key advantages of Random Forest is its ability to provide feature importance scores. This allowed me to rank the economic indicators by how much they contributed to accurate predictions. The most influential features turned out to be InflationConsumerPrice, GDP, and RealGDP. These results align well with macroeconomic principles, reinforcing the model’s credibility. Not only did the model perform well, but it also delivered meaningful economic insights, helping to identify which indicators the Fed might consider when adjusting interest rates.

The confusion matrix of the best-performing model further illustrated its accuracy. Out of all predictions, only a handful were misclassified. Specifically, the model correctly identified 81 “Low” rate observations and 84 “High” rate observations, with just five total misclassifications. This high precision and recall indicate that the model generalizes well to unseen data and is highly reliable for decision-making scenarios. It confirms that Random Forest doesn’t just memorize patterns, but actually captures underlying relationships in the data.

Overall, Random Forest proved to be a standout model for this project. It not only delivered top-tier performance in terms of accuracy and F1-score, but also offered interpretability, robustness, and adaptability. This makes it an ideal candidate for policymakers, financial analysts, and economists looking to forecast interest rate decisions with confidence. The model’s ability to highlight the most influential economic variables also opens the door for further economic analysis and potentially even real-time predictive applications.</p>''', unsafe_allow_html=True)

file_path = r"App/Tabs/Images/rf_cr.png"
st.image(file_path)

file_path = r"App/Tabs/Images/rf_cm.png"
st.image(file_path)

file_path = r"App/Tabs/Images/rf_trees.png"
st.image(file_path)

file_path = r"App/Tabs/Images/rf_features.png"
st.image(file_path)

st.divider()
