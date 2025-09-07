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

df_reg = pd.read_csv(r"App/Tabs/Datasets/RegressionData.csv")


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

st.title("Regression")
st.header("Regression: Because guessing isn’t data science.")

st.divider()

# -----------------------------------------------------------------------------
# Section 1

st.subheader("Linear Regression:")

st.markdown("<p class='justified-text'>Linear regression is a fundamental statistical and machine learning technique used to model the relationship between a dependent variable and one or more independent variables. In its simplest form — simple linear regression — it assumes a linear relationship between two variables, fitting a straight line (called the regression line) that best represents the data. The line is described by the equation y = mx + b, where y is the predicted value, x is the input feature, m is the slope (coefficient), and b is the y-intercept. The goal is to minimize the difference between the actual data points and the predicted values, which is done using a method called least squares.</p>", unsafe_allow_html=True)

st.markdown("<p class='justified-text'>Linear regression is widely used due to its simplicity and interpretability. It assumes that there is a linear relationship, the residuals are normally distributed, and there’s no multicollinearity among features in multiple linear regression. It is commonly applied in fields like economics, business, biology, and social sciences to forecast trends and make predictions. Despite being a basic model, linear regression lays the groundwork for understanding more complex algorithms and is a useful baseline for evaluating other models. Its performance can be evaluated using metrics like R² score, Mean Squared Error (MSE), and Mean Absolute Error (MAE).</p>", unsafe_allow_html=True)

file_path = r"App/Tabs/Animations/linerregression.gif"
st.image(file_path)

st.divider()

# -----------------------------------------------------------------------------
# Section 2

st.subheader("Logistic Regression:")

st.markdown('''<p class='justified-text'>Logistic regression is a supervised machine learning algorithm used primarily for binary classification problems, where the outcome variable is categorical and typically has two classes (e.g., Yes/No, 0/1, True/False). Unlike linear regression which predicts continuous values, logistic regression predicts the probability of an instance belonging to a particular class. It uses the sigmoid (logistic) function to map any real-valued number into a value between 0 and 1, which can then be interpreted as a probability. The model is defined by the equation:
P(y=1|x) = 1 / (1 + e^-(β₀ + β₁x)),
where β₀ is the intercept and β₁ is the coefficient for the input feature x.</p>''', unsafe_allow_html=True)

st.markdown("<p class='justified-text'>Logistic regression is widely used because it's fast, efficient, and easy to interpret. It works well when the relationship between the input features and the target is approximately linear, and the output is categorical. It’s often used in applications like spam detection, disease diagnosis, customer churn prediction, and credit scoring. The model’s performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Logistic regression can also be extended to multinomial logistic regression for multiclass problems or regularized versions like L1 (Lasso) and L2 (Ridge) to prevent overfitting. Despite its simplicity, it remains a strong and interpretable baseline for classification tasks.</p>", unsafe_allow_html=True)

file_path = r"App/Tabs/Animations/logisticregression.gif"
st.image(file_path)

st.divider()

# -----------------------------------------------------------------------------
# Section 3

st.subheader("Linear Regression And Logistic Regression: Similarities and Differences")

st.markdown('''<p class='justified-text'>Linear and logistic regression are both fundamental supervised learning algorithms that model the relationship between independent variables and a dependent variable. At their core, both approaches involve calculating weighted combinations of input features to make predictions. They share similar assumptions about the relationship between features and outputs being linear in nature (at least before transformation in logistic regression), and both use a form of the cost function to optimize model parameters via gradient descent or other optimization techniques. Both models can be extended to handle multiple features (multivariate) and use regularization techniques like L1 and L2 to prevent overfitting. They also rely on metrics to evaluate performance and are widely used as baseline models due to their simplicity and interpretability.</p>''', unsafe_allow_html=True)

st.markdown("<p class='justified-text'>Despite these similarities, the key difference lies in the type of output each model produces and the problems they solve. Linear regression is used for predicting continuous numerical values — such as prices, temperatures, or scores — and its output can range from negative to positive infinity. In contrast, logistic regression is used for classification problems, where the output is a probability between 0 and 1, often interpreted as the likelihood of belonging to a certain class. Logistic regression applies the sigmoid function to map predicted values to probabilities, whereas linear regression makes predictions directly from the linear equation. Additionally, their evaluation metrics differ: linear regression typically uses Mean Squared Error or R², while logistic regression uses classification metrics like accuracy, F1-score, and ROC-AUC. Logistic regression also interprets its coefficients in terms of odds ratios, making it more suitable for probability-based interpretations.</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 4

st.subheader("Use of Sigmoid Function in Logistic Regression:")

st.markdown('''<p class='justified-text'>Logistic Regression uses the sigmoid function as a core part of its model. 
Unlike linear regression which outputs a continuous value, logistic regression is designed for classification tasks, 
typically binary classification where the outcome is either 0 or 1. To achieve this, it uses a linear combination of the input features 
(similar to linear regression) and then applies the sigmoid (logistic) function to map the result to a range between 0 and 1.

The sigmoid function is defined as:  
**σ(z) = 1 / (1 + e^(-z))**,  
where *z* is the linear equation output (*β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ*).

This output represents the **probability** that a given input belongs to the positive class. 
If the probability is greater than 0.5, the model typically predicts class 1; otherwise, it predicts class 0. 
The sigmoid curve helps transform raw model scores into interpretable probabilities, making it essential 
for logistic regression to function as a classification model.</p>''', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 5

st.subheader("Maximum Likelihood Function in Logistic Regression")

st.markdown('''<p class='justified-text'>In logistic regression, the model predicts the probability of a binary outcome using the sigmoid function. 
To find the best-fitting parameters (weights), we use a method called Maximum Likelihood Estimation (MLE).  
The idea is to choose model parameters that maximize the probability of observing the actual training labels given the input features.

For each data point, logistic regression outputs a probability between 0 and 1. 
MLE treats each prediction as a Bernoulli trial, and constructs a likelihood function based on these probabilities. 
The likelihood is highest when the predicted probabilities are close to the actual outcomes (0 or 1).  
We usually take the log of the likelihood function (log-likelihood) for easier computation, turning it into a loss function we minimize using gradient descent.

In short, logistic regression doesn’t use squared error loss like linear regression — instead, it optimizes the log-likelihood, 
which is mathematically grounded in probability theory and provides the most probable parameter estimates for classification.
</p>''', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 6

st.subheader("Let's implement Regression model to our dataset")

st.write("Below is the dataset that is prepared to implement Regression Algorithm")
st.write(df_reg.head(10))

st.markdown("Click here to download cleaned dataset [link](%s)" % "https://github.com/Sangram-More/ML-Project/blob/master/App/Tabs/Datasets/RegressionData.csv")

st.markdown('''<p class='justified-text'>As our target variable is a continuous variable, in order to make it complaient for logestic regression, we changed it and fitted it into bins of size 2, (0,1) base on high/low fed rates. We also used PCA to find out 2 components so as to implement logestic regression. Below are the reuslts for the same: </p>''', unsafe_allow_html=True)

st.write("#### Accuracy Matrices:")
file_path = r"App/Tabs/Images/Regression_Results.png"
st.image(file_path)

st.write("#### Confusion Matrix:")
file_path = r"App/Tabs/Images/Regression_CF.png"
st.image(file_path)

st.write("#### Model Fit:")
file_path = r"App/Tabs/Images/regression_boundry.png"
st.image(file_path)

st.divider()

# -----------------------------------------------------------------------------
# Section 7

st.subheader("Results:")

st.markdown('''<p class='justified-text'>The logistic regression model performed well in classifying the binary categories (Low vs High) of the `FEDRates` variable after binarizing the dataset. From the classification report, the model achieved an accuracy of 82.35%, with a precision of 0.87 for class 0 and 0.79 for class 1. The recall values indicate that the model is more effective in predicting the "High" class (0.88) compared to the "Low" class (0.76), showing its strength in identifying positive outcomes. The decision boundary plot from PCA-reduced data illustrates a fairly linear separation between the two classes, confirming that logistic regression was appropriate. The confusion matrix shows most predictions were correct, with only 30 misclassifications out of 170. Overall, this model demonstrates good generalization and interpretability, making it a reliable choice for predicting economic indicators like `FEDRates` when reduced to two outcome categories. </p>''', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 8

st.subheader("Comparing: Naive Bayes Vs Decision Tree Vs Regression:")

st.markdown('''<p class='justified-text'>To effectively compare the performance of Naive Bayes, Decision Trees, and Logistic Regression, we implemented all three models on the same dataset, focusing on predicting the FEDRates variable. Each model type was chosen for its unique advantages. Naive Bayes models were tested with Multinomial, Bernoulli, and Categorical variants, Decision Trees were built with varying root nodes and configurations, and Logistic Regression was applied after binarizing the target variable. The evaluation criteria included accuracy, confusion matrix analysis, classification reports, and R² scores where applicable.

The Naive Bayes models performed decently, with Categorical Naive Bayes achieving the highest accuracy among its variants at 0.70. It handled nominal features well and offered fast computation and simplicity. However, it struggled to match the precision and adaptability of the other models. Logistic Regression, on the other hand, offered a strong classification boundary (as visualized via PCA), delivering 82.35% accuracy, with good precision and recall, especially for the positive class (class 1). It performed well in binary classification and showed a clean separation between the classes.

Decision Trees stood out in regression performance. By using different features as the root node, we compared multiple tree structures. One of the trees (with UnemployemenrRate as root) achieved the best R² score of 0.91, indicating excellent model fit. The visualized trees showed how features contributed to splits, adding interpretability to the model. The confusion matrices for binned outputs also confirmed strong predictive performance across classes, particularly for Trees 1 and 3.

In summary, all models demonstrated value, but Decision Trees emerged as the most powerful and interpretable choice for this dataset due to their flexibility, visual clarity, and high regression accuracy. Logistic Regression is a close second for classification tasks, especially when binary labels are needed. Naive Bayes, while efficient, was less accurate in this setting. Thus, depending on whether the goal is classification or regression, Decision Trees and Logistic Regression would be the preferred models respectively. </p>''', unsafe_allow_html=True)

st.divider()

st.markdown("Click here checkout Regression Code [link](%s)" % "https://github.com/Sangram-More/ML-Project/blob/master/Jupyter%20Lab%20Analysis/Resression.ipynb")