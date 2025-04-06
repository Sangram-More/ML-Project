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

df_DT = pd.read_csv(r"App/Tabs/Datasets/DTData.csv")

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

st.title("Decision Tree")
st.header("It's not the path you take, it's the leaf you end up on!")

st.divider()

# -----------------------------------------------------------------------------
# Section 1

st.header("What are Decision Trees?")

st.markdown('''<p class='justified-text'>Decision trees are intuitive and powerful machine learning algorithms that create a model resembling a flowchart or tree structure to make predictions or decisions. Starting from the root node, they split the data based on features that provide the most information gain, creating branches that lead to subsequent nodes or leaf nodes containing the final decision or prediction. Each internal node represents a "test" on a feature, each branch represents the outcome of that test, and each leaf node represents a class label or a regression value. Decision trees work by recursively partitioning the feature space to create regions where samples are as homogeneous as possible with respect to the target variable. Popular algorithms for building decision trees include ID3, C4.5, CART (Classification and Regression Trees), and newer variants that improve on these foundations.</p>''', unsafe_allow_html=True)

st.markdown('''<p class='justified-text'>Decision trees can be used for both classification and regression tasks across diverse fields. In classification, decision trees predict categorical outcomes such as whether an email is spam, whether a customer will churn, or diagnosing a patient's condition based on symptoms. In regression, they predict continuous values like house prices, temperature forecasts, or estimated sales figures. Decision trees excel in scenarios requiring interpretability, as they provide transparent decision rules that can be easily visualized and explained to stakeholders without technical backgrounds. They're particularly valuable in fields like medicine, finance, and customer relationship management where understanding the reasoning behind predictions is as important as the predictions themselves. Their intuitive nature makes them ideal for initial data exploration and establishing baseline models before moving to more complex algorithms.</p>''', unsafe_allow_html=True)

# Creating 2 columns to add animations side by side.

column1_1, column2_1 = st.columns(2, gap="large", vertical_alignment="center")

with column1_1:
    file_path = r"App/Tabs/Animations/decisiontree1.gif"
    st.image(file_path)

with column2_1:
    file_path = r"App/Tabs/Animations/decisiontree2.gif"
    st.image(file_path)

st.divider()

# -----------------------------------------------------------------------------
# Section 2

st.subheader("Decision Trees and Split Criteria")

st.write("""
Decision trees are versatile machine learning algorithms that create a flowchart-like structure 
for making predictions or decisions. At the heart of decision tree learning is the process of 
selecting the best feature to split the data at each node. This selection relies on metrics 
like **GINI impurity**, **Entropy**, and **Information Gain**.
""")

st.subheader("Split Criteria: GINI vs. Entropy vs. Information Gain")

st.write("#### GINI Impurity")
st.write("""
GINI impurity measures the probability of incorrectly classifying a randomly chosen element 
if it was randomly labeled according to the distribution of labels in the subset. 
Lower GINI values indicate better splits.
""")
st.markdown("""
**For a node with classes i=1 to m:**

- GINI = 1 - ∑(pi²)  
  where **pi** is the proportion of class *i* in the node
""")

st.write("#### Entropy")
st.write("""
Entropy measures the level of disorder or uncertainty in a system. 
Higher entropy indicates more disorder and less predictability.
""")
st.markdown("""
**For a node with classes i=1 to m:**

- Entropy = -∑(pi × log₂(pi))  
where **pi** is the proportion of class *i* in the node
""")

st.write("#### Information Gain")
st.write("""
Information Gain measures how much "information" a feature provides about the class. 
It's calculated as the difference between the entropy before the split and 
the weighted entropy after the split.
""")
st.markdown("""
- **Information Gain = Entropy(parent) - [Weighted Average of Entropy(children)]**
""")

st.subheader("Why These Metrics Matter?")

st.markdown("""
- **GINI Impurity** is computationally efficient and works well for binary classification. 
  It's the default in many implementations (like CART).
- **Entropy** tends to build more balanced trees and can handle multi-class problems well.
- **Information Gain** helps select the attribute that reduces uncertainty the most, 
  leading to more efficient trees.
""")


st.subheader("Example: Measuring Split Quality")

st.write("""
Let's consider a simple dataset about playing tennis based on weather conditions:
""")

# Data for the example
data = {
    "Day": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Overcast", "Sunny", "Rain", "Overcast"],
    "Play Tennis": ["No", "No", "Yes", "Yes", "Yes", "Yes", "No", "No", "Yes"]
}

df = pd.DataFrame(data)
st.table(df)

st.markdown("We want to determine whether **\"Outlook\"** is a good feature for splitting our data.")

# Step 1
st.subheader("Step 1: Calculate Entropy of the entire dataset")
st.markdown("""
- **Total**: 9 examples (5 Yes, 4 No)  
- **P(Yes)** = 5/9, **P(No)** = 4/9  
- **Entropy(S)** = -P(Yes)log₂P(Yes) - P(No)log₂P(No)  
- **Entropy(S)** = -(5/9)log₂(5/9) - (4/9)log₂(4/9) ≈ **0.991 bits**
""")

# Step 2
st.subheader("Step 2: Calculate Entropy after splitting on \"Outlook\"")

st.markdown("""
**For Sunny (3 examples):**  
- **P(Yes|Sunny)** = 0/3, **P(No|Sunny)** = 3/3  
- **Entropy(Sunny)** = -(0/3)log₂(0/3) - (3/3)log₂(3/3) = **0 bits**

**For Overcast (3 examples):**  
- **P(Yes|Overcast)** = 3/3, **P(No|Overcast)** = 0/3  
- **Entropy(Overcast)** = -(3/3)log₂(3/3) - (0/3)log₂(0/3) = **0 bits**

**For Rain (3 examples):**  
- **P(Yes|Rain)** = 2/3, **P(No|Rain)** = 1/3  
- **Entropy(Rain)** = -(2/3)log₂(2/3) - (1/3)log₂(1/3) ≈ **0.918 bits**
""")

# Step 3
st.subheader("Step 3: Calculate weighted average entropy after split")
st.markdown("""
- **WeightedEntropy** = (3/9)×Entropy(Sunny) + (3/9)×Entropy(Overcast) + (3/9)×Entropy(Rain)  
- **WeightedEntropy** = (3/9)×0 + (3/9)×0 + (3/9)×0.918 ≈ **0.306 bits**
""")

# Step 4
st.subheader("Step 4: Calculate Information Gain")
st.markdown("""
- **InformationGain** = Entropy(S) - WeightedEntropy  
- **InformationGain** = 0.991 - 0.306 = **0.685 bits**
""")

st.success("This substantial information gain of 0.685 bits indicates that \"Outlook\" is an excellent feature for splitting our dataset, as it significantly reduces uncertainty about our target variable \"Play Tennis.\"")

st.subheader("GINI Calculation for Feature Selection")

st.subheader("Step 1: Calculate GINI of the entire dataset")
st.markdown("""
- **GINI(S)** = 1 - [(5/9)² + (4/9)²]  
- = 1 - (25/81 + 16/81)  
- = 1 - 41/81  
- ≈ **0.494**
""")

st.subheader("Step 2: Calculate GINI after splitting on \"Outlook\"")
st.markdown("""
Following similar calculations as above, the **weighted GINI after split** would be lower than the original GINI,  
similarly indicating that **"Outlook"** is a good feature for splitting the dataset.

In practice, decision tree algorithms evaluate the gain for each potential feature and select the one  
that provides the **maximum gain** (or **minimum impurity**) for splitting at each node.
""")

st.divider()

# -----------------------------------------------------------------------------
# Section 3

st.subheader("Do you know, there can be INFINITE numbers of Decision Trees!")

st.markdown('''<p class='justified-text'>It is generally possible to create an infinite number of decision trees because of the numerous ways data can be split and structured. Each node in a decision tree offers multiple splitting options based on different features and thresholds, especially when dealing with continuous variables that allow for infinite split points. The sequence in which features are chosen to split the data also contributes to the variety of trees, as changing the order alters the entire structure. Furthermore, decision trees can grow to great depths if not pruned, with splits continuing until each data point is isolated, leading to highly specific and complex trees. Even with the same dataset, ensemble methods like Random Forests introduce randomness through bootstrapping and feature selection, resulting in many unique trees. This means that from a single dataset, countless structurally different trees can be generated. Additionally, noise or slight changes in data can produce new variations. Overall, the flexibility and sensitivity of decision tree construction processes make it theoretically possible to create an infinite number of decision trees.</p>''', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 4

st.subheader("Let's apply Decision Tree on our dataset.")

st.write("Here is the dataset we will be using for Traing and Testing:")

st.write("#### Training set:")
st.write(df_DT.head(10))

st.write("#### Testing set:")
st.write(df_DT.tail(10))

st.markdown("Click here to download cleaned dataset [link](%s)" % "https://github.com/Sangram-More/ML-Project/blob/master/Jupyter%20Lab%20Analysis/Cleaned/finaldataset.csv")

st.markdown('''<p class='justified-text'>In this regression task, our goal is to train a model to predict the FEDRates variable based on other macroeconomic indicators. To evaluate how well the model performs on unseen data, the dataset is divided into two mutually exclusive (disjoint) subsets:</p>''', unsafe_allow_html=True)

st.write('''
- **Training Set (80%):** This set is used by the Decision Tree Regressor to learn patterns in the data. It sees the actual target values (FEDRates) and adjusts the tree's structure accordingly.
- **Testing Set (20%):** This disjoint subset is used only for evaluation, simulating how the model would perform on new, real-world data that it hasn’t seen during training.
''')

st.markdown('''<p class='justified-text'>Disjoint sets are essential to prevent data leakage. If the model sees some of the test examples during training, it might overfit — memorizing rather than generalizing — which results in inflated accuracy and misleading performance metrics.

In our dataset, splitting without overlap ensures that the predictions on X_test are purely based on what the model learned from X_train, offering a realistic measure of model generalizability and true predictive power.</p>''', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 5

st.subheader("Decision Tree with different depth were implemented so as to find the best depth:")

st.write("Decision Trees:")
file_path = r"App/Tabs/Images/DT_multiple.png"
st.image(file_path)

st.write("Confusion Matrix for Decision Trees with different Depths:")
file_path = r"App/Tabs/Images/DT_CF.png"
st.image(file_path)

st.write("Results:")
file_path = r"App/Tabs/Images/DT_different_depths.png"
st.image(file_path)

st.markdown('''<p class='justified-text'>The decision tree models were evaluated using three configurations with varying depth, criterion, and splitter parameters. Tree 1, configured with squared_error and best splitter at depth 4, achieved an R² score of 0.86 and MSE of 2.11. Tree 2, using absolute_error and the same splitter at depth 5, performed best with 0.91 R² and lowest MSE of 1.39, showing the most accurate prediction capability. Tree 3, which used squared_error but with a random splitter at depth 6, achieved an R² of 0.88 and MSE of 1.71, balancing complexity and performance.

All confusion matrices indicate strong performance, especially in identifying Low and High FEDRates categories. Tree 1 and Tree 2 had perfect classification for the "Low" category. Tree 2 did slightly better with fewer misclassifications for the "High" category. Tree 3 showed a bit more misclassification in the "Medium" range, which is often the most ambiguous.

In summary, Tree 2 is the best model, balancing error and generalization. The use of different splitting strategies allowed us to explore how root selection and impurity measures affect decision-making. This analysis reinforces how tuning hyperparameters helps create customized and accurate decision tree models, even on economic datasets like FEDRates.</p>''', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 6

st.subheader("Here are differnet decision trees with different root nodes:")

st.write("#### Root Node: Inflation Consumer Price")
column1_1, column2_1 = st.columns(2, gap="large", vertical_alignment="center")

with column1_1:
    file_path = r"App/Tabs/Images/DT_rootICP.png"
    st.image(file_path)

with column2_1:
    file_path = r"App/Tabs/Images/DT_CF_ICP.png"
    st.image(file_path)

st.write("#### Root Node: Unemployeement Rate")
column1_1, column2_1 = st.columns(2, gap="large", vertical_alignment="center")

with column1_1:
    file_path = r"App/Tabs/Images/DT_rootUR.png"
    st.image(file_path)

with column2_1:
    file_path = r"App/Tabs/Images/DT_CF_UR.png"
    st.image(file_path)

st.write("#### Root Node: Median Consumer Price Index")
column1_1, column2_1 = st.columns(2, gap="large", vertical_alignment="center")

with column1_1:
    file_path = r"App/Tabs/Images/DT_rootMCPI.png"
    st.image(file_path)

with column2_1:
    file_path = r"App/Tabs/Images/DT_CF_MCPI.png"
    st.image(file_path)

st.divider()

# -----------------------------------------------------------------------------
# Section 6

st.subheader("Results:")

st.write("Accuracy Table:")
file_path = r"App/Tabs/Images/DT_Accuracy.png"
st.image(file_path)

st.markdown('''<p class='justified-text'>The visualizations and results from our decision tree models provide a thorough comparison of model performance across different configurations and root feature selections. Starting with the R² score table, we observed that Tree 1 (root: InflationConsumerPrice) and Tree 3 (root: MedianConsumerPriceIndex) performed quite similarly with R² scores of 0.791 and 0.784, respectively. Tree 2 (root: UnemployemenrRate), however, had a noticeably lower R² of 0.659, suggesting that this feature is less predictive of the target variable, FEDRates.

The confusion matrices for each tree based on binned FEDRates into Low, Medium, and High categories give additional insight. Tree 1 and Tree 3 were able to classify categories with relatively balanced performance, though minor misclassifications between Medium and High were visible. Tree 2, however, had more errors, particularly in misclassifying High FEDRates as Medium, aligning with its lower R² score.

The structure of the decision trees themselves further supports these findings. Tree 1 and Tree 3 had clearer and more balanced splits, indicating better feature separability. In contrast, Tree 2 displayed more depth and complexity but less effective partitioning, hinting at possible overfitting without gaining predictive strength.

The second set of trees, where we varied decision tree parameters (depth, criterion, and splitter), highlighted that Tree 2 with absolute error and a depth of 5 achieved the highest R² score of 0.91 and the lowest MSE of 1.39. This suggests that optimizing splitting strategies and depth significantly enhances performance. Tree 3, with a random splitter, achieved decent performance but not as strong as Tree 2.

Finally, the composite comparison plot showcasing confusion matrices and tree diagrams confirms that InflationConsumerPrice and MedianConsumerPriceIndex are stronger features when used as the root. These features lead to better generalization and predictive accuracy, as seen both in visual classification matrices and regression-based metrics.

In summary, root feature choice, splitting criterion, and depth have notable impacts on decision tree model performance. Trees built on InflationConsumerPrice and MedianConsumerPriceIndex demonstrated better predictive power for FEDRates, confirming their importance as strong indicators in economic modeling.</p>''', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 7

st.subheader("Conclusion:")

st.markdown('''<p class='justified-text'>From this analysis, we learned that certain economic indicators like InflationConsumerPrice and MedianConsumerPriceIndex are strong predictors of FEDRates, as shown by higher R² scores and clearer decision boundaries. The decision trees built on these features consistently outperformed those using UnemployemenrRate as the root. Additionally, tuning parameters like the splitting criterion and tree depth significantly impacted model accuracy and interpretability. The confusion matrices further revealed that trees using strong root features made fewer classification errors. This insight helps prioritize variables that are more influential in forecasting interest rate movements. Ultimately, decision trees offer a transparent and effective way to understand and predict economic trends using interpretable logic-based structures.</p>''', unsafe_allow_html=True)

st.divider()