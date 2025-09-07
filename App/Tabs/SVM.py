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
import numpy as np

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

st.title("Support Vector Machine (SVM)")
st.header("So what exactly is Support Vector Machines (SVM)")

# -----------------------------------------------------------------------------
# Section 1

st.markdown('''<p class='justified-text'>Support Vector Machines (SVMs) are a type of machine learning model used to classify things into two groups—for example, deciding if an email is spam or not. They work by drawing a boundary (called a "hyperplane") between two categories in a way that separates them as clearly as possible. Imagine putting a straight line between red and blue dots on a piece of paper—SVMs try to find the best line that keeps all the red dots on one side and all the blue dots on the other. This is why SVMs are called linear separators—they try to divide things with straight lines (or flat surfaces in higher dimensions). The goal is to find the most "confident" separation possible, meaning the line is as far as it can be from the nearest points in each group.</p>''', unsafe_allow_html=True)

st.markdown('''<p class='justified-text'>But not all problems can be solved with a straight line. Sometimes the data is more complicated and tangled. This is where the idea of a "kernel" comes in. A kernel is a clever trick that lets SVMs deal with this by imagining the data in a higher-dimensional space where it can be separated by a straight line. Instead of actually moving the data to this new space (which would be hard to do), the kernel helps the SVM pretend it's already there. This is where the dot product becomes important. The dot product is a way to measure how similar two data points are, and the kernel uses it to calculate relationships between the points in this new, invisible space—without ever having to move the data there.</p>''', unsafe_allow_html=True)

file_path = r"App/Tabs/Animations/SVM1.gif"
st.image(file_path)

file_path = r"App/Tabs/Animations/svm2.gif"
st.image(file_path)

st.markdown("<p class='justified-text'>There are different types of kernels that allow SVMs to work on different types of data. Two popular ones are the polynomial kernel and the RBF (Radial Basis Function) kernel. The polynomial kernel adds curves and bends to the decision boundary, kind of like drawing more flexible lines that can wrap around clusters of points. The RBF kernel goes even further—it allows the SVM to draw soft, circular boundaries around data, making it very powerful for messy or scattered data. These kernels let SVMs adapt to complex situations, helping them make better decisions even when the separation between categories isn't obvious.</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------

st.markdown("""
Support Vector Machines (SVMs) are powerful classifiers that work by finding the best boundary between two classes. They are called **linear separators** because they aim to draw a straight line (or hyperplane) that splits the data. However, not all problems are linearly separable.

This is where **kernels** come in. Kernels allow SVMs to separate data in higher dimensions without explicitly transforming the data. The **dot product** is crucial here—it lets the kernel compute similarities as if the data were mapped to a higher-dimensional space.

We'll now look at the **polynomial kernel**:
\[ K(x, y) = (x^T y + r)^d \]

And show how it maps a 2D point to a higher dimension using a specific example.
""")

# Example points
x1 = np.array([1, 2])
x2 = np.array([3, 4])
r = 1
d = 2

# Define polynomial kernel
def polynomial_kernel(x, y, r=1, d=2):
    return (np.dot(x, y) + r) ** d

# Explicit feature mapping for (r=1, d=2) for 2D to 6D
# φ(x) = [x1^2, sqrt(2)*x1*x2, x2^2, sqrt(2)*x1, sqrt(2)*x2, 1]
def explicit_mapping(x):
    return np.array([
        x[0]**2,
        np.sqrt(2)*x[0]*x[1],
        x[1]**2,
        np.sqrt(2)*x[0],
        np.sqrt(2)*x[1],
        1
    ])

k_val = polynomial_kernel(x1, x2, r=r, d=d)
phi_x1 = explicit_mapping(x1)
phi_x2 = explicit_mapping(x2)
dot_phi = np.dot(phi_x1, phi_x2)

st.subheader("Example Calculation")
st.markdown(f"**x1:** {x1.tolist()}")
st.markdown(f"**x2:** {x2.tolist()}")
st.markdown(f"**Polynomial Kernel Value (K(x1, x2))** = {k_val}")
st.markdown(f"**Dot Product in Transformed Space (φ(x1) • φ(x2))** = {dot_phi}")

st.markdown("This confirms that the kernel function computes the dot product in a higher-dimensional space without explicitly going there!")

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Original 2D space
axs[0].scatter(x1[0], x1[1], color='blue', label='x1')
axs[0].scatter(x2[0], x2[1], color='red', label='x2')
axs[0].set_title("Original 2D Points")
axs[0].set_xlabel("x1")
axs[0].set_ylabel("x2")
axs[0].legend()
axs[0].grid(True)

# Transformed space (first 2 dimensions of φ(x))
axs[1].scatter(phi_x1[0], phi_x1[1], color='blue', label='φ(x1)')
axs[1].scatter(phi_x2[0], phi_x2[1], color='red', label='φ(x2)')
axs[1].set_title("Projected 6D Mapping (2D View)")
axs[1].set_xlabel("Dimension 1")
axs[1].set_ylabel("Dimension 2")
axs[1].legend()
axs[1].grid(True)

st.pyplot(fig)

st.markdown("""
**Conclusion:** The kernel trick helps us use powerful nonlinear classifiers like SVMs without ever having to manually convert data into higher-dimensional space. All the heavy lifting is done by simple dot products inside the kernel function!
""")

st.divider()

# -----------------------------------------------------------------------------

st.markdown("### Data Format:")

st.markdown("<p class='justified-text'>Supervised learning methods require labeled data because they learn by example. Each training instance must include both the input (features) and the correct output (label), so the model knows what to predict. These labels act like answers in a study guide, helping the model understand the relationship between inputs and outputs. Without labels, the model has no feedback to guide its learning process. This is different from unsupervised learning, which finds patterns without labels. Supervised methods are commonly used in tasks like classification (e.g., spam detection) and regression (e.g., predicting house prices). The better and more accurate the labels, the more effective the model becomes.</p>", unsafe_allow_html=True)

st.markdown("### Here is our dataset of Fed Rates with different features that we will be using to perform SVM on.")
st.write(df.iloc[89:99].head(10))

st.markdown("Click here to download cleaned dataset [link](%s)" % "https://github.com/Sangram-More/ML-Project/blob/master/Jupyter%20Lab%20Analysis/Cleaned/finaldataset.csv")

st.divider()

# -----------------------------------------------------------------------------

st.markdown("### Here are some screenshots of how our traing and testing dataset looks like:")

column1_1, column2_1 = st.columns(2, gap="large", vertical_alignment="center")

with column1_1:
    file_path = r"App/Tabs/Images/svmtrain.png"
    st.image(file_path)

with column2_1:
    file_path = r"App/Tabs/Images/svmtest.png"
    st.image(file_path)

st.markdown("<p class='justified-text'>To prepare the dataset for Support Vector Machine (SVM) modeling, I first cleaned and normalized all the key economic indicators, such as GDP, inflation, unemployment rate, and various price indices. I then split the cleaned data into two parts: the training set and the testing set. The training set contains 80% of the data and is used to train the SVM model, while the remaining 20% forms the testing set, which is used to evaluate the model’s performance on unseen data. This split is important because it ensures that the model is not just memorizing the data, but actually learning to generalize. Each record in the training and testing sets is labeled with a binary target called FEDRates_Binary, which allows the model to classify rate movements effectively. Since SVM requires labeled and numeric input, this setup ensures everything is ready for accurate and meaningful predictions.</p>", unsafe_allow_html=True) 

st.markdown('''<p class='justified-text'>T created the training and testing sets by using an 80/20 split of the cleaned dataset. This means 80% of the data was randomly selected to train the model, and the remaining 20% was reserved for testing it. I used the train_test_split() method from scikit-learn, which ensures that both sets are mutually exclusive — meaning the model is tested on data it hasn't seen before. This is crucial to evaluate how well the model generalizes to new, unseen information rather than just memorizing the training data.

The SVM model also requires the dataset to be fully numeric and labeled. That’s why all categorical columns were either excluded or converted to numeric format earlier, and the FEDRates_Binary column was used as the label. SVMs work by drawing a hyperplane between classes in a high-dimensional numeric space, so it cannot process non-numeric inputs or unlabeled data. Ensuring clean, numeric, and disjoint training and testing sets helps build a robust and accurate SVM classifier.</p>''', unsafe_allow_html=True) 

st.divider()

# -----------------------------------------------------------------------------

st.markdown("### Results:")

file_path = r"App/Tabs/Images/svm_cm.png"
st.image(file_path)

file_path = r"App/Tabs/Images/svm_kernal.png"
st.image(file_path)

file_path = r"App/Tabs/Images/svm_costs.png"
st.image(file_path)

st.markdown('''<p class='justified-text'>Your SVM model results show valuable insights across three different kernel types: RBF, Linear, and Polynomial (degree=2). The confusion matrices indicate that the RBF kernel performs best overall, achieving an accuracy of 0.92 with minimal misclassification. It predicts both "Low" and "High" Fed Rate categories with strong precision and balance, reflecting that the RBF kernel effectively captures complex, non-linear relationships in the data. On the other hand, the linear kernel lags behind with an accuracy of 0.81, suggesting that linear separation doesn't fully capture the underlying patterns in the Fed Rates dataset.

When we examine the performance metrics in the grouped bar chart, we see the RBF kernel maintaining high scores across all evaluation criteria: accuracy, precision, recall, and F1-score. This consistency confirms its robustness and suitability for modeling the binary Fed Rate prediction task. The polynomial kernel, although performing better than the linear version, shows a slightly lower precision, especially in distinguishing between closely spaced decision boundaries. This could be due to polynomial models being more sensitive to noise or overfitting when the degree isn't optimal.

Lastly, the Accuracy vs Cost (C) graph reveals how tuning the regularization parameter affects performance. RBF maintains its lead across most cost values, peaking around C=10, before slightly declining. Linear and polynomial kernels also benefit from higher C values, but their performance stabilizes or plateaus. This emphasizes the importance of kernel selection and hyperparameter tuning in achieving optimal performance. Overall, SVM with the RBF kernel appears to be the most reliable choice for this dataset, balancing generalization and accuracy effectively.</p>''', unsafe_allow_html=True) 

st.divider()

# -----------------------------------------------------------------------------

st.markdown("### Conclusion:")

st.markdown('''<p class='justified-text'>I learned that economic indicators such as inflation rates, GDP growth, unemployment rates, and consumer price indices carry strong predictive power for forecasting movements in the Federal Reserve interest rates. By applying a variety of machine learning models—such as Support Vector Machines, Random Forest, Decision Trees, and Naive Bayes—I was able to uncover patterns and relationships that often precede a rate hike or cut. In particular, Random Forest and SVM with RBF kernel consistently delivered high accuracy, indicating their ability to interpret complex, non-linear interactions in economic data.

Using this knowledge, we can make informed predictions about whether the Fed is likely to raise or lower interest rates in the near future. These insights can be valuable for financial analysts, policy makers, investors, and anyone whose decisions are influenced by interest rate changes. For example, if we see rising inflation and falling unemployment in the data, the model might predict an upcoming rate hike. Overall, this research demonstrates the potential of machine learning to enhance economic forecasting and support data-driven decision-making.</p>''', unsafe_allow_html=True) 

st.divider()

st.markdown("Click here to check out the code [link](%s)" % "https://github.com/Sangram-More/ML-Project/blob/master/Jupyter%20Lab%20Analysis/SVM.ipynb")
