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

df_NB_data = pd.read_csv(r"App/Tabs/Datasets/NBTrain.csv")

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

st.title("Naive Bayes Alogrithm")
st.header("It's a classifier that's conditionally certain, but fundamentally wrong!")

st.divider()

# -----------------------------------------------------------------------------
# Section 1

st.subheader("Naive Bayes is a simple algorithm that uses probability to make predictions. Here's how it works:")

st.write("""
1. It learns from examples by counting how often things happen together. 
2. When making predictions, it calculates the probability of each possible outcome based on what it's observed before. 
3. It assumes that all features (characteristics) are independent of each other, which makes calculations simpler but isn't always true in real life.
4. It picks the outcome with the highest probability as its answer.
""")

file_path = r"App/Tabs/Animations/Naive_Bayes_Classifier.gif"
st.image(file_path)

st.write("###")

st.subheader("Example:")
st.write('''If you're trying to classify emails as spam or not spam, Naive Bayes would:
- Count how many spam emails contain certain words.
- When seeing a new email, calculate the probability it's spam based on its words.
- Choose "spam" or "not spam" based on which is more probable.
         
**The "naive" part comes from that independence assumption, which is often incorrect but works surprisingly well in practice.**
''')

st.divider()

# -----------------------------------------------------------------------------
# Section 2

st.subheader("So what does Naive Bayes does in general?")
st.markdown('''<p class='justified-text'>Naive Bayes is a classification algorithm that uses probability to predict which category something belongs to. It works based on Bayes' theorem, which calculates the probability of an event based on prior knowledge. The algorithm makes the "naive" assumption that all features (characteristics) are independent of each other, even though they often aren't in real life. It learns by calculating how frequently different features appear in each category in your training data, then makes predictions by finding which category has the highest probability given the features present. It's especially useful for text classification (like spam filtering, sentiment analysis, and document categorization), simple diagnostic systems, and recommendation systems. The main advantages of Naive Bayes are that it's fast to train and make predictions, works well with small amounts of training data, is simple to understand and implement, and is effective for many real-world problems despite its "naive" assumption. It essentially asks: "Based on what I've seen before, what's the most likely category for this new example?"</p>''', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 3

st.subheader("It's Use Cases:")

st.markdown('''<p class='justified-text'>Naive Bayes is primarily used for classification tasks where speed and simplicity are priorities. It excels in text classification problems such as spam filtering, sentiment analysis, document categorization, and language detection because text data naturally fits its bag-of-words approach. It's also popular for recommendation systems, medical diagnosis (as a preliminary screening tool), and real-time prediction scenarios where computational efficiency matters.</p>''', unsafe_allow_html=True)

st.markdown('''<p class='justified-text'>Organizations choose Naive Bayes when they need quick results with limited computational resources, have high-dimensional data, or possess relatively small training datasets. Despite its "naive" independence assumption, it often performs surprisingly well in practice, especially when the independence assumption is not severely violated or when the prediction depends mainly on strong individual features rather than subtle feature interactions.</p>''', unsafe_allow_html=True)

st.markdown('''<p class='justified-text'>The algorithm's greatest strengths are its computational efficiency (both in training and prediction), scalability to large datasets, interpretability of results, and robustness to irrelevant features. These qualities make it particularly valuable as a baseline classifier and in applications where transparency and speed are more important than achieving the absolute highest accuracy.</p>''', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 4

st.subheader("There are multiple forms of Naive Bayes Algorithms, some of them are listed below:")

st.write("#### 1. Multinomial Naive Bayes (MultinomialNB):")

st.markdown('''<p class='justified-text'>Multinomial Naive Bayes (MultinomialNB) is optimized for discrete count data, making it ideal for text classification tasks where features represent word frequencies or TF-IDF scores. This variant assumes features follow a multinomial distribution and performs well when working with document categorization or spam detection using bag-of-words representations. MultinomialNB calculates the probability of a document belonging to a class based on the occurrence count of each word, considering the word frequencies. It's particularly effective with large vocabularies and sparse datasets common in natural language processing. The implementation in scikit-learn also includes smoothing parameters to handle zero probabilities, commonly using Laplace or Lidstone smoothing to prevent zero probability issues.</p>''', unsafe_allow_html=True)

file_path = r"App/Tabs/Animations/multimodal_Naive_Bayes.gif"
st.image(file_path)

st.write("###")

st.write("#### 2. Gaussian Naive Bayes (GaussianNB):")

st.markdown('''<p class='justified-text'>Gaussian Naive Bayes (GaussianNB) is designed for continuous data that follows a normal distribution, modeling each feature with a Gaussian probability density function. Unlike other variants, it models features using means and variances, making it suitable for classification problems with continuous measurements like medical diagnostics where features might include blood pressure, temperature, or other quantitative measurements. Its ability to handle continuous variables makes it versatile, though it assumes normally distributed features. GaussianNB stores the mean and variance of each feature for each class during training, then uses these parameters to calculate conditional probabilities when predicting new samples. This variant tends to perform well when working with smaller datasets of continuous variables and doesn't require discretization of features. It's useful for mixed datasets of continuous and categorical features (after appropriate encoding) and often serves as a good baseline classifier.</p>''', unsafe_allow_html=True)

file_path = r"App/Tabs/Animations/gaussian_nb_animation.gif"
st.image(file_path)

st.write("###")

st.write("#### 3. Bernoulli Naive Bayes (BernoulliNB):")

st.markdown('''<p class='justified-text'>Bernoulli Naive Bayes (BernoulliNB) specializes in binary feature representations, focusing only on whether a feature is present (1) or absent (0), not its frequency or intensity. This makes it particularly useful for text classification tasks where you're only concerned with word occurrence rather than count. It uses Bernoulli distributions for modeling and often outperforms Multinomial NB in cases where feature presence/absence is more informative than frequency. BernoulliNB calculates the probability of a document belonging to a class based solely on whether words appear in the document, ignoring their frequency. This variant is particularly effective for short documents or when working with extremely sparse feature matrices. The scikit-learn implementation also includes a binarize parameter to convert non-binary features to binary representations using a threshold, allowing more flexibility in feature preprocessing and representation.</p>''', unsafe_allow_html=True)

file_path = r"App/Tabs/Animations/bernoulli_nb_animation.gif"
st.image(file_path)

st.write("###")

st.write("#### 4. Categorical Naive Bayes (CategoricalNB):")

st.markdown('''<p class='justified-text'>Categorical Naive Bayes (CategoricalNB) addresses classification tasks with categorical variables that have multiple possible values, not just binary outcomes like Bernoulli NB handles. Unlike Bernoulli NB, it can handle features with several discrete categories, making it appropriate for data where features are nominal categories like colors, sizes, or types. This variant is relatively new to scikit-learn and fills an important gap for handling truly categorical data without forcing binary encoding. CategoricalNB expects features to be encoded as integers representing different categories and models each feature using a categorical distribution rather than Bernoulli or Gaussian distributions. It's particularly useful for datasets with nominal categorical attributes that don't have a natural ordering relationship. The implementation includes alpha parameter for smoothing, similar to other Naive Bayes variants, to handle instances of categories not seen during training.</p>''', unsafe_allow_html=True)

file_path = r"App/Tabs/Animations/categorical_nb_animation.gif"
st.image(file_path)

st.divider()

# -----------------------------------------------------------------------------
# Section 5

st.header("So, when to use what?")

st.markdown('''<p class='justified-text'>Choosing the right variant depends primarily on understanding your data's nature and distribution assumptions, as each implementation is optimized for specific types of features. For text classification using word counts, MultinomialNB is typically preferred due to its ability to leverage frequency information in documents. For continuous measurements that roughly follow normal distributions, GaussianNB works best as it directly models the probability density without requiring discretization. When dealing with binary features indicating presence/absence, BernoulliNB is the appropriate choice, especially for short texts or sparse binary features. For true categorical features with multiple possible values, CategoricalNB should be used to properly model the categorical distribution of each feature. In practice, comparing performance through cross-validation often provides the clearest indication of which variant will work best for your specific dataset. The computational efficiency of all Naive Bayes variants makes them excellent candidates for initial baseline models or for problems with limited computational resources.</p>''', unsafe_allow_html=True)

st.subheader("Key Differences and Comparison - Naive Bayes Variants")

data = {
    "Variant": ["MultinomialNB", "GaussianNB", "BernoulliNB", "CategoricalNB"],
    "Data Type": ["Discrete counts", "Continuous", "Binary", "Categorical"],
    "Distribution": ["Multinomial", "Gaussian/Normal", "Bernoulli", "Categorical"],
    "Feature Representation": [
        "Word counts, frequencies",
        "Real-valued measurements",
        "Presence (1) or absence (0)",
        "Integer-encoded categories"
    ],
    "Typical Applications": [
        "Text classification, document categorization",
        "Medical diagnosis, classification with continuous features",
        "Text classification with binary features, simple feature presence",
        "Classification with nominal categorical variables"
    ]
}

df2 = pd.DataFrame(data)
st.dataframe(df2, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 6

st.subheader("Let's try implementing different types of Naive Bayes Algorithms that suits our dataset.")

st.write("Here is a glims of the dataset that we will be using for Naive Bayes:")
st.write(df.head(10))

st.markdown("Click here to download cleaned dataset [link](%s)" % "https://github.com/Sangram-More/ML-Project/blob/master/Jupyter%20Lab%20Analysis/Cleaned/finaldataset.csv")


st.divider()

# -----------------------------------------------------------------------------
# Section 7

st.subheader("After data preperation, here is how the Training and Testing dataset looks:")

st.write("Training set:")
st.write(df_NB_data.head(10))

st.write("Testing set:")
st.write(df_NB_data.tail(10))

st.markdown('''<p class='justified-text'>In supervised learning, including Naive Bayes classification, it is essential to keep the training and testing sets disjoint to ensure an unbiased evaluation of model performance. This principle is critical in our analysis of the uploaded dataset, where the target variable FEDRates was discretized into categories such as Low, Medium, and High. The training set is used to compute the conditional probabilities of these categories given the input features (like inflation, GDP, consumer income, etc.). If we were to include the same samples in the test set, the model could simply memorize these examples and produce deceptively high accuracy — a phenomenon known as data leakage</p>''', unsafe_allow_html=True)

st.markdown('''<p class='justified-text'>For example, if a row in the dataset where inflation is high and GDP is low appears in both training and testing sets, and the model has already seen that this combination often leads to a "High" FEDRates label, it will predict correctly not because it has generalized, but because it has memorized the answer. This defeats the purpose of testing. By keeping the sets disjoint, we ensure that our Naive Bayes classifier must rely solely on learned probability distributions, not repetition of known labels. This approach mimics real-world deployment, where the model must make predictions on unseen data.</p>''', unsafe_allow_html=True)

st.markdown('''<p class='justified-text'>Moreover, metrics such as accuracy, precision, and recall become meaningful only when evaluated on data the model hasn’t been trained on. In our case, when we split the dataset 80/20 for training and testing, the classifier had to infer FEDRates from new examples, providing a genuine reflection of its predictive power. Hence, the disjoint nature of the training and test sets is not just a best practice — it is a foundational principle for building trustworthy machine learning models.</p>''', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 7

st.subheader("Results:")

st.write("#### Model Evaluation Summary:")

st.markdown('''<p class='justified-text'>Based on the results from applying three different variants of the Naive Bayes algorithm—MultinomialNB, BernoulliNB, and CategoricalNB—on the FEDRates dataset, we observe significant variations in performance across models. CategoricalNB clearly stands out, achieving the highest accuracy of 70%, along with strong precision (0.69), recall (0.69), and F1-score (0.68). It consistently classifies all three classes (Low, Medium, High) more effectively, especially excelling in the High class prediction.</p>''', unsafe_allow_html=True)

st.markdown('''<p class='justified-text'>In comparison, BernoulliNB yields a moderate performance with an accuracy of around 58%. It shows decent capability in classifying the High class (recall of 0.93), but struggles more with the Medium class, indicated by a low recall of 0.33 and an F1-score of 0.41. This suggests that binary feature assumptions may not be as suitable for the FEDRates dataset, which likely contains more nuanced categorical patterns.</p>''', unsafe_allow_html=True)

st.markdown('''<p class='justified-text'>On the other hand, MultinomialNB performs the weakest, with an accuracy of 54% and a macro F1-score of only 0.47. The recall for the Medium class is particularly low at just 0.09, indicating that the model frequently misclassifies those instances. This could be attributed to the model's assumption of discrete count features, which may not align well with the nature of FEDRates data.

The confusion matrices further reinforce these insights. MultinomialNB misclassifies Medium and High classes more often, while CategoricalNB shows better-balanced predictions across all categories. Performance visualizations reveal that CategoricalNB leads in all macro-averaged metrics—precision, recall, and F1-score—making it the most reliable choice for this dataset.

In conclusion, for datasets like FEDRates, which appear to contain integer-encoded categorical features, CategoricalNB is best suited, thanks to its modeling assumptions. BernoulliNB can be a fallback but lacks consistency. MultinomialNB is least effective here due to its incompatibility with the data's underlying structure.</p>''', unsafe_allow_html=True)

st.write("#### Naive Bayes: Multimodal")
file_path = r"App/Tabs/Images/NB_multimodal_results.png"
st.image(file_path)

st.write("#### Naive Bayes: Bernoulli")
file_path = r"App/Tabs/Images/NB_bernoulli.png"
st.image(file_path)

st.write("#### Naive Bayes: Categorical")
file_path = r"App/Tabs/Images/NB_categorical.png"
st.image(file_path)

st.write("#### Naive Bayes: Matrices Comparision")
file_path = r"App/Tabs/Images/NaiveBayesResults.png"
st.image(file_path)

st.write("#### Naive Bayes: Confusion Matrices")
file_path = r"App/Tabs/Images/NaiveBayesCF.png"
st.image(file_path)

st.divider()

# -----------------------------------------------------------------------------
# Section 8

st.subheader("Conclusion:")

st.markdown('''<p class='justified-text'>From this analysis using Naive Bayes algorithms on the FEDRates dataset, I learned that model choice strongly depends on the nature of the input features. The superior performance of the Categorical Naive Bayes model shows that this dataset is best represented using discrete categorical features, where relationships among variables are better captured with integer-encoded categories. This insight emphasizes the importance of data preprocessing and correct algorithm alignment for supervised learning.

Furthermore, I learned that not all Naive Bayes models behave equally well across all types of features. While BernoulliNB handled the High FEDRates class relatively well, it struggled with the Medium class due to its binary assumption. MultinomialNB underperformed overall, likely because it assumes count data, which may not reflect the structure of economic indicators in this dataset.

In terms of prediction, these results suggest that we can use CategoricalNB to build a reasonably accurate classifier that predicts future interest rate environments—Low, Medium, or High FEDRates—based on macroeconomic features. This could be highly beneficial for financial institutions or investors who seek to adjust strategies based on projected interest rate changes.

Overall, this project helped me understand the practical application of Naive Bayes models, and how performance metrics and confusion matrices can guide model selection and optimization for real-world forecasting tasks.</p>''', unsafe_allow_html=True)

st.divider()

st.markdown("Click here checkout Naive Bayes Code [link](%s)" % "https://github.com/Sangram-More/ML-Project/blob/master/Jupyter%20Lab%20Analysis/NaiveBayes.ipynb")