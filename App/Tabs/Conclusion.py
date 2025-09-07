import streamlit as st
from streamlit_lottie import st_lottie
import json
import os
from statistics import correlation
import streamlit as st


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
# st.markdown("<p class='justified-text'></p>", unsafe_allow_html=True)

st.title("Conclusion")
st.header("Lets answer the question that we discussed in introduction tab:")

st.divider()

# -----------------------------------------------------------------------------

st.markdown("### 1. Can we predict future Federal Reserve interest rate decisions using economic data?")
st.markdown("<p class='justified-text'>Yes, by analyzing key economic indicators such as inflation, GDP, and unemployment rates, we can anticipate potential movements in the Federal Reserve's interest rate decisions.</p>", unsafe_allow_html=True)
st.markdown("###")

st.markdown("### 2. Which economic indicators are most influential in predicting Fed rate changes?")
st.markdown("<p class='justified-text'>Indicators like inflation rates, GDP growth, and unemployment rates were found to be significant predictors, aligning with the factors the Federal Reserve considers in its policy decisions.</p>", unsafe_allow_html=True)
st.markdown("###")

st.markdown("### 3. How effective are machine learning models in forecasting Fed rate movements?")
st.markdown("<p class='justified-text'>Machine learning models, particularly Random Forest and Support Vector Machines, demonstrated high accuracy in predicting rate changes, indicating their effectiveness in economic forecasting.</p>", unsafe_allow_html=True)
st.markdown("###")

st.markdown("### 4. What does clustering analysis reveal about economic conditions related to interest rates?")
st.markdown("<p class='justified-text'>Clustering techniques grouped similar economic conditions, highlighting patterns and relationships between different economic indicators and interest rate levels.</p>", unsafe_allow_html=True)
st.markdown("###")

st.markdown("### 5. Can association rule mining uncover hidden relationships between economic factors and interest rates?")
st.markdown("<p class='justified-text'>Yes, association rule mining identified patterns such as high inflation often coinciding with higher interest rates, providing insights into the interplay between various economic factors.</p>", unsafe_allow_html=True)
st.markdown("###")

st.markdown("### 6. How does principal component analysis (PCA) help in understanding economic data?")
st.markdown("<p class='justified-text'>PCA reduced the complexity of the dataset by identifying key components that explain most of the variance, simplifying the analysis without significant loss of information.</p>", unsafe_allow_html=True)
st.markdown("###")

st.markdown("### 7. What was the overall accuracy achieved in predicting Fed rate changes?")
st.markdown("<p class='justified-text'>The Random Forest model achieved an accuracy of approximately 97%, indicating a strong predictive capability based on the selected economic indicators.</p>", unsafe_allow_html=True)
st.markdown("###")

st.markdown("### 8. How does the model's performance vary with different machine learning algorithms?")
st.markdown("<p class='justified-text'>While all tested models provided valuable insights, Random Forest outperformed others in accuracy, followed by Support Vector Machines, showcasing the importance of model selection.</p>",unsafe_allow_html=True)
st.markdown("###")

st.markdown("### 9. Can these predictive models assist policymakers and investors?")
st.markdown("<p class='justified-text'>Absolutely, these models can serve as tools for policymakers to assess potential economic scenarios and for investors to make informed decisions based on anticipated interest rate movements.</p>", unsafe_allow_html=True)
st.markdown("###")

st.markdown("### 10. What are the practical implications of this research for the general public?")
st.markdown("<p class='justified-text'>Understanding and predicting interest rate changes can help individuals make better financial decisions, such as timing for loans or investments, by anticipating shifts in borrowing costs.</p>", unsafe_allow_html=True)
st.markdown("###")

st.divider()

# -----------------------------------------------------------------------------

st.markdown("### Finally but not the least:")

st.markdown('''<p class='justified-text'>After months of exploring numbers, trends, and tools, the heart of this project comes down to one simple question: can we anticipate where Federal Reserve interest rates are headed, just by looking at what’s happening in the economy? The answer is: yes, we can get surprisingly close. By collecting key indicators like inflation, unemployment, and economic growth, and examining how they interact, this project built an intelligent system that can predict future interest rate shifts with impressive accuracy. It’s like putting together clues from the economy to predict what the Fed might do next.

One of the biggest lessons learned is that inflation is one of the loudest signals the economy sends out. When inflation starts rising, there’s a good chance interest rates won’t stay put. But inflation doesn’t act alone — things like GDP (how much the country is producing) and employment numbers also send important cues. What made this project really special was being able to look at all of these signals together and see how they move in sync, like a group of dancers reacting to the same music.

Using smart techniques, like grouping similar economic moments (clustering) and finding patterns (association rule mining), the project uncovered hidden stories in the data. For instance, we saw that when inflation is high and GDP is stable, there’s a consistent tendency for rates to rise. These patterns weren’t just interesting — they were practical. They offer clues to policymakers, investors, and even everyday people trying to make smarter financial decisions. Whether someone’s applying for a mortgage or investing their savings, knowing where interest rates might go is hugely valuable.

One especially exciting moment came when different models were tested to predict future rates. Some models were better than others, but one clearly stood out: Random Forest. Think of it as asking a crowd of people for advice instead of just one — and in this case, that crowd was right over 97% of the time. Not only was it incredibly accurate, but it also helped show which economic indicators were the most important in the decision-making process. In other words, it didn’t just give us answers — it taught us how to ask better questions.
            
More than anything, this project reminded us that the economy, while complex, follows some patterns we can learn from. We don’t need to guess or rely on gut feelings to understand what might happen next. With the right tools, data, and curiosity, we can begin to see the road ahead more clearly. And in a world where even a small change in interest rates can affect everything from student loans to housing markets, having that clarity isn’t just powerful — it’s necessary.

In the end, this research didn’t just build models; it built understanding. It gave us a way to read the signals of the economy and feel just a little less in the dark. And perhaps the biggest takeaway is this: when we listen closely to what the numbers are telling us, they have a lot more to say than we might expect.</p>''', unsafe_allow_html=True)

file_path = r"App/Tabs/Images/rf_features.png"
st.image(file_path)

file_path = r"App/Tabs/Images/finalresult.png"
st.image(file_path)

st.divider()
