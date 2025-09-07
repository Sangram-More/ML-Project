import streamlit as st
from streamlit_lottie import st_lottie
import json
import os
import pandas as pd

st.title("Association Rule Mining")
st.divider()

# --------------- Data Analystics code --------------------------
# df_uncleaned = pd.read_csv(r"Tabs/Datasets/Merged_Data.csv")
df = pd.read_csv(r"App/Tabs/Datasets/association_rules.csv")
lift = df.sort_values('lift', ascending=False).head(15)
support = df.sort_values('support', ascending=False).head(15)
confidence = df.sort_values('confidence', ascending=False).head(15)

# df['date'] = pd.to_datetime(df['date'])

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

# -----------------------------------------------------------------------------
# Section 1

st.markdown("### Have You ever wondered that why milk, bread and eggs are kept side by side in a grocery store?")

st.markdown("<p class='justified-text'>Association Rule Mining is a technique used in data mining to discover interesting relationships between variables in large datasets. It is especially popular in market basket analysis, where the goal is to uncover which products are frequently bought together by customers. ARM helps businesses understand buying patterns, design better marketing strategies, and optimize product placements. The core idea is to find rules that describe how the presence of some items in a transaction implies the presence of other items. These rules follow the form: If {Item A, Item B} then {Item C}, meaning customers who buy items A and B are likely to also buy item C.</p>", unsafe_allow_html=True)

st.markdown("#### Key Measures: Support, Confidence, and Lift:")

st.markdown("<p class='justified-text'>To evaluate the importance and usefulness of each rule, three key measures are used: Support, Confidence, and Lift.</p>", unsafe_allow_html=True)

st.markdown("""
1. Support measures how frequently a set of items appears together in the dataset. It is the proportion of transactions that contain the itemset compared to all transactions.
2. Confidence measures the reliability of the rule. It calculates how often the consequent (the "then" part of the rule) appears in transactions that contain the antecedent (the "if" part of the rule). Higher confidence means the rule is more likely to be true.
3. Lift compares the confidence of a rule with the expected confidence if the items were independent. Lift greater than 1 indicates a positive association (items occur together more often than expected), lift equal to 1 means no association, and lift less than 1 indicates a negative association (items occur together less than expected).
""")

st.markdown("#### What are Association Rules?")

st.markdown("<p class='justified-text'>In ARM, association rules are statements that capture relationships between items within a dataset. A rule like {Milk, Bread} → {Butter} means that transactions containing milk and bread are likely to also contain butter. These rules are derived after scanning the transactional data for frequently occurring item combinations (frequent itemsets). Rules can vary in strength and usefulness, so the measures of support, confidence, and lift help filter and rank these rules, ensuring that only meaningful and statistically significant rules are selected.</p>", unsafe_allow_html=True)

file_path = r"App/Tabs/Images/ARM_vs_Clustering.png"
st.image(file_path)

st.divider()

st.markdown("#### Apriori Algorithm:")

st.markdown("<p class='justified-text'>The Apriori Algorithm is one of the most widely used algorithms for association rule mining. It works by identifying frequent itemsets — combinations of items that appear together frequently in transactions — and then generating rules from these itemsets. Apriori uses a bottom-up approach, starting with single items and gradually expanding to larger itemsets, provided they meet a minimum support threshold. The algorithm applies the Apriori Property, which states that if an itemset is frequent, then all of its subsets must also be frequent. This property allows Apriori to efficiently prune the search space, focusing only on candidate itemsets that have the potential to be frequent.</p>", unsafe_allow_html=True)

file_path = r"App/Tabs/Images/Apriori_example.jpg"
st.image(file_path)

st.markdown("#### How Apriori Generates Rules:")

st.markdown("<p class='justified-text'>Once frequent itemsets are identified, Apriori generates association rules by splitting each frequent itemset into antecedent (left-hand side) and consequent (right-hand side). For each possible rule, the algorithm calculates confidence and compares it to a predefined threshold. Only rules that exceed this confidence threshold are considered significant. Finally, lift can be calculated to further assess the strength and usefulness of each rule. The result is a list of interpretable rules that highlight important co-occurrence patterns in the data. These rules form the basis for valuable insights in retail, e-commerce, recommendation systems, and many other fields.</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 1

st.markdown("### Here is what we get after applying Apriori Algorithm for Association Rule Mining:")

st.markdown("#### Top 15 Rules for SUPPORT:")
st.write(support)

st.markdown("#### Top 15 Rules for LIFT:")
st.write(lift)

st.markdown("#### Top 15 Rules for CONFIDENCE:")
st.write(confidence)

st.markdown("#### Visual representation of ruled mined:")
file_path = r"App/Tabs/Images/Apriori_dataset.png"
st.image(file_path)

st.divider()

# -----------------------------------------------------------------------------
# Section 2

st.markdown("### Results and Intepretations:")

st.markdown("<p class='justified-text'>The application of Association Rule Mining (ARM) to the given macroeconomic dataset revealed insightful relationships between economic indicators. Using the Apriori algorithm, we were able to discover rules that highlight the co-occurrence of certain economic conditions, such as the relationship between GDP, RealGDP, InflationConsumerPrice, and RealPotentialGDP across various levels (High, Medium, Low). These rules help us understand how different economic factors tend to move together, indicating potential causal or co-influence patterns. For example, when RealGDPPerCapita is Low, GDP is Low, and Inflation is Low, there is a strong association with RealPotentialGDP being Low as well. This aligns with economic intuition, where sluggish economic performance tends to reflect across multiple related indicators.</p>", unsafe_allow_html=True)

st.markdown("<p class='justified-text'>The support, confidence, and lift values provided by the algorithm further allowed us to assess the strength and importance of these rules. High confidence indicated that these relationships are reliable and occur frequently in the dataset, while high lift values revealed that the occurrence of certain economic states makes others significantly more likely. The top rules by support pointed out combinations that were most frequent in the data, which helps in understanding the most common economic situations, while confidence-based rules gave insights into the predictive strength of one set of conditions leading to another. The lift metric allowed us to isolate the rules that have the most significant impact above random chance, helping prioritize the most interesting and meaningful patterns.</p>", unsafe_allow_html=True)

st.markdown("<p class='justified-text'>One notable learning point was how GDP and inflation levels serve as central features in many association rules. This reflects the importance of these indicators as economic health barometers, influencing or being influenced by other macroeconomic variables. Furthermore, the network visualization of the rules emphasized how interconnected these indicators are, forming clusters where certain groups of economic conditions frequently co-occur. This visualization helps to intuitively grasp how multiple economic forces collectively interact rather than viewing them in isolation.</p>", unsafe_allow_html=True)

st.markdown("<p class='justified-text'>Through this analysis, it became clear that economic performance is rarely driven by a single factor, but rather a complex interplay of multiple indicators. By discovering association rules, we gained an empirical understanding of which factors have historically tended to appear together in certain economic climates. This knowledge is particularly valuable for policy makers, economists, and analysts, as it can guide them in diagnosing the early signals of economic downturns or booms. Additionally, by segmenting the rules into those with high, medium, and low levels of economic indicators, we were able to understand the dynamics of both strong and weak economic conditions.</p>", unsafe_allow_html=True)

st.markdown("<p class='justified-text'>Overall, the ARM process highlighted the importance of multi-variable analysis in macroeconomics. It confirmed that relationships between economic indicators are far from linear, with complex dependencies emerging across the dataset. These findings demonstrate how ARM techniques can be powerful exploratory tools to identify hidden structures in economic data, supporting better forecasting models and more informed economic policies. This exercise not only reinforced existing economic theory but also unearthed new potential areas for deeper investigation based on unexpected associations found through data-driven analysis.</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------

st.markdown("Click here to download code for Association Rule Mining [link](%s)" % "https://github.com/Sangram-More/ML-Project/blob/master/Jupyter%20Lab%20Analysis/ARM.ipynb")

# -----------------------------------------------------------------------------