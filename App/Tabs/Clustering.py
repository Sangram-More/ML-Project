import streamlit as st
from streamlit_lottie import st_lottie
import json
import os
import pandas as pd

st.title("What do you know about different types of Clustering: Kmeans, Hierarchical, DBSCAN.")
st.divider()

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

# -----------------------------------------------------------------------------
# Section 1

st.markdown("### K-means:")
st.markdown("<p class='justified-text'>K-Means clustering is a way to group data points into different clusters based on how similar they are to each other. First, the algorithm randomly chooses a few center points (called centroids) that represent each cluster. Then, every data point is assigned to the nearest centroid, forming clusters. After that, the centroids are moved to the average position of all the points in their cluster, and the process repeats until the centroids stop moving much. The goal is to make the points within each cluster as close to each other as possible, while keeping clusters far apart. One important thing about K-Means is that you have to decide how many clusters (k) you want beforehand, which can be tricky if you don’t know the data well. K-Means works best when the clusters are round and evenly sized, but it might not work well for complex shapes or data with lots of outliers.</p>", unsafe_allow_html=True)

file_path = r"App/Tabs/Animations/k_means_example.gif"
st.image(file_path)
st.divider()

# -----------------------------------------------------------------------------
# Section 2

st.markdown("### Hierarchical Clustering:")
st.markdown("<p class='justified-text'>Hierarchical clustering is a way to group data points into clusters by building a tree-like structure called a dendrogram. It works either bottom-up (starting with each data point as its own cluster and gradually merging the closest ones) or top-down (starting with all data points in one big cluster and gradually splitting them). One of the best things about hierarchical clustering is that you don’t need to know the number of clusters beforehand—you can decide after looking at the dendrogram. This method is very easy to understand visually, because the dendrogram shows how and when each point or cluster was grouped. However, it works best on small datasets, because the process of comparing all data points to each other can be very slow when the dataset is large. It also struggles with noisy data and is sensitive to outliers, which can distort the tree structure. Overall, hierarchical clustering is a great choice for exploring the natural structure of a small dataset, especially when you want to understand relationships between clusters.</p>", unsafe_allow_html=True)

file_path = r"App/Tabs/Animations/Hierarchical_Clustering_example.gif"
st.image(file_path)
st.divider()

# -----------------------------------------------------------------------------
# Section 3

st.markdown("### DBSCAN:")
st.markdown("<p class='justified-text'>DBSCAN, which stands for Density-Based Spatial Clustering of Applications with Noise, is a popular clustering algorithm that works by grouping together points that are close to each other. Instead of asking you to predefine the number of clusters (like KMeans), DBSCAN looks for dense areas in the data and treats points that are too far away as noise or outliers. It works well when clusters have irregular shapes or when the data has a lot of noise. The two main settings you need to choose are eps, which is the maximum distance to consider points as neighbors, and min_samples, which is the minimum number of points needed to form a dense cluster. One big advantage of DBSCAN is that it can find clusters of all shapes and sizes, unlike some other methods that only work for round or evenly sized clusters. However, choosing good values for eps and min_samples is very important, and if these are not set well, the clustering results can be poor. Overall, DBSCAN is a great choice when you want to detect flexible-shaped clusters and handle noisy data automatically.</p>", unsafe_allow_html=True)

file_path = r"App/Tabs/Animations/DBSCAN_example.gif"
st.image(file_path)
st.divider()

# -----------------------------------------------------------------------------
# Section 4

st.markdown("### Applying K-Means algorithm on Fed Rates Dataset:")

st.markdown("##### Silhouette process to obtain optimal K values:")
file_path = r"App/Tabs/Images/Silhouette_Optimal_k.png"
st.image(file_path)

st.markdown("#")

st.markdown("##### K-Means implementation using optimal values:")

# # Creating 2 columns to add animations side by side.

# column1_1, column2_1, column3_1 = st.columns(3, gap="small", vertical_alignment="center")

# with column1_1:
#     file_path = r"App/Tabs/Images/K_means_k3.png"
#     st.image(file_path)

# with column2_1:
#     file_path = r"App/Tabs/Images/K_means_k5.png"
#     st.image(file_path)

# with column3_1:
#     file_path = r"App/Tabs/Images/K_means_k6.png"
#     st.image(file_path)

file_path = r"App/Tabs/Images/K_means_k3.png"
st.image(file_path)

file_path = r"App/Tabs/Images/K_means_k5.png"
st.image(file_path)

file_path = r"App/Tabs/Images/K_means_k6.png"
st.image(file_path)

st.markdown("#")

st.markdown("<p class='justified-text'>The silhouette method was applied to determine the optimal number of clusters (k) for the dataset after PCA reduction. Based on the silhouette scores, k=5 emerged as the best option, followed by k=3 and k=6. The silhouette score for k=5 is the highest at approximately 0.43, indicating the data points are well-separated and fit appropriately within their respective clusters. The second-best choice is k=3 with a silhouette score around 0.41, and the third is k=6 with a score of approximately 0.40. In the k=5 plot, we see that the data is divided into well-separated clusters, and the centroids are positioned to minimize intra-cluster variance. The k=3 plot, however, shows a more coarse clustering where fewer groupings try to cover larger regions of the data, potentially merging some distinct clusters into one. On the other hand, the k=6 plot divides the data into smaller clusters, but the silhouette score decreases, indicating that some clusters might be too close to each other or poorly separated. This shows the trade-off between granularity and cluster separation — fewer clusters provide general insights, while more clusters capture finer patterns at the cost of overlap or noise. The use of PCA (Principal Component Analysis) prior to clustering helped reduce dimensionality, making the clustering more efficient and interpretable. Overall, k=5 seems to provide the best balance between distinct cluster formation and capturing meaningful patterns within the economic indicators influencing FEDRates. This comprehensive approach using PCA + KMeans + Silhouette Analysis ensures the clustering process is both statistically valid and visually interpretable.</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 5

st.markdown("### Applying Hierarchical Clustering algorithm on Fed Rates Dataset:")

st.markdown("##### Dendrogram of Hierarchical Clustering:")
file_path = r"App/Tabs/Images/Hierarchical_Clustering_dendrogram.png"
st.image(file_path)

st.markdown("<p class='justified-text'>The dendrogram shown is a hierarchical clustering visualization that helps us understand how data points in the given dataset are merged into clusters. Using Ward’s method, it minimizes the variance within each cluster as new points are added. The red horizontal line represents the threshold at which the data is split into exactly three clusters, indicating that the optimal number of clusters (k=3) is selected. Observing the branching, we can see that some clusters merge at very short distances, indicating they are highly similar, while others join at much larger distances, meaning they are more distinct. This suggests that the dataset has a natural structure where some groups of data points are closely related, while others are more distant. Overall, the dendrogram confirms that the data exhibits a hierarchical structure, and selecting k=3 seems reasonable for further analysis using hierarchical clustering.</p>", unsafe_allow_html=True)


st.markdown("##### Hierarchical Clustering (k=3):")
file_path = r"App/Tabs/Images/Hierarchical_Clustering_k3.png"
st.image(file_path)

st.markdown("<p class='justified-text'>The above graph is a visualization of Hierarchical Clustering applied to the dataset after reducing it to two principal components using PCA. The data points are divided into three clusters, as indicated by the color variations. We can observe that the leftmost cluster contains data points that are closely packed, indicating a dense region with strong similarity between points. The middle cluster is more spread out, suggesting that this group contains more variation in data characteristics. The rightmost cluster is even more scattered, which could indicate a broader range of patterns within that group. Hierarchical clustering works well in this case as it identifies these natural groupings based on distance and similarity, but the presence of overlapping points (especially in the central area) hints that some points could potentially belong to either of two neighboring clusters. Overall, this visualization gives a clear view of how the data naturally segments into three distinct groups based on hierarchical relationships.</p>", unsafe_allow_html=True)

st.markdown("#")

# -----------------------------------------------------------------------------
# Section 6

st.markdown("### Applying DBSCAN algorithm on Fed Rates Dataset:")

st.markdown("##### K-Distance Graph:")
file_path = r"App/Tabs/Images/K_distance_graph.png"
st.image(file_path)

st.markdown("##### DBSCAN Clustering:")
file_path = r"App/Tabs/Images/DBSCAN_Clustering.png"
st.image(file_path)

st.markdown("<p class='justified-text'>The DBSCAN clustering method has successfully identified 9 distinct clusters within the dataset, while also marking 90 points as noise, indicating that these points do not belong to any cluster. This demonstrates that DBSCAN is particularly effective at detecting outliers, which is an advantage over KMeans and Hierarchical clustering methods. In the k-distance graph, the elbow point helps determine an optimal eps value, which is a crucial parameter for DBSCAN. The graph shows a distinct upward bend around 0.8, suggesting this is a suitable eps value, which was used in the clustering process. The resulting clusters vary in density, showing DBSCAN's ability to handle irregularly shaped clusters, which is another benefit compared to KMeans, which tends to find spherical clusters. The noise points detected by DBSCAN might represent economic conditions or events that were significantly different from the general patterns, possibly indicating outliers such as financial crises or major economic policy changes.</p>", unsafe_allow_html=True)

st.markdown("<p class='justified-text'>The visual representation of the clusters in the PCA space shows that some clusters are compact, while others are more dispersed, further highlighting DBSCAN's flexibility in handling data with varying densities. This clustering approach is particularly useful for economic data where normal periods and abnormal periods (like recessions) need to be separated. Compared to KMeans, DBSCAN does not require specifying the number of clusters in advance, making it more adaptive when the number of natural groupings is not known beforehand. Additionally, DBSCAN’s ability to handle noise is highly valuable when working with real-world economic data, which often contains anomalies. Compared to Hierarchical clustering, DBSCAN also scales better to larger datasets and provides more meaningful clusters when the data does not form clear hierarchical structures. Overall, DBSCAN's strengths in finding arbitrarily shaped clusters, handling noise, and adapting to the data's inherent structure make it particularly suitable for this type of dataset, where economic factors fluctuate over time and some periods may exhibit distinct behavior.</p>", unsafe_allow_html=True)

st.markdown("#")

st.divider()

# -----------------------------------------------------------------------------
# Section 7

st.markdown("## Conclusion:")

st.markdown("<p class='justified-text'>Through the application of KMeans, Hierarchical Clustering, and DBSCAN on the dataset related to FEDRates and economic indicators, several important insights emerged.</p>", unsafe_allow_html=True)

st.markdown("<p class='justified-text'>KMeans Clustering provided a structured way to group the data into clusters based on similarity, and the Silhouette Score helped identify the optimal number of clusters, which was k=5. This method worked well after PCA reduction, but it had a strong assumption of spherical clusters and equal-sized groups, which may not always align with economic data that exhibits irregular patterns due to recessions, policy changes, and crises.</p>", unsafe_allow_html=True)

st.markdown("<p class='justified-text'>Hierarchical Clustering, on the other hand, provided a dendrogram, which gave a clear visual hierarchy of how clusters merge at different distances. With k=3, this method captured broader economic patterns but lacked flexibility in capturing fine-grained structures within the data. While hierarchical clustering is useful for visual inspection, it is computationally expensive for larger datasets and does not handle noise explicitly.</p>", unsafe_allow_html=True)

st.markdown("<p class='justified-text'>DBSCAN proved to be the most insightful for this dataset. It identified 9 clusters and 90 noise points, showing that economic data contains several outliers or irregular events, such as sharp interest rate hikes or unusual economic downturns. DBSCAN’s ability to detect clusters of different shapes and sizes without pre-specifying k made it particularly useful in this case, where the structure of the economic data is complex and not necessarily evenly distributed.</p>", unsafe_allow_html=True)

st.markdown("<p class='justified-text'>In summary, KMeans offered a balanced clustering based on compactness, Hierarchical Clustering revealed a hierarchical relationship between economic periods, and DBSCAN excelled at detecting anomalies and irregular clusters. These results collectively indicate that economic data is inherently noisy, often non-spherical, and shaped by both gradual trends and abrupt shocks. This reinforces the understanding that clustering economic data requires choosing methods based on the purpose — KMeans for regular trends, Hierarchical for structural relationships, and DBSCAN for identifying unusual or crisis periods.</p>", unsafe_allow_html=True)

st.divider()