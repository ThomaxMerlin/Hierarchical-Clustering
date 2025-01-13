# Hierarchical-Clustering

This Jupyter Notebook provides an analysis of customer segmentation using the Mall_Customers.csv dataset. The goal is to group customers based on their demographic and behavioral attributes, such as age, annual income, and spending score, using hierarchical clustering.

Table of Contents
Introduction

Dataset Overview

Data Preprocessing

Exploratory Data Analysis (EDA)

Hierarchical Clustering

Visualization

Conclusion

1. Introduction
This notebook demonstrates how to perform customer segmentation using hierarchical clustering. The dataset contains information about mall customers, including their CustomerID, Gender, Age, Annual Income (k$), and Spending Score (1-100). The analysis aims to group customers into clusters based on their similarities in income and spending behavior.

2. Dataset Overview
The dataset contains 200 rows and 5 columns:

CustomerID: Unique ID for each customer.

Gender: Gender of the customer (Male/Female).

Age: Age of the customer.

Annual Income (k$): Annual income in thousands of dollars.

Spending Score (1-100): Spending score assigned by the mall (1-100).

3. Data Preprocessing
Handling Outliers: Outliers in the Annual Income (k$) column were identified and removed using a boxplot and the np.where function.

Feature Selection: Only the relevant features (Annual Income (k$) and Spending Score (1-100)) were used for clustering.

4. Exploratory Data Analysis (EDA)
Boxplots: Visualized the distribution of Annual Income (k$), Spending Score (1-100), and Age to identify outliers and understand the data spread.

Scatterplot: Plotted Annual Income (k$) vs. Spending Score (1-100) to observe patterns and relationships between the two variables.

5. Hierarchical Clustering
Dendrogram: A dendrogram was created using the linkage function with the ward method to determine the optimal number of clusters.

Agglomerative Clustering: Applied hierarchical clustering using AgglomerativeClustering from sklearn with n_clusters=5 for 2D clustering and n_clusters=3 for 3D clustering.

6. Visualization
2D Clustering: A scatterplot was used to visualize the clusters based on Annual Income (k$) and Spending Score (1-100).

3D Clustering: A 3D scatterplot was created using plotly.express to visualize clusters based on Age, Annual Income (k$), and Spending Score (1-100).

7. Conclusion
The analysis successfully grouped customers into distinct clusters based on their income and spending behavior.

The 2D clustering revealed 5 clusters, while the 3D clustering with the addition of Age resulted in 3 clusters.

These clusters can help the mall management tailor marketing strategies and improve customer satisfaction.

How to Use This Notebook
Install Required Libraries:
Ensure you have the following Python libraries installed:

bash
Copy
pip install numpy pandas matplotlib seaborn scikit-learn plotly
Download the Dataset:
Place the Mall_Customers.csv file in the same directory as the notebook.

Run the Notebook:
Open the Jupyter Notebook and run each cell sequentially to reproduce the analysis.

Interact with Visualizations:
Use the interactive 3D scatterplot to explore the clusters in detail.

Code Snippets
Import Libraries
python
Copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
Load Dataset
python
Copy
MD = pd.read_csv('Mall_Customers.csv')
MD.head()
Remove Outliers
python
Copy
out = np.where(MD['Annual Income (k$)'] > 130)
MD.drop(out[0], inplace=True)
Dendrogram
python
Copy
plt.figure(figsize=(10, 7))
dendrogram(linkage(X, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customer")
plt.ylabel("Euclidean Distance")
plt.axhline(y=150, color='r', linestyle='--')
plt.show()
2D Clustering
python
Copy
cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean')
cluster.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap='rainbow')
plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()
3D Clustering
python
Copy
fig = px.scatter_3d(
    MD,
    x='Age',
    y='Annual Income (k$)',
    z='Spending Score (1-100)',
    color=cluster.labels_.astype(str),
    opacity=0.8,
    title='Clusters in Age-Annual Income-Spending Score'
)
fig.show()
Future Work
Experiment with other clustering algorithms like K-Means or DBSCAN.

Incorporate additional features like Gender for more nuanced segmentation.

Perform cluster profiling to interpret the characteristics of each group.

Contact
For questions or feedback, please contact [Your Name] at [Your Email].

Enjoy exploring the notebook! 🚀
