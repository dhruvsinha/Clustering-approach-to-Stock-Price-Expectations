# Clustering approach to stock price expectations

## Abstract
Creating reasonable and reliable expectations of the future, when faced with micronumerosity, is considered
to be the pinnacle of financial forecasting . To obtain reliable forecasts, we employ an unsupervised machine
learning algorithm to cluster companies listed in the Standard & Poor’s (S&P) index. We use 18 different
financial attributes to perform the clustering and then use these clusters to predict individual stock prices
of each company using the stock price of other companies within its cluster as independent variables in a
regression framework. This approach to stock price prediction not only reduces the amount of information
required to perform the task but also reduces the sample required by current techniques to predict stock
prices. This paper, therefore, in addition to extending the sparse literature that exists on the intersection of
machine learning and stock price prediction, also provides a new approach to thinking about the uses of the
former in the financial domain.

## Files Attached

There are three important files attached in this repository. 
1. processing.py: This script was used to collect and process the data
2. ML.py: This script was used to train our KNN and regression models
3. .pdf: This is the final project report

## Methodology

As stated above, we used 18 different financial attributes to perform clustering

### Clustering
We adopted the k-means clustering algorithm for the first stage of this exercise, implemented through
the scikit-learn library in Python. There were a number of key parameters to be identified, all of which are
listed below:
1. n - number of clusters.
2. n init - number of times the algorithm is run with different centroid seeds.
3. algorithm - either naive k-means (lloyd) or triangle inequality-based variation (elkan)

The performance metrics used to compare cluster size (n) are:
1. Homogeneity Score - Proportion of clusters that have companies with a homogeneous Industry label.
2. Completeness Score - Proportion of Industries for which all companies are in the same cluster.
3. V-Measure - Harmonic mean of Homogeneity and Completeness.
4. Inertia - Mean squared distance between each company and its closest centroid.
5. Silhouette Score - It is defined for each instance i as (di,j − dc)/max(di,j , dc), where di,j is the mean
distance to other instances in the nearest cluster and dc is the distance across clusters.


| ![k_means_clustering.jpg](/stock_price/k_means_clustering.jpg) | 
|:--:| 
||

We pick 6 as the cluster size. The justification for this cluster size is explained in the project report. 

### Regression

Once each company is assigned into a cluster, we break up the dataset consisting of stock price data into
smaller chunks. Each chunk contains the data just for companies within a single cluster, resulting in a total
of 6 smaller datasets, each with the dimension $756*N_i$, where $N_i$ represents the number of companies in cluster i (same as the companies in dataset i). We then operate on each cluster i individually, where we iterate over every single company Ci
in that cluster and use $C_i$’s stock price as the target variable y and the stock price for all other companies $C_{j/neqi}$ as the input features X.
