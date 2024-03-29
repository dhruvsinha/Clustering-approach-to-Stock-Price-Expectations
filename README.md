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
of 6 smaller datasets, each with the dimension $756 N_i$, where $N_i$ represents the number of companies in cluster i (same as the companies in dataset i). We then operate on each cluster i individually, where we iterate over every single company Ci
in that cluster and use $C_i$’s stock price as the target variable y and the stock price for all other companies $C_{j \neq i}$ as the input features X.

## Results

In our exploration we find that the data, at least in 18 dimensions, is not very separable. Nonetheless,
we attempt to visualise our results by plotting the two most populated clusters on a combination of two
dimensions below:

| ![k_means_clustering.jpg](/stock_price/cluster_1.jpg) | 
|:--:| 
||

In Figure 2(a), companies with a higher Receivables Turnover and lower Debt/Equity tend to be in Cluster 1, whereas companies with a higher Debt/Equity and somewhat lower Receivables Turnover generally
appear in Cluster 2. This indicates that Cluster 1 contains companies that are more efficient in getting their
short-term payments from clients and do so without the overhead of long-term debt, indicating that these
companies are financially more stable. More simply, these companies are at least above-average performers in
terms of financial soundness. On the other hand, companies in Cluster 2 are those with greater Debt/Equity,
which makes them more leveraged than the average company.

Likewise, in Figure 2(b), companies classified into Cluster 2 are those with a lower-than-average Return
on Assets, even though their Price-to-Sales ratio is not markedly different from companies in Cluster 1.
This suggests that companies in Cluster 2 are those that did not perhaps make the best available use of
their assets—possibly taking risks that did not pay off, or not innovating enough—when compared to their
industry averages. In other words, these companies are less efficient with their use of capital than their
counterparts in Cluster 1. From an investor’s standpoint, companies in Cluster 1 represent a better avenue
to grow their investment, which is akin to saying that they are better-than-average performers (thereby explaining the sometimes higher Price-to-Sales that the market is willing to pay to buy into these companies’
growth)

The report has detailed analysis on cluster and what each clustier intuitively signifies. 

## Conclusion

Our primary aim for this exercise was to find companies that were similar to a given company, allowing
a potential investor to use those companies as a means of forming and managing expectations about the
company’s future stock price path, particularly for new companies with limited past data. We find that it
is possible to cluster companies into performance-based categories, such that a company’s stock price path
can be semi-reliably predicted using other companies in its cluster. Nonetheless, we do not assert that this
framework necessarily allows an individual to predict the stock price of a company at a particular point in
time. Rather, it allows investors to potentially use a larger (in a temporal sense) dataset of historical stock
price data to chart expectations for a new company, i.e. create a focused confidence interval for its future
price path. Thus, we believe that this research, if effective, can help assuage a key problem in time series
forecasting - overcoming the small-sample problem to reliably manage expectations.
