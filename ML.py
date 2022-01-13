import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from sklearn import metrics
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.linear_model import LinearRegression

############################################
# Section 1 - Importing and Combining Data
############################################

# np.random.seed(12345)

# Getting company names and sector labels
df = pd.read_csv('D:/ML/book1.csv', header=0, index_col='Ticker')
to_keep = ['Name', 'Sector']
dfA = df[to_keep]

# Getting financial ratio data
dfB = pd.read_csv('D:/ML/ratios.csv', header=0, index_col='Ticker')
ratioNames = np.array(dfB.columns.values)

# Concatenating dataframes to get primary dataset
companyData = dfA.join(dfB).drop(['BF-B', 'CTVA', 'FRC'])
companyData = companyData.fillna(0)
clusterData = np.array(companyData)
companies = np.array(companyData.index)

############################################
# Section 2 - Computing Ranked Measures
############################################

# Storing sector-wise means of ratios
dt = companyData.groupby('Sector').mean()

# Function to get industry-relative ratios
def getRelative(ratioData):
    ratios = ratioData[:, 2:]
    sector = ratioData[:, 1]
    
    for i in range(len(sector)):
        # Get sector of company and sector-wise averages of ratios
        ind = sector[i]
        indAvgs = dt.loc[ind]

        for j in range(len(indAvgs)):
            ratios[i, j] = ratios[i, j] / indAvgs[j]
    return ratios

# Storing the relative ratios for future use
finalData = pd.DataFrame(getRelative(clusterData), index=companies, columns=ratioNames).fillna(0)

####################################################
# Section 3 - Identifying Optimal Number of Clusters
###################################################

# Loading the feature dataset
X = np.array(finalData)
comp = clusterData[:, 1]

# Encoding output labels
lab = LabelEncoder()
labels = lab.fit_transform(comp)


# Algorithm to compare cluster sizes (adapted from Scikit-learn's documentation)
def bench_k_means(classifier, name, data):
    
    # Prints labels of measures used
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

    t0 = time()
    classifier.fit(data)

    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), classifier.inertia_,
             metrics.homogeneity_score(labels, classifier.labels_),
             metrics.completeness_score(labels, classifier.labels_),
             metrics.v_measure_score(labels, classifier.labels_),
             metrics.adjusted_rand_score(labels, classifier.labels_),
             metrics.adjusted_mutual_info_score(labels,  classifier.labels_, average_method='arithmetic'),
             metrics.silhouette_score(data, classifier.labels_, metric='euclidean', sample_size=497))
        )
    return classifier.inertia_


# List to store inertia for Elbow Method of cluster size identification
wcss = []

# Comparing multiple values of k (chose to use 4)
for i in range(2, 12):
    print("Calculating measurement scores for Cluster Size {}".format(i))

    cluster = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, 
                     precompute_distances=True, random_state=12345)

    inert = bench_k_means(classifier=cluster, name = "k-means++", data = X)
    print('')
    wcss.append(inert)

# Plotting inertia for different values of k to identify 'elbow'
plt.figure(figsize=(10, 10))
plt.plot(range(2, 12), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 12))
plt.show()


# Function to visualise two most populous clusters on two random axes
def plotClusters(kmeans_out, dimA, dimB):

    (values, counts) = np.unique(kmeans_out, return_counts=True)
    filled = np.stack((values, counts), axis=1)
    sortedFill = filled[filled[:, 1].argsort()]

    # Pick the last two clusters, i.e. most populous
    for i in [-1, -2]:
        cID = sortedFill[i][0]

        if i == -1:
            plt.scatter(X[kmeans_out == cID, dimA], X[kmeans_out == cID, dimB], s=50, c='lightblue',
                        marker='o', edgecolor='black', label='cluster 1')
        
        else:
            plt.scatter(X[kmeans_out == cID, dimA], X[kmeans_out == cID, dimB], s=50, c='lightgreen',
                        marker='s', edgecolor='black', label='cluster 2')

    plt.legend(scatterpoints=1)
    plt.grid()
    plt.xlabel('Dimension A')
    plt.ylabel('Dimension B')
    plt.title('Visual Decomposition of Clustering')
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)
    plt.show()

    return sortedFill[-1][0], sortedFill[-2][0], sortedFill[-3][0], sortedFill[-4][0]
    

# Cluster Size chosen from previous section
size = 7

# Visualising k-means using two random axes
kmeans = KMeans(n_clusters=size, init='random', n_init=20, max_iter=300,
                precompute_distances=True)
kmeans_out = kmeans.fit_predict(X)
cIDA, cIDB, cIDC, cIDD = plotClusters(kmeans_out, 6, 12)

####################################################
# Section 4 - Using PCA to visualise clusters
###################################################

# Fitting K-means to reduced-form data
pca = PCA(n_components=size-1).fit_transform(X)
cluster = KMeans(init='random', n_clusters=size, n_init=20)
pca_out = cluster.fit_predict(pca)
plotClusters(pca_out, 0, 1)

clusterID = pd.DataFrame(kmeans_out, index=companies, columns=['ClusterID'])
xData = pd.concat((finalData, clusterID), axis=1)
# clusterID.to_csv('D:/companylist.csv')

####################################################
# Section 5 - Creating Datasets for Regression
###################################################

stockData = np.array(pd.read_csv('D:/ML/tickerdata.csv', index_col='Date').drop(['BF-B', 'CTVA', 'FRC'], axis=1).fillna(0)).T
clusterStock = np.concatenate((stockData, np.array(clusterID)), axis=1)

# Boolean conditions to split data
condA = [row for row in clusterStock if row[-1] == cIDA]
condB = [row for row in clusterStock if row[-1] == cIDB]
condC = [row for row in clusterStock if row[-1] == cIDC]
condD = [row for row in clusterStock if row[-1] == cIDD]

# Creating separate datasets
regDataA = np.array(condA).reshape((len(condA), len(condA[0])))[:, :-1].T
regDataB = np.array(condB).reshape((len(condB), len(condB[0])))[:, :-1].T
regDataC = np.array(condC).reshape((len(condC), len(condC[0])))[:, :-1].T
regDataD = np.array(condD).reshape((len(condD), len(condC[0])))[:, :-1].T

# Function to apply Linear Regression to every company inside a given cluster
def runRegression(regDataA, regDataB, regDataC, regDataD):
    for numerics in [regDataA, regDataB, regDataC, regDataD]:
        numComp = np.shape(numerics)[1]
        numEx = np.shape(numerics)[0]

        # Array to store weights
        if numComp > 300:
            weightsA = np.ndarray((numComp, numComp-1))
        elif numComp >= 80:
            weightsB = np.ndarray((numComp, numComp-1))
        elif numComp >= 8:
            weightsC = np.ndarray((numComp, numComp-1))
        else:
            weightsD = np.ndarray((numComp, numComp-1))

        # Get features and output values for linear regression
        for i in range(numComp):
            yData = numerics[:, i]
            xData = np.delete(numerics[:], i, axis=1)

            linReg = LinearRegression()
            linReg.fit(xData, yData)

            beta = linReg.coef_

            if numComp >= 300:
                weightsA[i] = beta
            elif numComp >= 30:
                weightsB[i] = beta
            elif numComp >= 8:
                weightsC[i] = beta
            else:
                weightsD[i] = beta

        print("Done for one cluster.")
    return weightsA, weightsB, weightsC, weightsD

# Storing weights as CSV files        
wMatA, wMatB, wMatC, wMatD = runRegression(regDataA, regDataB, regDataC, regDataD)

# Getting list of companies in cluster A and B
indA = xData[xData['ClusterID'] == cIDA].index
indB = xData[xData['ClusterID'] == cIDB].index
indC = xData[xData['ClusterID'] == cIDC].index
indD = xData[xData['ClusterID'] == cIDD].index

pd.DataFrame(wMatA, index=indA).to_csv('D:/ML/clusterA.csv')
pd.DataFrame(wMatB, index=indB).to_csv('D:/ML/clusterB.csv')
pd.DataFrame(wMatC, index=indC).to_csv('D:/ML/clusterC.csv')
pd.DataFrame(wMatD, index=indD).to_csv('D:/ML/clusterD.csv')

pd.DataFrame([cIDA, cIDB, cIDC, cIDD], columns=['ClusterID']).to_csv('D:/ML/popclusters.csv')
####################################################
# Section 6 - Listing similar companies
###################################################

# Loading necessary data
tickers = pd.read_csv('D:/ML/companylist.csv', index_col=0)
clusterA = pd.read_csv('D:/ML/clusterA.csv', index_col=0)
clusterB = pd.read_csv('D:/ML/clusterB.csv', index_col=0)
clusterC = pd.read_csv('D:/ML/clusterC.csv', index_col=0)
clusterD = pd.read_csv('D:/ML/clusterD.csv', index_col=0)
popClusters =pd.read_csv('D:/ML/popclusters.csv', index_col=0)

# Function to return companies in the same cluster + their regression coefficients
def getSimilar(ticker):

    # ClusterID for company
    clID = tickers.loc[ticker]['ClusterID']

    # Separately handling output for different clusterIDs
    if clID == popClusters['ClusterID'][0]:
        
        # List of other companies in cluster
        compList = [tick for tick in clusterA.index if tick != ticker]
        compArray = np.array(compList).reshape((len(compList), 1))

        print("Company\tWeight")
        for i in range(len(compArray)):
            print("{} \t {}".format(compArray[i], clusterA.loc[ticker][i]))

    elif clID == popClusters['ClusterID'][1]:
        
        # List of other companies in cluster
        compList = [tick for tick in clusterB.index if tick != ticker]
        compArray = np.array(compList).reshape((len(compList), 1))

        print("Company\tWeight")
        for i in range(len(compArray)):
            print("{} \t {}".format(compArray[i], clusterB.loc[ticker][i]))
    
    elif clID == popClusters['ClusterID'][2]:

        # List of other companies in cluster
        compList = [tick for tick in clusterC.index if tick != ticker]
        compArray = np.array(compList).reshape((len(compList), 1))

        print("Company\tWeight")
        for i in range(len(compArray)):
            print("{} \t {}".format(compArray[i], clusterC.loc[ticker][i]))
    
    elif clID == popClusters['ClusterID'][3]:

        # List of other companies in cluster
        compList = [tick for tick in clusterD.index if tick != ticker]
        compArray = np.array(compList).reshape((len(compList), 1))

        print("Company\tWeight")
        for i in range(len(compArray)):
            print("{} \t {}".format(compArray[i], clusterD.loc[ticker][i]))

    else:
        print("Sorry, but our database does not have (multiple) companies similar to {}".format(ticker))
