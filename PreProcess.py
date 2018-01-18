import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

#Function to categorize based on median
def categorizeNumpyArray(NArr):
    fq=NArr.describe()["25%"]
    mid=NArr.describe()["50%"]
    tq=NArr.describe()["75%"]
    
    categoryArr=list()
    for i in range(len(NArr)):
        if (NArr[i]<=fq):
            categoryArr.append(0.25)
        elif (fq<NArr[i]<=mid):
            categoryArr.append(0.5)
        elif (mid<NArr[i]<=tq):
            categoryArr.append(0.75)
        elif (tq<NArr[i]):
            categoryArr.append(1.0)
    
    return categoryArr

#Preprocess the data and reduce the dimension
def PreProcess(X):
    X["Median home value"]=categorizeNumpyArray(X["Median home value"])
    X["Median age"]=categorizeNumpyArray(X["Median age"])
    X["Per capita income"]=categorizeNumpyArray(X["Per capita income"])
    X["%Homeless(approx)"]=((X["Total population"]-X["Total households"]*X["Average household size"]).astype(float)/X["Total population"])*100
    X=X.drop(labels=["Total households","Average household size","House hold growth"],axis=1)
    
    X=(X - np.mean(X, axis = 0)) / np.std(X, axis = 0) #Normalizing

    #Principal component analysis, dimension reduction
    pca = PCA(n_components=6)
    principalComponents = pca.fit_transform(X)
    X = pd.DataFrame(data = principalComponents)
    
    return X
