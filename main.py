from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

beans = pd.read_excel(r'Dry_Bean_Dataset.xlsx')
trainBeans = beans.sample(frac=0.8, random_state=1)
testBeans = beans.loc[~beans.index.isin(trainBeans.index)]
featureNum = trainBeans.shape[1] - 1
sampleNum = trainBeans.shape[0]

TrainAttribute = pd.DataFrame(trainBeans.drop(columns="Class")).to_numpy()
TrainTarget = pd.DataFrame(trainBeans.drop(columns=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'])).to_numpy()
TestAttribute = pd.DataFrame(testBeans.drop(columns="Class")).to_numpy()
TestTarget = pd.DataFrame(testBeans.drop(columns=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'])).to_numpy()

# we get lower accuracy using this
# NORMALIZATION
# TrainAttribute = (TrainAttribute-TrainAttribute.min())/(TrainAttribute.max()-TrainAttribute.min())
# TestAttribute = (TestAttribute-TestAttribute.min())/(TestAttribute.max()-TestAttribute.min())


TrainTargetEncode = preprocessing.LabelEncoder().fit_transform(y=TrainTarget.ravel())
TestTargetEncode = preprocessing.LabelEncoder().fit_transform(y=TestTarget.ravel())




#   GAUSSIAN NAIVE BAYS
naiveBays = GaussianNB()
naiveBays.fit(TrainAttribute, TrainTargetEncode)

targetPredicted = naiveBays.predict(TestAttribute)
print("Naive Bayes using Gaussian")
print("Accuracy: ", metrics.accuracy_score(TestTargetEncode, targetPredicted))
print("-------------------------------------")

# DECISION TREE
clf = tree.DecisionTreeClassifier()
clf.fit(TrainAttribute, TrainTargetEncode)
fig = tree.plot_tree(clf, filled=True)
DTPredicted = clf.predict(TestAttribute)
print("Decision Tree")
print("Accuracy: ", metrics.accuracy_score(TestTargetEncode, DTPredicted))
print("-------------------------------------")

# KNN
trainBeans = trainBeans.to_numpy()
testBeans = testBeans.to_numpy()

dist = np.array([])
k = 100

hit = 0
miss = 0


def edist(a, b):
    total = 0
    for pos in range(featureNum):
        total += (a[pos] - b[pos]) ** 2

    return np.sqrt(total)

for test in testBeans:
    count = 0
    derma = 0
    seker = 0
    sira = 0
    barbu = 0
    bomba = 0
    kNear = {}
    target = np.zeros(sampleNum, dtype='U5')
    dist = np.zeros((sampleNum, 2))

    for data in trainBeans:
        dist[count][0] = edist(test, data)
        dist[count][1] = count
        target[count] = data[featureNum]
        count += 1

    dist.view('i8,i8').sort(order=['f0'], axis=0)

    for x in range(k):
        temp = np.zeros(k, dtype='U5')
        temp[x] = target[int(dist[x][1])]

        kNear = {'SEKER': np.count_nonzero(temp == 'SEKER'),
                 'DERMASON': np.count_nonzero(temp == 'DERMA'),
                 'BARBUNYA': np.count_nonzero(temp == 'BARBU'),
                 'BOMBAY': np.count_nonzero(temp == 'BOMBA'),
                 'SIRA': np.count_nonzero(temp == 'SIRA'),
                 'CALI': np.count_nonzero(temp == 'CALI'),
                 'HOROZ': np.count_nonzero(temp == 'HOROZ')}

    guess = max(kNear, key=kNear.get)

    if guess == test[featureNum]:
        hit += 1
    else:
        miss += 1

print("K-NN")
print("Hits: ", hit, " Misses: ", miss, " Accuracy: ", hit/(hit+miss))
print("-------------------------------------")
