
import treeplot
from sklearn.cluster import KMeans

def loadDataSet(filepath):
    '''
    Returns
    -----------------
    data: 2-D list
        each row is the feature and label of one instance
    featNames: 1-D list
        feature names
    '''
    data=[]
    featNames = None
    fr = open(filepath)
    for (i,line) in enumerate(fr.readlines()):
        array=line.strip().split(',')
        if i == 0:
            featNames = array[:-1]
        else:
            data.append(array)
    return data, featNames


def splitData(dataSet, axis, value):
    '''
    Split the dataset based on the given axis and feature value

    Parameters
    -----------------
    dataSet: 2-D list
        [n_sampels, m_features + 1]
        the last column is class label
    axis: int 
        index of which feature to split on
    value: string
        the feature value to split on

    Returns
    ------------------
    subset: 2-D list 
        the subset of data by selecting the instances that have the given feature value
        and removing the given feature columns
    '''
    subset = []
    for instance in dataSet:
        if instance[axis] == value:    # if contains the given feature value
            reducedVec = instance[:axis] + instance[axis+1:] # remove the given axis
            subset.append(reducedVec)
    return subset


def chooseBestFeature(dataSet):
    '''
    choose best feature to split based on Gini index
    
    Parameters
    -----------------
    dataSet: 2-D list
        [n_sampels, m_features + 1]
        the last column is class label

    Returns
    ------------------
    bestFeatId: int
        index of the best feature
    '''
    #TODO
    bestFeatId = 0

    num_feat = len(dataSet[0]) - 1

    label = {}
    for i in range(0, len(dataSet)):
        label[dataSet[i][num_feat]] = label.get(dataSet[i][num_feat],0) + 1

    sum_gini = 0
    for key in label:
        sum_gini = sum_gini + ((label[key]/len(dataSet)) ** 2)

    gini_parent = 1 - sum_gini

    subf_count = []
    for feat in range(0, len(dataSet[0])):
        temp_dict = {}
        for j in range(0, len(dataSet)):
            temp_dict[dataSet[j][feat]] = temp_dict.get(dataSet[j][feat],0) + 1
        subf_count.append(temp_dict)

    gini_child = [0] * num_feat

    #specefic feature
    for c in range(0, num_feat):
        #iterate through sub feat
        for subf in subf_count[c]:
            temp_label = {}
            total_cnt = subf_count[c][subf]
            for itr in range(0, len(dataSet)):
                if subf == dataSet[itr][c]:
                    temp_label[dataSet[itr][num_feat]] = temp_label.get(dataSet[itr][num_feat],0) + 1
            mini_gini = 0
            for tiny_key in temp_label:
                mini_gini = mini_gini + ((temp_label[tiny_key] / total_cnt) ** 2)
            real_mini_gini = (1 - mini_gini) * (total_cnt / len(dataSet))
            gini_child[c] = gini_child[c] + real_mini_gini

    gain = []
    for lol in gini_child:
        gain.append(gini_parent-lol)

    maxVal = max(gain)

    bestFeatId = gain.index(maxVal)

    return bestFeatId


def stopCriteria(dataSet):
    '''
    Criteria to stop splitting: 
    1) if all the classe labels are the same, then return the class label;
    2) if there are no more features to split, then return the majority label of the subset.

    Parameters
    -----------------
    dataSet: 2-D list
        [n_sampels, m_features + 1]
        the last column is class label

    Returns
    ------------------
    assignedLabel: string
        if satisfying stop criteria, assignedLabel is the assigned class label;
        else, assignedLabel is None 
    '''
    assignedLabel = None
    # TODO
    lastCol = len(dataSet[0]) - 1
    labels = {}
    for i in range(0, len(dataSet)):
        labels[dataSet[i][lastCol]] = labels.get(dataSet[i][lastCol],0) + 1

    if len(labels) == 1:
        assignedLabel = list(labels.keys())[0]
    elif lastCol == 0:
        for key in labels:
            if assignedLabel == None:
                assignedLabel = key
            elif labels[assignedLabel] < labels[key]:
                assignedLabel = key

    return assignedLabel



def buildTree(dataSet, featNames):
    '''
    Build the decision tree

    Parameters
    -----------------
    dataSet: 2-D list
        [n'_sampels, m'_features + 1]
        the last column is class label

    Returns
    ------------------
        myTree: nested dictionary
    '''
    assignedLabel = stopCriteria(dataSet)
    if assignedLabel:
        return assignedLabel

    bestFeatId = chooseBestFeature(dataSet)
    bestFeatName = featNames[bestFeatId]

    myTree = {bestFeatName:{}}
    subFeatName = featNames[:]
    del(subFeatName[bestFeatId])
    featValues = [d[bestFeatId] for d in dataSet]
    uniqueVals = list(set(featValues))
    for value in uniqueVals:
        myTree[bestFeatName][value] = buildTree(splitData(dataSet, bestFeatId, value), subFeatName)
    
    return myTree

def dataClustering(data):
    Age_data = []
    ka_data = []
    for i in range(0, len(data)):
        Age_data.append([float(data[i][0]), 0])
        ka_data.append([float(data[i][4]), 0])

    kmean_ka_data = KMeans(n_clusters=3, random_state=0).fit(ka_data)
    kmean_age_data = KMeans(n_clusters=3, random_state=0).fit(Age_data)

    lab_ka = kmean_ka_data.labels_
    lab_age = kmean_age_data.labels_
    for i in range(0, len(data)):
        if lab_ka[i] == 0:
            data[i][4] = "Normal"
        if lab_ka[i] == 1:
            data[i][4] = "High"
        if lab_ka[i] == 2:
            data[i][4] = "Low"
        data[i][0] = lab_age[i]

    return data

def correctPredictions(data, featNames, dtTree):
    pred = {}

    pos_outcomes = []
    for i in range(0, len(data)):
        if data[i][5] not in pos_outcomes:
            pos_outcomes.append(data[i][5])


    for j in range(0, len(data)):
        init = list(dtTree.keys())[0]
        ind = featNames.index(init)
        bridge = dtTree[init]
        val = bridge[data[j][ind]]
        while(val not in pos_outcomes):
            init = list(val.keys())[0]
            ind = featNames.index(init)
            bridge = val[init]
            val = bridge[data[j][ind]]
        if val == data[j][5]:
            pred['Yes'] = pred.get('Yes', 0) + 1
        else:
            pred['No'] = pred.get('No', 0) + 1


    print(pred)
    print(pos_outcomes)


    #pred.get(, 0) + 1
    return


if __name__ == "__main__":
    data, featNames = loadDataSet('drug200.csv')
    data = dataClustering(data)
    dtTree = buildTree(data, featNames)
    correctPredictions(data, featNames, dtTree)
    print (dtTree)
    treeplot.createPlot(dtTree)
    