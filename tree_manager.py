from sklearn import tree
from sklearn.model_selection import validation_curve
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pydotplus
import collections

def create_decision_tree(x, y):

    clf = tree.DecisionTreeClassifier(criterion = 'entropy')
    #Find the best value for min_impurity decrease
    param_range = np.logspace(-7,0,10)
    train_score, validation_score = validation_curve(clf, x, y, 'min_impurity_decrease', param_range, cv = 4)
    y1 = np.mean(train_score, axis = 1)
    y2 = np.mean(validation_score, axis = 1)
    plt.plot(param_range, y1, marker = 'o', label='Training Scores')
    plt.plot(param_range, y2, marker = 'o', label='Test Scores')
    plt.xscale('log')
    plt.xlabel('Min Impurity Decrease')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
    #Return the best tree
    index = int(np.where(y2 == y2.max())[0][0])
    best_clf = tree.DecisionTreeClassifier(criterion = 'entropy', min_impurity_decrease = param_range[index] )
    best_clf.fit(x,y)
    return best_clf

def sink_tree():

    plt.close('all')
    with open("training_data.pickle", "rb") as file:  
        data = pickle.load(file)
        file.close()
    #Create features and labels
    x = []
    y = []
    for index, feature in data.items():
        obj = data[index]
        if obj['my_category']in ('sink', 'bathrm_sink', 'kitch_sink'):
            y.append(obj['my_category'])
            features = []
            for cat, values in obj.items():
                if cat != 'my_category' and cat[1] != '_':
                    features = features + [values[0][0], int(values[0][1])]
                elif cat[1] == '_':
                    features = features + [values]
            features = [1000 if i==np.inf else i for i in features]
            x.append(features)
    
    clf = create_decision_tree(x, y)
    tree.export_graphviz(clf, out_file= 'sink_data.dot',class_names= ['bathrm_sink', 'kitch_sink', 'sink'],
    filled=False, rounded=True, special_characters=True, leaves_parallel=False)
    
    with open("sink_data.dot", "rb") as file:
            dot_data = file.read()
            file.close()

    graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    graph.write_png("sink_tree.png")

def table_tree():

    plt.close('all')
    with open("training_data.pickle", "rb") as file:  
        data = pickle.load(file)
        file.close()
    #Create features and labels
    x = []
    y = []
    for index, feature in data.items():
        obj = data[index]
        if obj['my_category']in ('table', 'small_table', 'dining_table'):
            y.append(obj['my_category'])
            features = []
            for cat, values in obj.items():
                if cat != 'my_category' and cat[1] != '_':
                    features = features + [values[0][0], int(values[0][1])]
                elif cat[1] == '_':
                    features = features + [values]
            features = [1000 if i==np.inf else i for i in features]
            x.append(features)
    
    clf = create_decision_tree(x, y)
    tree.export_graphviz(clf, out_file= 'table_data.dot',class_names= ['dining_table', 'small_table', 'table'],
    filled=False, rounded=True, special_characters=True, leaves_parallel=False)
    
    with open("table_data.dot", "rb") as file:
            dot_data = file.read()
            file.close()

    graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    graph.write_png("table_tree.png")
