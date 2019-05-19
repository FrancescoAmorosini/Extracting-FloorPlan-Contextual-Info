import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import pickle
from random import randint
from collections import Counter

def show_obj_hist(categories, obj = None):
    with open("output.pickle", "rb") as file:  
        data = pickle.load(file)
    
    if obj is None:
        obj = categories[randint(0,len(categories))]
    curves = {}
    for x in categories:
        counter = Counter(data[(obj, x)])
        count = counter.most_common()
        curves.update({x : data[(obj, x)]})
        if len(data[(obj,x)]) > 3:
            plt.hist(data[(obj, x)], bins=25, label = x, alpha = 0.8, histtype='step')

    plt.title(obj)
    plt.legend()
    plt.show()