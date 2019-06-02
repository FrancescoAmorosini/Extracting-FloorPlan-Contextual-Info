import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import pickle
from random import randint
from scipy.interpolate import make_interp_spline, BSpline


def show_obj_hist(categories, obj = None):
    plt.close('all')
    with open("hist.pickle", "rb") as file:  
        data = pickle.load(file)
    
    if obj is None:
        obj = categories[randint(0,len(categories))]
    for x in categories:
        if obj == 'sink':
            newdata = data[(obj,x)] + data[('bathrm_sink', x)] + data[('kitch_sink', x)]
            hist = np.histogram(newdata, bins=25)
        elif obj == 'table':
            newdata = data[(obj,x)] + data[('small_table', x)] + data[('dining_table', x)]
            hist = np.histogram(newdata, bins=25)
        else:
            hist = np.histogram(data[(obj,x)], bins=25)

        if (max(hist[0]) > 4 and sum(hist[0] > 7)) or (sum(hist[0]) > 3 and (x == 'bidet' or obj == 'bidet')):
            hist = (hist[0], np.delete(hist[1], -1))
            xnew = np.linspace(min(hist[1]), max(hist[1]), 300)
            spl = make_interp_spline(hist[1], hist[0], k=1)
            power_smooth = spl(xnew)
            plt.plot(xnew, power_smooth, label=x)

    plt.title(obj)
    plt.legend()
    plt.xlabel('Distance/Longest Side Ratio')
    plt.ylabel('# of Occurrencies')
    plt.show()

def check_subcategories(node_list, edge_list, banned):
    added = np.array([], dtype=node_list.dtype)
    for node in node_list:
        if (node['category'] in ('table, sink')):
            closest = None
            dist = np.inf
            for edge in edge_list:
                #Check table
                if edge['node1'] == node['index'] and node['category'] == 'table':
                    if node_list[edge['node2']]['category'] in ('couch', 'armchair', 'small_table', 'dining_table' 'chair') and edge['distance'] < dist:
                        closest = node_list[edge['node2']]['category']
                        dist = edge['distance']
                elif edge['node2'] == node['index'] and node['category'] == 'table':
                    if node_list[edge['node1']]['category'] in ('couch', 'armchair', 'small_table', 'dining_table', 'chair') and edge['distance'] < dist:
                        closest = node_list[edge['node1']]['category']
                        dist = edge['distance']
                #Check sink
                elif edge['node1'] == node['index'] and node['category'] == 'sink':
                    if node_list[edge['node2']]['category'] in ('bathtub', 'shower', 'toilet', 'bidet', 'hot_plate', 'bathrm_sink', 'kitch_sink', 'chair', 'table', 'dining_table') and edge['distance'] < dist:
                        closest = node_list[edge['node2']]['category']
                        dist = edge['distance']
                    if closest is None and node_list[edge['node2']]['category'] == 'sink':
                        closest = node_list[edge['node2']]['category']
                elif edge['node2'] == node['index'] and node['category'] == 'sink':
                    if node_list[edge['node1']]['category'] in ('bathtub', 'shower', 'toilet', 'bidet', 'hot_plate', 'bathrm_sink', 'kitch_sink', 'chair', 'table', 'dining_table') and edge['distance'] < dist:
                        closest = node_list[edge['node1']]['category']
                        dist = edge['distance']
                    if closest is None and node_list[edge['node1']]['category'] == 'sink':
                        closest = node_list[edge['node1']]['category']

            #Change category
            if closest in ('couch', 'armchair', 'small_table') and node['category'] == 'table':
                node['category'] = 'small_table'
            elif closest in ('chair', 'dining_table') and node['category'] == 'table':
                node['category'] = 'dining_table'
            elif closest in ('bathtub', 'shower', 'toilet', 'bidet', 'bathrm_sink') and node['category'] == 'sink':
                node['category'] = 'bathrm_sink'
            elif closest in ('hot_plate', 'kitch_sink', 'chair', 'table', 'dining_table') and node['category'] == 'sink':
                node['category'] = 'kitch_sink'
            elif closest is not None and node not in banned:
                added = np.append(added, node)
                banned.append(node)
    
    if added.size > 0:
        node_list = check_subcategories(node_list, edge_list, banned)
    return node_list

objects_stored = 0

def create_trainingset(node_list, edge_list, categories):
    global objects_stored
    training_set = {}
    for node in node_list:
        data = {}
        data['my_category'] = node['category']
        for cat in categories:
            data[cat] = np.array([(np.inf, 0)], dtype=[('distance',float), ('occurrences', int)])
        for edge in edge_list:
            if edge['node1'] == node['index']:
                data[node_list[edge['node2']]['category']]['occurrences'] += 1
                if edge['distance'] < data[node_list[edge['node2']]['category']]['distance']:
                   data[node_list[edge['node2']]['category']]['distance'] = edge['distance']

            elif edge['node2'] == node['index']:
                data[node_list[edge['node1']]['category']]['occurrences'] += 1
                if edge['distance'] < data[node_list[edge['node1']]['category']]['distance']:
                   data[node_list[edge['node1']]['category']]['distance'] = edge['distance']
        
        training_set[objects_stored] = data
        objects_stored +=1
    return training_set

def visualize_histogram(data, upper_limit):
    bins = len(set(data))
    plt.hist(data, bins, facecolor='blue', alpha=0.5)
    axes = plt.gca()
    axes.set_xlim([0,upper_limit])
    axes.set_ylim([0, 1000])
    plt.xlabel('Line Length')
    plt.ylabel('# of Occurrencies')
    plt.show()