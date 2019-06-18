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

        if (max(hist[0]) > 4 and sum(hist[0] > 10)) or ((max(hist[0]) > 1 or sum(hist[0] > 3)) and (x == 'bidet' or obj == 'bidet')):
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

def create_trainingset(node_list, edge_list, anns, categories):
    global objects_stored
    training_set = {}
    for node in node_list:
        data = {}
        data['my_category'] = node['category']
        data['d_bathtub'] = np.inf
        data['d_shower'] = np.inf
        data['d_toilet'] = np.inf
        data['d_bidet'] = np.inf
        data['d_bathrm_sink'] = np.inf 
        data['d_hot_plate'] = np.inf
        data['d_dining_table'] = np.inf
        data['d_table'] = np.inf
        data['d_chair'] = np.inf
        data['d_kitch_sink'] = np.inf
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
        
        if data['my_category'] == 'sink' :
            closest = np.inf
            for ann in anns:
                tol = calculate_tol(anns)
                dist = calculate_center_dist(anns[node['index']],ann)
                if dist < tol*1.3 and dist > 0:
                    actual_dist = calculate_actual_dist(anns[node['index']], ann)
                    key = 'd_' + ann['category_name']
                    if key in data and actual_dist < data[key] :
                        if ann['category_name'] in ('bathtub', 'shower', 'toilet', 'bidet', 'bathrm_sink'):
                            data[key] = actual_dist
                            if actual_dist < closest:
                                closest = actual_dist
                                node['category'] = 'bathrm_sink'
                        elif ann['category_name'] in ('hot_plate', 'kitch_sink', 'chair', 'table', 'dining_table'):
                            data[key] = actual_dist
                            if actual_dist < closest:
                                closest = actual_dist
                                node['category'] = 'kitch_sink'
        training_set[objects_stored] = data
        objects_stored +=1
    return training_set

def is_close(ann0, ann1, dist, tol, bitimg):
    if dist > tol :
        return False
    
    #Find centers
    box0 = [ ann0['bbox'][0], ann0['bbox'][1], 
        ann0['bbox'][0] + ann0['bbox'][2], ann0['bbox'][1] + ann0['bbox'][3] ]
    box1 = [ ann1['bbox'][0], ann1['bbox'][1], 
        ann1['bbox'][0] + ann1['bbox'][2], ann1['bbox'][1] + ann1['bbox'][3] ]

    x0 ,y0 = ( np.average([box0[0], box0[2]]), np.average([box0[1], box0[3]]))
    x1 ,y1 = ( np.average([box1[0], box1[2]]), np.average([box1[1], box1[3]]))
    #Check black points in center axis
    if x0 != x1 or y0 != y1:
        line = xiaoline(x0, y0, x1, y1)
        connected = True
        for point in line:
            if bitimg[int(point[0]), int(point[1])] == 0:
                connected = False
                break
        if connected:
            return True 
    for seg0 in ann0['segmentation']:
        for coord0 in range(0, len(seg0), 2):
            x0 = seg0[coord0]
            y0 = seg0[coord0 + 1]
            for seg1 in ann1['segmentation']:
                for coord1 in range(0, len(seg1), 2):
                    x1 = seg1[coord1]
                    y1 = seg1[coord1 + 1]
                    #Consider line between segmentation points
                    line = np.array([])
                    if x0 != x1 or y0 != y1:
                        line = xiaoline(x0, y0, x1, y1)
                    #Check black points in line
                    connected = True
                    for point in line:
                        if bitimg[int(point[0]), int(point[1])] == 0:
                            connected = False
                            break
                    if connected:
                        return True
    return False


def calculate_center_dist(ann0, ann1):
    box0 = [ ann0['bbox'][0], ann0['bbox'][1], 
        ann0['bbox'][0] + ann0['bbox'][2], ann0['bbox'][1] + ann0['bbox'][3] ]
    box1 = [ ann1['bbox'][0], ann1['bbox'][1], 
        ann1['bbox'][0] + ann1['bbox'][2], ann1['bbox'][1] + ann1['bbox'][3] ]

    x0 ,y0 = ( np.average([box0[0], box0[2]]), np.average([box0[1], box0[3]]))
    x1 ,y1 = ( np.average([box1[0], box1[2]]), np.average([box1[1], box1[3]]))

    dist = np.sqrt( (x1 - x0)**2 + (y1 - y0)**2)

    return dist

def calculate_actual_dist(ann0, ann1):
    box0 = [ ann0['bbox'][0], ann0['bbox'][1], 
        ann0['bbox'][0] + ann0['bbox'][2], ann0['bbox'][1] + ann0['bbox'][3] ]
    box1 = [ ann1['bbox'][0], ann1['bbox'][1], 
        ann1['bbox'][0] + ann1['bbox'][2], ann1['bbox'][1] + ann1['bbox'][3] ]

    x0 ,y0 = ( np.average([box0[0], box0[2]]), np.average([box0[1], box0[3]]))
    x1 ,y1 = ( np.average([box1[0], box1[2]]), np.average([box1[1], box1[3]]))

    if ann0 == ann1:
        return 0
    
    #Initialize dist on center axis
    poly0 = np.array(ann0['segmentation'][0], dtype=np.int).reshape((int(len(ann0['segmentation'][0]) / 2), 2))
    poly1 = np.array(ann1['segmentation'][0], dtype=np.int).reshape((int(len(ann1['segmentation'][0]) / 2), 2))
    axis = xiaoline(x0, y0, x1, y1)

    mindist = len(axis)
    min0 = [x0,y0]
    min1 = [x1,y1]

    prev = poly0[-1]
    for point0 in poly0:
        line = xiaoline(point0[0], point0[1], prev[0], prev[1])
        crossing = [x for x in line if x in axis]
        if crossing:
            for x in crossing:
                dist = np.sqrt( (min1[0] - x[0])**2 + (min1[1] - x[1])**2)
                if dist < mindist:
                    mindist = dist
                    min0 = x
        prev = point0
    prev = poly1[-1]
    for point1 in poly1:
        line = xiaoline(point1[0], point1[1], prev[0], prev[1])
        crossing = [x for x in line if x in axis]
        if crossing:
            for x in crossing:
                dist = np.sqrt( (min0[0] - x[0])**2 + (min0[1] - x[1])**2)
                if dist < mindist:
                    mindist = dist
                    min1 = x
        prev = point1
    #Find closest vertex of poly0 to poly1 contour
    for point0 in poly0:
        line = xiaoline(x1, y1, point0[0], point0[1])
        prev = poly1[-1]
        for point1 in poly1:
            #Check distance between vertices
            dist = np.sqrt( (point0[0] - point1[0])**2 + (point0[1] - point1[1])**2)
            if dist < mindist:
                mindist = dist
                min0 = point0
                min1 = point1
            #Find closest contour point in poly1
            line2 = xiaoline(prev[0], prev[1], point1[0], point1[1])
            crossing = [x for x in line2 if x in line]
            if crossing:
                for x in crossing:
                    dist = np.sqrt( (point0[0] - x[0])**2 + (point0[1] - x[1])**2)
                    if dist < mindist:
                        mindist = dist
                        min0 = point0
                        min1 = x   
            prev = point1
    #Find point of poly0 contour to closest point of poly0 contour
    line = xiaoline(x0, y0, min1[0], min1[1])
    prev = poly0[-1]
    for point0 in poly0:
        line2 = xiaoline(point0[0], point0[1], prev[0], prev[1])
        crossing = [x for x in line2 if x in line]
        if crossing:
            for x in crossing:
                dist = np.sqrt( (min1[0] - x[0])**2 + (min1[1] - x[1])**2)
                if dist < mindist:
                    mindist = dist
                    min0 = x
        prev = point0
    return mindist

def calculate_tol(anns):
    widths = np.array([])
    heights = np.array([])
    for ann in anns:
        widths = np.append(widths, ann['bbox'][2])
        heights = np.append(heights, ann['bbox'][3])

    mean_width = np.mean(widths)
    mean_height = np.mean(heights)

    mean_diag = np.sqrt( mean_width**2 + mean_height**2)
    return 3*mean_diag


def xiaoline(x0, y0, x1, y1):
    #Returns a zip with the points between the input
    if x0 == x1 and y0 == y1:
        return []
    x=[]
    y=[]
    dx = x1-x0
    dy = y1-y0
    steep = abs(dx) < abs(dy)

    if steep:
        x0,y0 = y0,x0
        x1,y1 = y1,x1
        dy,dx = dx,dy

    if x0 > x1:
        x0,x1 = x1,x0
        y0,y1 = y1,y0

    gradient = float(dy) / float(dx)  # slope

    """ handle first endpoint """
    xend = round(x0)
    yend = y0 + gradient * (xend - x0)
    xpxl0 = int(xend)
    ypxl0 = int(yend)
    x.append(xpxl0)
    y.append(ypxl0) 
    x.append(xpxl0)
    y.append(ypxl0+1)
    intery = yend + gradient

    """ handles the second point """
    xend = round (x1)
    yend = y1 + gradient * (xend - x1)
    xpxl1 = int(xend)
    ypxl1 = int (yend)
    x.append(xpxl1)
    y.append(ypxl1) 
    x.append(xpxl1)
    y.append(ypxl1 + 1)

    """ main loop """
    for px in range(xpxl0 + 1 , xpxl1):
        x.append(px)
        y.append(int(intery))
        x.append(px)
        y.append(int(intery) + 1)
        intery = intery + gradient

    if steep:
        y,x = x,y

    coords=zip(x,y)

    return list(coords)


def visualize_histogram(data, upper_limit):
    bins = len(set(data))
    plt.hist(data, bins, facecolor='blue', alpha=0.5)
    axes = plt.gca()
    axes.set_xlim([0,upper_limit])
    axes.set_ylim([0, 1000])
    plt.xlabel('Line Length')
    plt.ylabel('# of Occurrencies')
    plt.show()