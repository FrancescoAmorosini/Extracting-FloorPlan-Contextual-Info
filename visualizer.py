import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import util
from PIL import Image, ImageDraw

def visualize_graph(wallimg, anns, colors, categories, path, showImg=True, showGraph = False):
    import networkx as nx
    from networkx.drawing.nx_agraph import graphviz_layout

    layer = Image.new('RGBA', wallimg.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    drawhite = ImageDraw.Draw(wallimg)
    tol = calculate_tol(anns)

    node_list = np.array([], dtype=[('index',int), ('color',"U7"), ('category',"U12")])
    edge_list = np.array([], dtype=[('node1', int), ('node2', int), ('distance', float)])
    occurrence_data = np.array([], dtype=[('node1', "U12"), ('node2', "U12"), ('distance', float)])
    for ann in anns:
        category_name = ann['category_name']
        c = colors[ann['category_id']]
        #Draws polygons on new image
        for seg in ann['segmentation']:
            poly = np.array(seg,dtype=np.int).reshape((int(len(seg) / 2), 2))
            drawhite.polygon([(x,y) for x, y in poly],
                         fill='white', outline='white')
            draw.polygon([(x,y) for x, y in poly],
                         fill=(c[0], c[1], c[2], 127), outline=(c[0], c[1], c[2], 255))
        #Appends nodes and edges
        n_c= '#%02x%02x%02x2' % (c[0],c[1],c[2])
        index = anns.index(ann)
        node_list = np.append(node_list, np.array([(index, n_c, category_name)], dtype= node_list.dtype))

        for node in range(len(node_list)):
            dist = calculate_center_dist(ann, anns[node])
            if dist > 0 and is_close(ann, anns[node], dist, tol, wallimg.load()):
                dist =calculate_actual_dist(ann, anns[node])
                normalized = dist / max([ann['bbox'][2], ann['bbox'][3]])
                edge_list = np.append(edge_list, np.array([(node, anns.index(ann), round(normalized, 3))], dtype= edge_list.dtype))
    #Check subcategories and save results
    node_list = util.check_subcategories(node_list, edge_list, banned = [])
    crop_sinks(anns, node_list, path)
    crop_tables(anns, node_list, path)
    
    for edge in edge_list:
        occurrence_data = np.append(occurrence_data, np.array([(node_list[edge['node1']]['category'], node_list[edge['node2']]['category'], edge['distance'])], dtype= occurrence_data.dtype))
    #Draw graph and img
    if showImg:
        image = Image.alpha_composite(wallimg.convert('RGBA'), layer)
        image.show() 

    if showGraph:
        G = nx.Graph()
        G.add_nodes_from(node_list['index'])
        G.add_weighted_edges_from(edge_list, 'distance')
        pos= graphviz_layout(G, prog='circo')
        for node in range(len(node_list)):
            nx.draw_networkx_nodes(G,pos, nodelist=[node_list[node]['index']],
            node_color= node_list[node]['color'],alpha = .8, node_size=150) 
        nx.draw_networkx_edges(G, pos)
        #Adds labels
        mapping = dict(zip(node_list['index'], node_list['category']))
        nx.draw_networkx_labels(G, pos, labels = mapping, font_size = 8)
        edge_labels = nx.get_edge_attributes(G,'distance')
        nx.draw_networkx_edge_labels(G,pos, edge_labels=edge_labels, font_size = 6, alpha=0.8)
    
        plt.xticks([])
        plt.yticks([]) 
        plt.show()
    
    training_set = util.create_trainingset(node_list, edge_list, categories)
    return occurrence_data, training_set

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
    return 2.8*mean_diag


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

save = False
def crop_sinks(anns, node_list, img_path):
    import os
    #Create sink folders
    wallimg = Image.open(img_path).convert('L')
    global save
    if not os.path.isdir('./sinks'):
        os.mkdir('./sinks')
        os.mkdir('./sinks/bathrm_sink')
        os.mkdir('./sinks/kitch_sink')
        save = True
    #If sink folders are empty, fill them
    if save:
        for node in node_list:
            if node['category'] in ('sink', 'bathrm_sink', 'kitch_sink'):
                bbox = anns[node['index']]['bbox']
                area = (int(bbox[0] - 1.5*bbox[2]), int(bbox[1] - 1.5*bbox[3]), int(bbox[0] + 3*bbox[2]), int(bbox[1] + 3*bbox[3]))
                path = './sinks'
                img_name = img_path[20::]
                img_name = img_name[0:-4]
                if node['category'] in ('bathrm_sink', 'kitch_sink'):
                    path = path + '/' + node['category']
                
                newimg= wallimg.crop(area)
                newimg.save(path + img_name + 'ID' + str(node['index']) + '.png')
        return None
    #If skink folder is filled, update obj classes
    else:
        img_name = img_path[20::]
        img_name = img_name[0:-4]
        for node in node_list:
           if node['category'] in ('sink','bathrm_sink', 'kitch_sink'):
                if os.path.isfile('./sinks/' + node['category'] + '/' + img_name + 'ID' + str(node['index']) + '.png'):
                   continue
                else:
                    if os.path.isfile('./sinks/bathrm_sink/' + img_name + 'ID' + str(node['index']) + '.png'):
                        node['category'] = 'bathrm_sink'
                    elif os.path.isfile('./sinks/kitch_sink/' + img_name + 'ID' + str(node['index']) + '.png'):
                        node['category'] = 'kitch_sink'
    return node_list

def crop_tables(anns, node_list, img_path):
    import os
    #Create sink folders
    wallimg = Image.open(img_path).convert('L')
    global save
    if not os.path.isdir('./tables'):
        os.mkdir('./tables')
        os.mkdir('./tables/small_table')
        os.mkdir('./tables/dining_table')
        save = True
    #If table folders are empty, fill them
    if save:
        for node in node_list:
            if node['category'] in ('table','small_table', 'dining_table'):
                bbox = anns[node['index']]['bbox']
                area = (int(bbox[0] - 1.5*bbox[2]), int(bbox[1] - 1.5*bbox[3]), int(bbox[0] + 3*bbox[2]), int(bbox[1] + 3*bbox[3]))
                path = './tables'
                img_name = img_path[20::]
                img_name = img_name[0:-4]
                if node['category'] in ('small_table', 'dining_table'):
                    path = path + '/' + node['category']
                
                newimg= wallimg.crop(area)
                newimg.save(path + img_name + 'ID' + str(node['index']) + '.png')
        return None
    #If table folder is filled, update obj classes
    else:
        img_name = img_path[20::]
        img_name = img_name[0:-4]
        for node in node_list:
           if node['category'] in ('table','small_table', 'dining_table'):
                if os.path.isfile('./tables/' + node['category'] + '/' + img_name + 'ID' + str(node['index']) + '.png'):
                   continue
                else:
                    if os.path.isfile('./tables/small_table/' + img_name + 'ID' + str(node['index']) + '.png'):
                        node['category'] = 'small_table'
                    elif os.path.isfile('./tables/dining_table/' + img_name + 'ID' + str(node['index']) + '.png'):
                        node['category'] = 'dining_table'
    return node_list