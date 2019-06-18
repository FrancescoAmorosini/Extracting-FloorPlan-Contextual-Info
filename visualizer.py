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
    tol = util.calculate_tol(anns)

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
            dist = util.calculate_center_dist(ann, anns[node])
            if dist > 0 and util.is_close(ann, anns[node], dist, tol, wallimg.load()):
                dist = util.calculate_actual_dist(ann, anns[node])
                normalized = dist / max([ann['bbox'][2], ann['bbox'][3]])
                edge_list = np.append(edge_list, np.array([(node, anns.index(ann), round(normalized, 3))], dtype= edge_list.dtype))
    #Check subcategories and save results
    node_list = util.check_subcategories(node_list, edge_list, banned = [])
    crop_tables(anns, node_list, path)
    training_set = util.create_trainingset(node_list, edge_list, anns, categories)
    node_list = util.check_subcategories(node_list, edge_list, banned = [])

    crop_sinks(anns, node_list, path)
    
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
    
    return occurrence_data, training_set

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