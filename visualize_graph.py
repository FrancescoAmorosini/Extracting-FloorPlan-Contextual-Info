
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

def visualize(wallimg, bitimg, anns, colors, showImg=True, showGraph = True):
    from PIL import Image, ImageDraw
    import networkx as nx

    layer = Image.new('RGBA', wallimg.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    tol = calculate_tol(anns)

    node_list = np.array([], dtype=[('index',int), ('color',"U7")])
    edge_list = np.array([], dtype=[('node1', int), ('node2', int)])
    for ann in anns:
        category_name = ann['category_name']
        c = colors[ann['category_id']]
        #Draws polygons on image
        for seg in ann['segmentation']:
            poly = np.array(seg,dtype=np.int).reshape((int(len(seg) / 2), 2))
            draw.polygon([(x,y) for x, y in poly],
                         fill=(c[0], c[1], c[2], 127), outline=(c[0], c[1], c[2], 255))
        #Appends nodes and edges
        n_c= '#%02x%02x%02x2' % (c[0],c[1],c[2])
        index = anns.index(ann)
        node_list = np.append(node_list, np.array([(index, n_c)], dtype= node_list.dtype))

        for node in range(len(node_list)):
            if is_close(ann, anns[node], tol, bitimg):
                edge_list = np.append(edge_list, np.array([(node, anns.index(ann))], dtype= edge_list.dtype))
    
    G = nx.Graph()
    G.add_nodes_from(node_list['index'])
    G.add_edges_from(edge_list)
    pos= nx.spring_layout(G)
    #Draw Graph
    for node in range(len(node_list)):
        nx.draw_networkx_nodes(G,pos, nodelist=[node_list[node]['index']],
         node_color= node_list[node]['color'], node_size=50)
    nx.draw_networkx_edges(G,pos, alpha=0.8)

    image = Image.alpha_composite(wallimg.convert('RGBA'), layer)
    if showImg:
        image.show()
    if showGraph:
        plt.show()    

def is_close(ann1, ann2, tol, bitimg):
    #Find center of the bboxes
    box1 = [ ann1['bbox'][0], ann1['bbox'][1], 
        ann1['bbox'][0] + ann1['bbox'][2], ann1['bbox'][1] + ann1['bbox'][3] ]
    box2 = [ ann2['bbox'][0], ann2['bbox'][1], 
        ann2['bbox'][0] + ann2['bbox'][2], ann2['bbox'][1] + ann2['bbox'][3] ]

    x1 ,y1 = ( np.average([box1[0], box1[2]]), np.average([box1[1], box1[3]]))
    x2 ,y2 = ( np.average([box2[0], box2[2]]), np.average([box2[1], box2[3]]))

    line = np.array([])
    if x1 != x2 and y1 != y2:
        line = xiaoline(x1, y1, x2, y2)

    dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2)

    if dist < tol:
        for point in line:
            if bitimg[point[0], point[1]] == 0:
                return False
        return True
    else: 
        return False

def calculate_tol(anns):
    widths = np.array([])
    heights = np.array([])
    for ann in anns:
        widths = np.append(widths, ann['bbox'][2])
        heights = np.append(heights, ann['bbox'][3])

    mean_width = np.mean(widths)
    mean_height = np.mean(heights)

    mean_diag = np.sqrt( mean_width**2 + mean_height**2)
    return 2.5*mean_diag


def xiaoline(x0, y0, x1, y1):

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

    return np.array(list(coords))