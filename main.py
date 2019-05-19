from visualizer import visualize_graph
from wall_detector import detect_walls
from  Flo2PlanManager.manager.visual import ManageCOCO
import numpy as np
import pickle
import os
import util

json_path = './trainingset/annotations/flo2plan_icdar_instances.json'
image_path = './trainingset/images/'
script_path = './Flo2PlanManager/visual_instances_image.py'

def __main__():
    manager = ManageCOCO(json_path, image_path)

    categories = manager.getCategories()
    for cat in categories:
            categories[cat - 1] = manager.getCategoryName(cat)
    #Produce data on first run
    if not os.path.isfile('./output.pickle'):
        ids = manager.getImageIds()
        inferences_data = np.array([], dtype=[('node1', "U11"), ('node2', "U11"), ('distance', int)])
        #Detect object contexts
        for indx in ids:
            print(indx)
            colors = manager.getColorCategories()
            path, anns = manager.getImageAnnotations(indx)
            wallimg, bitimg = detect_walls(path, show_histogram= False)
            inferences_data = np.append(inferences_data, visualize_graph(wallimg, anns, colors, showImg=False, showLabels=True))
        #Build histograms of object occurrences
        hists = {}
        combinations = [(x,y) for x in categories for y in categories]
        for x in combinations:
            hists.update({x : []})
        for x in inferences_data:
            hists[(x['node1'], x['node2'])].append(x['distance'])
        #Save the results in a text file
        with open("output.pickle", "wb") as file:
            pickle.dump(hists, file)
    
    for cat in categories:
        util.show_obj_hist(categories, cat)


if __name__ == __main__():
    __main__()

thin='G0636 G0438 G0243 G0222 G0795 G0546 014 131 G0653  G0747 G0741 < 087 G0922 104 G0581 090'
medium = ' G0543 G0229 G0736 G0937 G0209  G0503 15 '
big = 'G0316 G0478 032 001 G0065 G0004 G0449 <  G0850 G1006'

dirty = 'G0488' 'G0286' 'G0299'