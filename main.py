from visualizer import visualize_graph
from wall_detector import detect_walls
import util

json_path = './trainingset/annotations/flo2plan_icdar_instances.json'
image_path = './trainingset/images/'
script_path = './Flo2PlanManager/visual_instances_image.py'

def __main__():
    path, anns, colors = util.get_image_params(json_path, image_path, key=None)
    wallimg, bitimg = detect_walls(path, draw_histogram= False)
    visualize_graph(wallimg, anns, colors, showImg=True, showLabels=True)


if __name__ == __main__():
    __main__()

thin='G0636 G0438 G0243 G0222  G0795 G0546 014 131 G0653 G0747 < 10  087 G0922 104 G0581 090'
medium = ' G0543 G0229 G0736 G0937 G0209 < 15 '
big = 'G0316 G0478 032 001 G0065 G0004> 15  G0850 G1006'

dirty = 'G0488'