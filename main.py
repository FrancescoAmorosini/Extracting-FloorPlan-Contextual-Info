from visualize_graph import visualize
import util

json_path = './trainingset/annotations/flo2plan_icdar_instances.json'
image_path = './trainingset/images/'
script_path = './Flo2PlanManager/visual_instances_image.py'

def draw_walls(img_path, line_length=15):
    from PIL import Image
    import numpy as np

    grey_tol = 130

    img = Image.open(img_path).convert('RGBA')
    pix = img.load()

    newpix = np.zeros(shape = img.size)

    #Horizontal walls
    line = np.array([])
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            #Check for dark pixel
            if pix[x,y] < (grey_tol, grey_tol, grey_tol, np.inf):
                line = np.append(line, [(x,y)])
            else: #Chek line length
                if line.shape[0] > line_length:
                    line = line.reshape((int(len(line) / 2), 2))
                    for point in line:
                        newpix[int(point[0]), int(point[1])] = 1
                line = np.array([])
    #Vertical walls
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            #Check pixel
            if pix[x,y] < (grey_tol, grey_tol, grey_tol, np.inf):
                line = np.append(line, [(x,y)])
            else:
                if line.shape[0] > line_length:
                    line = line.reshape((int(len(line) / 2), 2))
                    for point in line:
                        newpix[int(point[0]), int(point[1])] = 1
                line = np.array([])
    img.show()
    
    return morphologic_elaboration(newpix)

    
def morphologic_elaboration(bitimg):
    from scipy.ndimage import binary_dilation, binary_erosion
    from PIL import Image
    import numpy as np

    structure = np.ones((3,3), dtype = np.int)

    modified = binary_erosion(bitimg, structure=structure).astype(np.int)
    modified = binary_dilation(modified, structure = structure).astype(np.int)
    modified = 1 - modified

    newimg = Image.new('1', bitimg.shape, 1)
    newpix = newimg.load()

    #Create new img from modified bitimg
    for y in range(bitimg.shape[1]):
        for x in range(bitimg.shape[0]):
            if modified[x,y] == 0:
                newpix[x,y] = 0

    print('Walls have been observed!')
    return newimg, modified
    


def __main__():
    path, anns, colors = util.get_image_params(json_path, image_path, key=None)
    
    wallimg, bitimg = draw_walls(path)
    visualize(wallimg, bitimg, anns, colors, showImg=True, showGraph=True)


if __name__ == __main__():
    __main__()

key='G0387'