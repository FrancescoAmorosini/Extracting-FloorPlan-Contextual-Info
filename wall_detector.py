from PIL import Image
import numpy as np

def detect_walls(img_path, draw_histogram = False):

    grey_tol = 200

    img = Image.open(img_path).convert('RGBA')
    pix = img.load()
    
    row_len, col_len, wallimg = calculate_avg_wall(img,grey_tol, draw_histogram)
    newpix = np.rot90(np.flip(np.asarray(wallimg, dtype=np.int), axis = 1))

    print('Average wall width = ', row_len, ' Average wall height= ', col_len )
    img.show()
    
    return morphologic_elaboration(newpix, row_len, col_len)

    
def morphologic_elaboration(bitimg, row_len, col_len):
    from scipy.ndimage import binary_dilation, binary_erosion

    negative = 1 - bitimg
    modified = np.zeros(bitimg.shape)

    clean_structure = np.zeros((3,3), dtype=np.int)
    clean_structure[:, 1] = 1
    clean_structure[1, :] = 1

    structures = get_structures(row_len, col_len)

    #Union of multiple indepentent erosions
    for strc in structures:
        temp = binary_erosion(negative, structures[strc]).astype(np.int)
        temp = binary_erosion(temp, clean_structure, int((row_len + col_len)/22)+1 ).astype(np.int)
        modified = np.logical_or(modified, temp).astype(np.int)
    
    modified = 1 - modified

    newimg = Image.new('1', bitimg.shape, 1)
    newpix = newimg.load()

    #Create new img from modified bitimg
    for y in range(bitimg.shape[1]):
        for x in range(bitimg.shape[0]):
            if modified[x,y] == 0:
                newpix[x,y] = 0

    return newimg, modified
    

def calculate_avg_wall(img, grey_tol, draw_histogram = False):
    from collections import Counter

    upper_limit = 30
    pix = img.load()
    newimg = Image.new('1', img.size, 1)
    newpix = newimg.load()

    #Horizontal lines
    row_sum = np.array([])
    sum = 0
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pix[x,y] < (grey_tol, grey_tol, grey_tol, np.inf):
                sum +=1
                newpix[x,y] = 0
            else:
                if sum > 0 and sum <= upper_limit:
                    row_sum = np.append(row_sum, sum)
                sum = 0

    occurrene_count = Counter(row_sum)
    count_h = occurrene_count.most_common()
    min_frequency = np.mean( [x[1] for x in count_h]) / 3

    for x in count_h:
        if x[1] < min_frequency:
            count_h = count_h[:count_h.index(x)]
            row_avg = max([y[0] for y in count_h])
            break

    #Vertical lines
    col_sum = np.array([])
    sum = 0
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if pix[x,y] < (grey_tol, grey_tol, grey_tol, np.inf):
                sum +=1
                newpix[x,y] = 0
            else:
                if sum > 0 and sum <= upper_limit:
                    col_sum = np.append(col_sum, sum)
                sum = 0

    occurrene_count = Counter(col_sum)
    count_v = occurrene_count.most_common()
    min_frequency = np.mean( [x[1] for x in count_v]) / 3

    for x in count_v:
        if x[1] < min_frequency:
            count_v = count_v[:count_v.index(x)]
            col_avg = max([y[0] for y in count_v])
            break

    if draw_histogram:
        from visualizer import visualize_histogram
        visualize_histogram(row_sum)
        visualize_histogram(col_sum)
        
    return row_avg, col_avg, newimg


def get_structures(row_len, col_len):
    structures = {}
    #Horizontal
    structures['horizontal']= np.ones((int(row_len*2), 1))
    #Vertical
    structures['vertical']= np.ones((1, int(row_len*2)))
    #45 degrees
    diag = int(np.sqrt(row_len**2 + col_len**2)/4)
    square = np.zeros((diag, diag))
    np.fill_diagonal(square, 1)
    structures['45deg'] = square
    structures['45deg-flipped'] = np.rot90(square)

    return structures