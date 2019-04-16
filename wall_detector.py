from PIL import Image
import numpy as np

def detect_walls(img_path, showHistogram = False):

    grey_tol = 200

    img = Image.open(img_path).convert('RGBA')
    pix = img.load()
    
    img.show()
    row_len, col_len, wallimg = calculate_avg_wall(img,grey_tol, showHistogram)
    newpix = np.rot90(np.flip(np.asarray(wallimg, dtype=np.int), axis = 1))
    
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
        temp = binary_erosion(temp, clean_structure, int(min([row_len, col_len])/13)+1).astype(np.int)
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
    

def calculate_avg_wall(img, grey_tol, showHistogram = False):
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
    mean_freq = np.mean( [x[1] for x in count_h])
    row_avg = find_local_max(count_h, mean_freq)
    print('Average wall width = ', row_avg)
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
    mean_freq = np.mean( [x[1] for x in count_v])
    col_avg = find_local_max(count_v, mean_freq)
    print(' Average wall height= ', col_avg) 

    if showHistogram:
        from visualizer import visualize_histogram
        visualize_histogram(row_sum)
        visualize_histogram(col_sum)
        
    return row_avg, col_avg, newimg


def find_local_max(count, mean_freq):
    sorted_by_value = sorted(count, key= lambda tup:tup[0])
    maxes = []
    previous = None
    prevprev = None
    for x in sorted_by_value:
        if previous is not None  and prevprev is not None:
            if previous[1] > x[1] and previous[1] > prevprev[1]:
                if previous[0] > 4:
                    maxes.append(previous)
        prevprev = previous
        previous = x

    while len(maxes) > 3:
        maxes.remove( min(maxes, key=lambda tup:tup[1]))
    return  maxes[1][0]


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
    #30/60 degrees
    square = np.zeros((5,3))
    square[0,0] = 1
    square[1,1] = 1
    square[2,1] = 1
    square[3,2] = 1
    square[4,2] = 1
    structures['30deg'] = square
    structures['60deg'] = np.rot90(square, k=1)
    structures['30deg-flipped'] = np.rot90(square, k=2)
    structures['60deg-flipped'] = np.rot90(square, k=3)

    return structures