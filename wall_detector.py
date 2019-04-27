from PIL import Image
import numpy as np

def detect_walls(img_path, show_histogram = False):

    grey_tol = 220

    img = Image.open(img_path).convert('L')
    pix = img.load()
    
    img.show()
    img_diag = int(np.sqrt(img.size[0]**2+img.size[1]**2))
    row_len, col_len, wallimg = calculate_avg_wall(img, grey_tol, show_histogram)
    newpix = np.rot90(np.flip(np.asarray(wallimg, dtype=np.int), axis = 1))

    return morphologic_elaboration(newpix, row_len, col_len, img_diag)


    
def morphologic_elaboration(bitimg, row_len, col_len, img_diag):
    from scipy.ndimage import binary_dilation, binary_erosion

    negative = 1 - bitimg
    modified = np.zeros(bitimg.shape)

    clean_structure = np.zeros((3,3), dtype=np.int)
    clean_structure[:, 1] = 1
    clean_structure[1, :] = 1

    itr = int(img_diag/750)
    if itr == 0 : itr +=1
    print('Diagonal: ', img_diag, ' Erosions: ', itr)
    structures = get_structures(row_len, col_len)
    #Union of multiple indepentent erosions
    for strc in structures:
        temp = binary_erosion(negative, structures[strc]).astype(np.int)
        temp = binary_erosion(temp, clean_structure, itr).astype(np.int)
        temp = binary_dilation(temp, clean_structure, itr).astype(np.int)
        #temp = binary_dilation(temp).astype(np.int)
        modified = np.logical_or(modified, temp).astype(np.int)
    
    modified = 1 - modified

    #Create new img from modified bitimg
    newimg = Image.new('1', bitimg.shape, 1)
    newpix = newimg.load()
    indexes = np.where(modified == 0)
    indexes = list(zip(indexes[0], indexes[1]))
    for x,y in indexes:
        newpix[int(x),int(y)] = 0
    
    return newimg, modified
    

def calculate_avg_wall(img, grey_tol, show_histogram):
    from collections import Counter
    #Calculate walls upper limit on img size
    img_diag = int(np.sqrt(img.size[0]**2+img.size[1]**2))
    upper_limit = int(img_diag/16)
    pix = img.load()
    newimg = Image.new('1', img.size, 1)
    newpix = newimg.load()
    #Count horizontal lines
    row_sum = np.array([])
    sum = 0
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pix[x,y] < grey_tol:
                sum +=1
                newpix[x,y] = 0
            else:
                if sum > 0 and sum < upper_limit:
                    row_sum = np.append(row_sum, sum)
                sum = 0
    occurrene_count = Counter(row_sum)
    count_h = occurrene_count.most_common()
    row_avg = find_local_max(count_h)
    print('Average wall width = ', row_avg)
    #Count vertical lines
    col_sum = np.array([])
    sum = 0
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if pix[x,y] < grey_tol:
                sum +=1
                newpix[x,y] = 0
            else:
                if sum > 0 and sum < upper_limit:
                    col_sum = np.append(col_sum, sum)
                sum = 0
    occurrene_count = Counter(col_sum)
    count_v = occurrene_count.most_common()
    col_avg = find_local_max(count_v)
    print('Average wall height= ', col_avg) 

    if show_histogram:
        from visualizer import visualize_histogram
        visualize_histogram(row_sum, upper_limit)
        visualize_histogram(col_sum, upper_limit)
    
    return row_avg, col_avg, newimg


def find_local_max(count):
    sorted_by_value = sorted(count, key= lambda tup:tup[0])
    #Inserts zero values
    for i in range(int(sorted_by_value[-1][0])):
        if i+1 != sorted_by_value[i][0]:
            sorted_by_value.insert(i, (i+1,0))
    sorted_by_value.insert(len(sorted_by_value), (len(sorted_by_value) + 1,0))
    maxes = []
    previous = (0,0)
    prevprev = (0,0)
    #Find all maxes
    for x in sorted_by_value:
        if previous[1] > x[1] and previous[1] > prevprev[1] and previous[0] < x[0]:
            maxes.append(previous)
        prevprev = previous
        previous = x
    #Find 3 global maxes and return the best one
    rest = [x for x in maxes if x[0] <= 4]
    maxes = [x for x in maxes if x[0] > 4]
    if len(maxes) == 0 : return rest[-1][0]
    elif len(maxes) < 3 :
        return max([rest[-1],maxes[0]], key=lambda tup:tup[1])[0]
    while len(maxes) > 3:
        min_value = min(maxes, key=lambda tup:tup[1])
        min_list = [i for i,x in enumerate(maxes) if x[1] == min_value[1]]
        maxes.pop(min_list[-1])
    if maxes[0][0] > maxes[0][1]:
        return maxes[0][0]

    return max([maxes[1], maxes[2]], key=lambda tup:tup[1])[0]


def get_structures(row_len, col_len):
    structures = {}
    #Horizontal
    structures['horizontal']= np.ones((int(row_len), 1))
    #Vertical
    structures['vertical']= np.ones((1, int(col_len)))
    #45 degrees
    diag = int(np.sqrt(row_len**2 + col_len**2)/2)
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