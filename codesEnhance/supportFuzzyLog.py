from skimage.color import rgb2lab, lab2rgb
from skimage.color import rgb2hsv,hsv2rgb

def global_stretching(img_L,height, width):
    length = height * width
    R_rray = (np.copy(img_L)).flatten()
    R_rray.sort()
    #print('R_rray',R_rray)
    I_min = int(R_rray[int(length / 100)])
    I_max = int(R_rray[-int(length / 100)])
    #print('I_min',I_min)
    #print('I_max',I_max)
    array_Global_histogram_stretching_L = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if img_L[i][j] < I_min:
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = 0
            elif (img_L[i][j] > I_max):
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = 100
            else:
                p_out = int((img_L[i][j] - I_min) * ((100) / (I_max - I_min)))
                array_Global_histogram_stretching_L[i][j] = p_out
    return (array_Global_histogram_stretching_L)

def global_Stretching_ab(a,height, width):
    array_Global_histogram_stretching_L = np.zeros((height, width), 'float64')
    for i in range(0, height):
        for j in range(0, width):
                p_out = a[i][j] * (1.3 ** (1 - math.fabs(a[i][j] / 128)))
                array_Global_histogram_stretching_L[i][j] = p_out
    return (array_Global_histogram_stretching_L)

def stretching(img):
    height = len(img)
    width = len(img[0])
    for k in range(0, 3):
        Max_channel  = np.max(img[:,:,k])
        Min_channel  = np.min(img[:,:,k])
        for i in range(height):
            for j in range(width):
                img[i,j,k] = (img[i,j,k] - Min_channel) * (255 - 0) / (Max_channel - Min_channel)+ 0
    return img
def  LABStretching(sceneRadiance):


    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    height = len(sceneRadiance)
    width = len(sceneRadiance[0])
    img_lab = rgb2lab(sceneRadiance)
    L, a, b = cv2.split(img_lab)

    img_L_stretching = global_stretching(L, height, width)
    img_a_stretching = global_Stretching_ab(a, height, width)
    img_b_stretching = global_Stretching_ab(b, height, width)

    labArray = np.zeros((height, width, 3), 'float64')
    labArray[:, :, 0] = img_L_stretching
    labArray[:, :, 1] = img_a_stretching
    labArray[:, :, 2] = img_b_stretching
    img_rgb = lab2rgb(labArray) * 255



    return img_rgb

def rghs(img):
  height = len(img)
  width = len(img[0])
  # sceneRadiance = RGB_equalisation(img)

  sceneRadiance = img
  # sceneRadiance = RelativeGHstretching(sceneRadiance, height, width)

  sceneRadiance = stretching(sceneRadiance)

  sceneRadiance = LABStretching(sceneRadiance)
  return sceneRadiance
