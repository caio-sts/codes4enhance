def MSE(img1, img2):
    squared_diff = (img1 -img2) ** 2
    summed = np.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
    err = summed / num_pix
    return err

def MSE_RGB(img1, img2):
    MSE_R = MSE(img1[:,:,0], img2[:,:,0])
    MSE_G = MSE(img1[:,:,1], img2[:,:,1])
    MSE_B = MSE(img1[:,:,2], img2[:,:,2])
    return (MSE_R+MSE_G+MSE_B)/3 

def RMSE(MSE_value): #image quality
  return np.sqrt(MSE_value)

def meanImg(img1):
  return scipy.ndimage.mean(img1)

def standDeviat(img1):
  return scipy.ndimage.standard_deviation(img1)

def CNR(img1, img2):
  im1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
  x = scipy.ndimage.mean(im1)
  im2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
  y = scipy.ndimage.mean(im2)

  C = x-y 
  N = 0
  for i in range(len(img1)):
    for j in range(len(img1[0])):
      N += (im1[i][j] - x )**2 
  N/=(len(img1)*len(img1[0]))

  return C/np.sqrt(N)

def Ambe(img1, img2 ):
   im1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
   meanstat1 = scipy.ndimage.mean(im1)

   im2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
   meanstat2 = scipy.ndimage.mean(im2)

   return meanstat1 - meanstat2

def Psnr(img2, MSE_value): #image quality
  if MSE_value == 0:
    return 0
  return -10*np.log10(MSE_value/(255)**2)

def Eme(img2):
  res = 0
  for i in range(1, len(img2)):
    for j in range(1, len(img2[0])):
      if np.min(img2[:i][:j]) != 0:
        res += 20*np.log(np.max(img2[:i][:j])/np.min(img2[:i][:j]))
  
  return res/(len(img2)*len(img2[0])) 

def Amee(img2):
  res = 0
  alpha = 0.5
  for i in range(1,len(img2)):
    for j in range(1,len(img2[0])):
      if np.min(img2[:i][:j]) != 0:
        res += (alpha * (np.max(img2[:i][:j])/np.min(img2[:i][:j]))**alpha)*np.log(np.max(img2[:i][:j])/np.min(img2[:i][:j]))
  
  return res/(len(img2)*len(img2[0])) 

def colourIndex(img1):
  R = img1[:, :, 0]
  G = img1[:, :, 1]
  B = img1[:, :, 2]
  alpha = R-G
  beta = 0.5*(R+G) - B
  varalpha = scipy.ndimage.standard_deviation(alpha)
  varbeta = scipy.ndimage.standard_deviation(beta)
  meanalpha = scipy.ndimage.mean(alpha)
  meanbeta = scipy.ndimage.mean(beta)

  return (np.sqrt(varalpha**2 + varbeta**2) +0.3*np.sqrt(meanalpha**2 + meanbeta**2))/85.59
          
def averGrad(img1):
  im = img1.astype('int32')
  dx = scipy.ndimage.sobel(im, 0)  # horizontal derivative
  dy = scipy.ndimage.sobel(im, 1)  # vertical derivative
  mag = np.hypot(dx, dy)

  return np.sum(mag)/(len(img1)*len(img1[0])*np.sqrt(2))


def calcIJ(img_patch):
    total_p = img_patch.shape[0] * img_patch.shape[1]
    if total_p % 2 != 0:
        tem = img_patch.flatten()
        center_p = tem[int(total_p / 2)]
        mean_p = (sum(tem) - center_p) / (total_p - 1)
        return (center_p, mean_p)
    else:
        print("modify patch size")

def calcEntropy2d(img, win_w=3, win_h=3, threadNum=6):
    height = img.shape[0]
    width = img.shape[1]

    ext_x = int(win_w / 2)
    ext_y = int(win_h / 2)

    
    ext_h_part = np.zeros([height, ext_x], img.dtype)
    tem_img = np.hstack((ext_h_part, img, ext_h_part))
    ext_v_part = np.zeros([ext_y, tem_img.shape[1]], img.dtype)
    final_img = np.vstack((ext_v_part, tem_img, ext_v_part))

    new_width = final_img.shape[1]
    new_height = final_img.shape[0]

    
    patches = []
    for i in range(ext_x, new_width - ext_x):
        for j in range(ext_y, new_height - ext_y):
            patch = final_img[j - ext_y:j + ext_y + 1, i - ext_x:i + ext_x + 1]
            patches.append(patch)

    
    pool = Pool(processes=threadNum)
    IJ = pool.map(calcIJ, patches)
    pool.close()
    pool.join()
    

    
    Fij = Counter(IJ).items()
    

   
    Pij = []
    for item in Fij:
        Pij.append(item[1] * 1.0 / (new_height * new_width))

    
    H_tem = []
    for item in Pij:
        h_tem = -item * (np.log(item) / np.log(2))
        H_tem.append(h_tem)

    
    H = sum(H_tem)
    return H