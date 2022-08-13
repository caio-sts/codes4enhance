from matplotlib import image as mpimg
import enhanceLib
from metricsLib import MSE, MSE_RGB, RMSE, CNR, AMBE, IEM, PSNR, EME, AMEE, colourIndex, averGrad, calcEntropy2d 
import os

def indexingImageFiles(origPath: str):
    """Code for index image files starting from zero to number of images in directory.

    Args:
        origPath (str): Path to search images for indexing.

    Raises:
        ValueError: raised when given path doesn't exist.
    """
    if not os.path.exists(origPath):
        raise ValueError("Path given doesn't exist!")
    else:
        numberImages = 0
        for image in os.listdir(origPath):
            if image[:-3] == "png" or "jpg" or "tif":
                os.rename(origPath+"/"+image, origPath+"/"+str(numberImages)+" "+image)
                numberImages += 1
        
        print(f"\n######## {numberImages} indexed files. ########")

        return 
    
def enhance(origPath: str, storePath: str, method: str):
    """Code for apply enhancement methods in an image folder 
    and save to other folder.
    
    Args:
        origPath (str): path to search images and apply an enhancement method to all of them.
        storePath (str): path to save enhanced images.
        method (str): method of enhancement coded in enhanceLib module.
        numberImagesEnhanced (int): counter of enhanced images.
    Raises:
        ValueError: origin and destiny paths mustn't be the same!
    """
    # Handling problems like equal paths and unexistent directories
    if origPath == storePath:
        raise ValueError("Origin and destiny paths mustn't be the same!")
    else:
        if not os.path.isdir(origPath):
            os.makedirs(origPath)
        if not os.path.isdir(storePath):
            os.makedirs(storePath)
    
    numberImagesEnhanced = 0
    enhanceApplier = enhanceLib.findEnhancer(method)

    for image in os.listdir(origPath):
        if image[:-3] == "png" or "jpg" or "tif":
            imgArray = mpimg.imread(origPath+"/"+image)
            img = enhanceApplier(imgArray)
            mpimg.imsave(storePath+"/"+str(numberImagesEnhanced)+" "+image, img)

            numberImagesEnhanced += 1

    print(f"\n######## {numberImagesEnhanced} enhanced images. ########")

    return 

import numpy as np

def calculateImageMetrics(origPath: str, storePath: str):
    
    i = 0
    metrics = np.zeros(10)

    for img1 in os.listdir(origPath):
        img2 = os.listdir(storePath)[i]
        ## faz array com todas as métricas, cada coluna(posição) é uma métrica,
        ##  soma a cada iteração, depois tira a média
        if len(img1.shape) == 2:
            metrics[0] += RMSE(MSE(img1, img2))
            metrics[4] += PSNR(img2, MSE(img1, img2))
        else:
            metrics[0] += RMSE(MSE_RGB(img1, img2))
            metrics[4] += PSNR(img2, MSE_RGB(img1, img2))
        
        metrics[1] += CNR(img1, img2)
        metrics[2] += AMBE(img1, img2)
        metrics[3] += IEM(img1, img2)

        metrics[5] += EME(img2)
        metrics[6] += AMEE(img2)
        metrics[7] += colourIndex(img2)#verificar se é a img2 que vai nessas 3 ultimas métricas
        metrics[8] += averGrad(img2)
        metrics[9] += calcEntropy2d(img2)

        i += 1

calculateImageMetrics("images1", "images2")
