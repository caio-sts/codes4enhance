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
import tabulate

def calculateImageMetrics(origPath: str, storePath: str):
    
    i = 0
    metrics = np.zeros(10)

    for imgind1 in os.listdir(origPath):
        imgind2 = os.listdir(storePath)[i]
        ## faz array com todas as métricas, cada coluna(posição) é uma métrica,
        ##  soma a cada iteração, depois tira a média
        
        img1 = mpimg.imread(origPath+"/"+imgind1)
        img2 = mpimg.imread(storePath+"/"+imgind2)

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

    headers = np.array(["RMSE", "CNR", "AMBE", "IEM", "PSNR", "EME", "AMEE", "colourIndex", "averGrad", "Entropy2d"])
    table = metrics / len(os.listdir(origPath))
    print(tabulate(table, headers, tablefmt="pretty"))
        
    return 

import random
from skimage import io 
from skimage import transform as tf

def geometricImageAugmentation(origPath: str, storePath: str, numberOfGenerations: int, paramShear: int = 0 , paramRotation: int = 0, paramTranslatX: int = 0, paramTranslatY: int = 0):
    """Apply geometric transformations on images like shear, rotation and translation for data augmentation.

    Args:
        origPath (str): Path to origin folder.
        storePath (str): Path to destiny folder.
        numberOfGenerations (int): number that must be generated for each image in origin folder.
        paramShear (int, optional): Angle for shearing in degrees. Defaults to 0.
        paramRotation (int, optional): Angle for rotation in degrees. Defaults to 0.
        paramTranslatX (int, optional): Scale factor for horizontal translation. Defaults to 0.
        paramTranslatY (int, optional): Scale factor for vertical translation. Defaults to 0.

    Raises:
        ValueError: origin and destiny paths mustn't be the same!
        ValueError: the number of Generations must be equal or larger than 1.
    """
    # Handling problems like equal paths and unexistent directories
    if origPath == storePath:
        raise ValueError("Origin and destiny paths mustn't be the same!")
    else:
        if not os.path.isdir(origPath):
            os.makedirs(origPath)
        if not os.path.isdir(storePath):
            os.makedirs(storePath)
    
    if numberOfGenerations <= 0:
        raise ValueError("The number of Generations must be equal or larger than 1.")
    else:
        print(f"\n######## Starting Data Augmentation. ########\n")
        for image in os.listdir(origPath):
            if image[:-3] == "png" or "jpg" or "tif":
                for i in range(numberOfGenerations):

                    operation = random.randint(0,2)
                    if operation == 0 and paramShear == 0:
                        img = mpimg.imread(origPath+"/"+image)
                        mpimg.imsave(storePath+"/"+str(i)+" "+image, img)
                    elif operation == 0 and paramShear != 0:
                        img = io.imread(origPath+"/"+image)
                        afine_tf = tf.AffineTransform(shear=paramShear*(np.pi/180))
                        img_shear = tf.warp(img, inverse_map=afine_tf)
                        io.imsave(storePath+"/"+str(i)+" "+image, (255*img_shear).astype(np.uint8))
                                
                    elif operation == 1 and paramRotation == 0:
                        img = mpimg.imread(origPath+"/"+image)
                        mpimg.imsave(storePath+"/"+str(i)+" "+image, img)
                    elif operation == 1 and paramRotation != 0:
                        img = io.imread(origPath+"/"+image)
                        afine_tf = tf.AffineTransform(rotation=paramRotation*(np.pi/180))
                        img_rotat = tf.warp(img, inverse_map=afine_tf)
                        io.imsave(storePath+"/"+str(i)+" "+image, (255*img_rotat).astype(np.uint8))

                    elif operation == 2 and paramTranslatX == 0 and paramTranslatY == 0:
                        imgArray = mpimg.imread(origPath+"/"+image)
                        mpimg.imsave(storePath+"/"+str(i)+" "+image, img)
                    elif operation == 2 and (paramTranslatY != 0 or paramTranslatX != 0):
                        img = io.imread(origPath+"/"+image)
                        afine_tf = tf.AffineTransform(translation=(paramTranslatX, paramTranslatY))
                        img_transl = tf.warp(img, inverse_map=afine_tf)
                        io.imsave(storePath+"/"+str(i)+" "+image, (255*img_transl).astype(np.uint8))
                        
            print(f" {i+1} images of {image} were generated.")        

        print(f"\n######## Total of {len(os.listdir(storePath))} generated images. ########")

        return
