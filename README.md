# codes4enhance

This repo is a tentative of organize the whole research on methods and metrics in enhancement of medical images. source: https://colab.research.google.com/drive/1rtKaxdgp6FbbcJz_3k_WFhydZYPrtnSB?usp=sharing

ASAP I will referencing papers and some repo fonts that I found some codes used in my research. I hope this style of code organization could be better for intuitive use and accelerate implementing points.

Links of datasets to be used:

- Brain Tumor MRI
https://www.kaggle.com/datasets/firqaaa/brain-tumor-mri-graph-superpixels

- Breast Tumor Ultrassound
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

The related researchers recommend Python <=3.10

After clone the repo, go to workfolder and type in terminal:
python main.py -h

There you will see arguments to apply enhance methods or do data augmentation by applying geometric transformations.

## How to use

 1. Clone this repo
 2. Use this command at terminal:
 ```
 python -m main -o "<origin folder>" -s "<to_save folder>" <other arguments>
 ```
   2.1 If you want to generate enhanced images using one of the available methods, add  the code below in place of <other arguments>   
 ``` 
 -n <referenced alias function code> 
 ```
   2.2 If you want to generate new images using random geometric transformations add  the code below in place of <other arguments>    
 ``` 
 -g True -x <number of images to generate> -sh <angle for shearing> -r <angle for rotation> -tx <horizontal scale translation factor> -ty <vertical scale translation  factor>
 ```
   2.3 If you want to calculate metrics comparing original images and them after enhance, add the code below in place of <other arguments>   
 ``` 
 -cm True 
 ```

## Result examples

Hand x-ray (Original)             |  Histogram Equalization (he)     |  Dynamic Histogram Equalization (dhe)
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/caio-sts/codes4enhance/blob/main/examples/hand-x-ray.png" height="400" width="425"/> |  <img src="https://github.com/caio-sts/codes4enhance/blob/main/examples/0%20hand-x-rayhe.png" height="400" width="425"/> |  <img src="https://github.com/caio-sts/codes4enhance/blob/main/examples/0%20hand-x-raydhe.png" height="400" width="425"/>

Exposure Fusion framework (ying)         |  Total Variation (tv)     |  Wavelet (wavelet)
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/caio-sts/codes4enhance/blob/main/examples/0%20hand-x-rayying.png" height="400" width="425"/> |  <img src="https://github.com/caio-sts/codes4enhance/blob/main/examples/0%20hand-x-raytv.png" height="400" width="425"/> |  <img src="https://github.com/caio-sts/codes4enhance/blob/main/examples/0%20hand-x-raywavelet.png" height="400" width="425"/>

Bilateral (bilateral)         |  Ilumination Map Estimation (lime)     |  Gamma Correction (iagc)
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/caio-sts/codes4enhance/blob/main/examples/0%20hand-x-raybilateral.png" height="400" width="425"/> |  <img src="https://github.com/caio-sts/codes4enhance/blob/main/examples/0%20hand-x-raylime.png" height="400" width="425"/> |  <img src="https://github.com/caio-sts/codes4enhance/blob/main/examples/0%20hand-x-raygammacorrection.png" height="400" width="425"/>
 
Underwater (rghs)         |
:-------------------------:|
<img src="https://github.com/caio-sts/codes4enhance/blob/main/examples/0%20hand-x-rayunderwater.png" height="400" width="425"/> |
 
 
## References
 - Histogram Equalization (HE)  
  W. Zhihong and X. Xiaohong, “Study on Histogram Equalization,” 2011 2nd International Symposium on Intelligence Information Processing and Trusted Computing, 2011. 
 - Dynamic Histogram Equalization (DHE)  
  M. Abdullah-Al-Wadud, M. Kabir, M. Akber Dewan, and O. Chae, “A dynamic histogram equalization for image contrast enhancement,” IEEE Transactions on Consumer Electronics, vol. 53, no. 2, pp. 593–600, 2007. 
 - Image contrast using exposure fusion framework (alias in code: ying)  
  Z. Ying, G. Li, Y. Ren, R. Wang, and W. Wang, “A new image contrast enhancement algorithm using exposure fusion framework,” Computer Analysis of Images and Patterns, pp. 36–46, 2017. 
 - Total Variation Denoise (alias in code: denoTV)  
  Chambolle, A. An Algorithm for Total Variation Minimization and Applications. Journal of Mathematical Imaging and Vision 20, 89–97 (2004).
 - Wavelet Denoise (alias in code: denoWavelet)  
  S. G. Chang, Bin Yu, and M. Vetterli, “Adaptive wavelet thresholding for image denoising and compression,” IEEE Transactions on Image Processing, vol. 9, no. 9, pp. 1532–1546, 2000.
 - Bilateral denoise (alias in code: denoBilat)  
  C. Tomasi and R. Manduchi, “Bilateral filtering for gray and color images,” Sixth International Conference on Computer Vision (IEEE Cat. No.98CH36271). 
 - Low-light image enhancement via Ilumination Map Estimation (LIME)  
  X. Guo, Y. Li, and H. Ling, “Lime: Low-light image enhancement via Illumination Map Estimation,” IEEE Transactions on Image Processing, vol. 26, no. 2, pp. 982–993, 2017. 
 - Contrast enhancement of brightness-distorted images by improved adaptive gamma correction (alias in code: iagc)  
  G. Cao, L. Huang, H. Tian, X. Huang, Y. Wang, and R. Zhi, “Contrast enhancement of brightness-distorted images by improved adaptive gamma correction,” Computers &amp; Electrical Engineering, vol. 66, pp. 569–582, 2018. 
 - Image enhancement and image restoration for underwater images (alias in code: rghs)  
  Y. Wang, W. Song, G. Fortino, L.-Z. Qi, W. Zhang, and A. Liotta, “An experimental-based review of image enhancement and image restoration methods for underwater imaging,” IEEE Access, vol. 7, pp. 140233–140251, 2019. 
