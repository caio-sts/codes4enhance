def denoTV(img1):
  return denoise_tv_chambolle(img1, weight=0.1, multichannel=True)

def denoBilat(img1):
  return denoise_bilateral(img1, multichannel=True, sigma_color=0.05, sigma_spatial=15)

def denoWavelet(img1):
  return denoise_wavelet(img1, multichannel=True, mode="soft", rescale_sigma=True)
