import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

np.random.seed(210)
N = 40

def generar_imagen_y_mascaras():
    size = 256
    img = np.full((size, size), 0.15)
    
    start, end = (size - 128) // 2, (size + 128) // 2
    img[start:end, start:end] = 0.45
    
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = 32
    circle = (x - center)**2 + (y - center)**2 <= radius**2
    img[circle] = 0.80
    
    mask_back = (img == 0.15)
    mask_cuadrado = (img == 0.45)
    mask_circulo = (img == 0.80)

    return img, mask_back, mask_cuadrado, mask_circulo






def kernel_gaussiano(sigma):
    if sigma == 0:
        return np.array([[1.0]])
    
    radio = int(np.ceil(3 * sigma))
    y, x = np.mgrid[-radio:radio+1, -radio:radio+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def aplicar_filtro_gaussiano(imagen, sigma):
    if sigma == 0:
        return imagen.copy()
    kernel = kernel_gaussiano(sigma)

    return convolve(imagen, kernel, mode='reflect')

def calcular_rmse(img1, img2, mascara=None):
    if mascara is None:
        return np.sqrt(np.mean((img1 - img2)**2))
    return np.sqrt(np.mean((img1[mascara] - img2[mascara])**2))