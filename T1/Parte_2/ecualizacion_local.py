import numpy as np

def calcular_cdf_limitada(region, num_bins=256, clip_limit=None):

    hist, _ = np.histogram(region.flatten(), bins=num_bins, range=(0, 256))
    
    if clip_limit is not None:
        exceso = 0
        for i in range(num_bins):
            if hist[i] > clip_limit:
                exceso += hist[i] - clip_limit
                hist[i] = clip_limit
        
        redistribucion = exceso // num_bins
        hist = hist + redistribucion
        
        
    cdf = hist.cumsum()
    cdf_normalizada = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    return cdf_normalizada.astype(np.uint8)

def ecualizacion_local_malla(img, region_size, step_size, clip_limit=None, num_bins=256):

    h, w = img.shape
    salida = np.zeros_like(img, dtype=np.float32)
    pesos_totales = np.zeros_like(img, dtype=np.float32)
    
    y_starts = np.arange(0, h - region_size + 1, step_size)
    x_starts = np.arange(0, w - region_size + 1, step_size)
    
    if len(y_starts) == 0:
        y_starts = np.array([0])

    if len(x_starts) == 0:
        x_starts = np.array([0])

    for y in y_starts:
        for x in x_starts:
            region = img[y:y+region_size, x:x+region_size]
            
            cdf_local = calcular_cdf_limitada(region, num_bins, clip_limit)
            
            region_ecualizada = cdf_local[region]
            
            y_grid, x_grid = np.ogrid[0:region.shape[0], 0:region.shape[1]]
            peso_y = np.minimum(y_grid + 1, region.shape[0] - y_grid)
            peso_x = np.minimum(x_grid + 1, region.shape[1] - x_grid)
            matriz_pesos = peso_y * peso_x
            
            salida[y:y+region_size, x:x+region_size] += region_ecualizada * matriz_pesos
            pesos_totales[y:y+region_size, x:x+region_size] += matriz_pesos

    pesos_totales[pesos_totales == 0] = 1 
    
    salida_final = salida / pesos_totales
    
    return np.clip(salida_final, 0, 255).astype(np.uint8)


import matplotlib.pyplot as plt
from skimage import io, color, exposure, util

img_in = io.imread('l.png')

## arreglar (tal vez)
if img_in.ndim == 3:
    if img_in.shape[2] == 4:
        img_in = img_in[:, :, :3]
    img_gray = color.rgb2gray(img_in)
else:
    img_gray = img_in

img_gray = util.img_as_ubyte(img_gray)

h, w = img_gray.shape
eq_global = ecualizacion_local_malla(img_gray, region_size=max(h,w), step_size=max(h,w), clip_limit=None)

eq_local_no_lim = ecualizacion_local_malla(img_gray, region_size=64, step_size=32, clip_limit=None)

eq_local_lim = ecualizacion_local_malla(img_gray, region_size=64, step_size=32, clip_limit=15)

clahe_ref = exposure.equalize_adapthist(img_gray, kernel_size=64, clip_limit=0.01)
clahe_ref = (clahe_ref * 255).astype(np.uint8)

# Ver
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
titulos = ['Original', 'Global', 'Local No Limitada', 'Local Limitada', 'CLAHE Ref']
imagenes = [img_gray, eq_global, eq_local_no_lim, eq_local_lim, clahe_ref]

for ax, img, titulo in zip(axes, imagenes, titulos):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(titulo)
    ax.axis('off')

plt.tight_layout()
plt.show()