import matplotlib.pyplot as plt
from skimage import io, color, exposure, util
from ecualizacion_local import ecualizacion_local_malla
import numpy as np

rutas_imagenes = ['./images/dog.jpg', './images/desierto.webp']

fig, axes = plt.subplots(len(rutas_imagenes), 5, figsize=(20, 5 * len(rutas_imagenes)))
titulos = ['Original', 'Global', 'Local No Limitada', 'Local Limitada', 'CLAHE']

for fila_idx, ruta in enumerate(rutas_imagenes):
    img_in = io.imread(ruta)
    
    if img_in.ndim == 3:
        if img_in.shape[2] == 4:
            img_in = img_in[:, :, :3]
        img_gray = color.rgb2gray(img_in)
    else:
        img_gray = img_in
        
    img = util.img_as_ubyte(img_gray)
    h, w = img.shape
    
    eq_global = ecualizacion_local_malla(img, region_size=max(h,w), step_size=max(h,w))
    eq_local_no_lim = ecualizacion_local_malla(img, region_size=64, step_size=32)
    eq_local_lim = ecualizacion_local_malla(img, region_size=64, step_size=32, clip_limit=50)
    
    clahe_ref = exposure.equalize_adapthist(img, kernel_size=64, clip_limit=0.01)
    clahe_ref = (clahe_ref * 255).astype(np.uint8)
    
    imagenes = [img, eq_global, eq_local_no_lim, eq_local_lim, clahe_ref]
    
    for col_idx, (img_procesada, titulo) in enumerate(zip(imagenes, titulos)):
        ax = axes[fila_idx, col_idx]
        ax.imshow(img_procesada, cmap='gray', vmin=0, vmax=255)
        
        if fila_idx == 0:
            ax.set_title(titulo)
        ax.axis('off')

plt.tight_layout()
plt.show()