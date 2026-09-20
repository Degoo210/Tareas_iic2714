import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, io, util, color
from ecualizacion_local import ecualizacion_local_malla

img_in = io.imread('./images/P4_IMG_2267_CFA.tif')

if img_in.ndim == 3:
    if img_in.shape[2] == 4:
        img_in = img_in[:, :, :3]
    img_gray = color.rgb2gray(img_in)
else:
    img_gray = img_in

img = util.img_as_ubyte(img_gray)

h, w = img.shape

# Se usa para comparar simplemente
img_global_sk = exposure.equalize_hist(img) * 255
img_global_sk = img_global_sk.astype(np.uint8)

region_size_global = max(h, w)
step_size_global = max(h, w)
img_global_malla = ecualizacion_local_malla(img, region_size=region_size_global, step_size=step_size_global)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_global_sk, cmap='gray')
plt.title('Global Clásica (skimage)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_global_malla, cmap='gray')
plt.title(f'Algoritmo Base (1 Región)')
plt.axis('off')

plt.tight_layout()
plt.show()