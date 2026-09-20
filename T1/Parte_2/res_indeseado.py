import matplotlib.pyplot as plt
from skimage import io, util, color, exposure
from ecualizacion_local import ecualizacion_local_malla
import numpy as np

img = io.imread('./images/P3_IMG_2387_crop.tif')

if img.ndim == 3:
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img_gray = color.rgb2gray(img)
else:
    img_gray = img

img = util.img_as_ubyte(img_gray)

res_no_limit = ecualizacion_local_malla(img, 64, 32, clip_limit=None)

res_limitado = ecualizacion_local_malla(img, 64, 32, clip_limit=40)

res_clahe = (exposure.equalize_adapthist(img, kernel_size=64, clip_limit=0.01) * 255).astype(np.uint8)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(res_no_limit, cmap='gray')
axes[1].set_title("Local No Limitada")
axes[1].axis('off')

axes[2].imshow(res_limitado, cmap='gray')
axes[2].set_title("Propuesto Limitado (Clip=40)")
axes[2].axis('off')

axes[3].imshow(res_clahe, cmap='gray')
axes[3].set_title("CLAHE (Clip=0.01)")
axes[3].axis('off')

plt.tight_layout()
plt.show()