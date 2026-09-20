import matplotlib.pyplot as plt
from skimage import io, util, color
from ecualizacion_local import ecualizacion_local_malla

img = io.imread('./images/P4_IMG_2267_CFA.tif')

if img.ndim == 3:
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img_gray = color.rgb2gray(img)
else:
    img_gray = img

img = util.img_as_ubyte(img_gray)

# Experimento 1
res_d1 = ecualizacion_local_malla(img, 256, 128, 100)
res_d2 = ecualizacion_local_malla(img, 128, 64, 100)
res_d3 = ecualizacion_local_malla(img, 64, 32, 100)

# Experimento 2
res_o1 = ecualizacion_local_malla(img, 128, 128, 100)
res_o2 = res_d2
res_o3 = ecualizacion_local_malla(img, 128, 32, 100)

# Visualización
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(res_d1, cmap='gray')
axes[0, 0].set_title("Baja Densidad\nRegion=256, Step=128")
axes[0, 0].axis('off')

axes[0, 1].imshow(res_d2, cmap='gray')
axes[0, 1].set_title("Media Densidad\nRegion=128, Step=64")
axes[0, 1].axis('off')

axes[0, 2].imshow(res_d3, cmap='gray')
axes[0, 2].set_title("Alta Densidad\nRegion=64, Step=32")
axes[0, 2].axis('off')

axes[1, 0].imshow(res_o1, cmap='gray')
axes[1, 0].set_title("Sin Overlap (0%)\nRegion=128, Step=128")
axes[1, 0].axis('off')

axes[1, 1].imshow(res_o2, cmap='gray')
axes[1, 1].set_title("Overlap Medio (50%)\nRegion=128, Step=64")
axes[1, 1].axis('off')

axes[1, 2].imshow(res_o3, cmap='gray')
axes[1, 2].set_title("Overlap Alto (75%)\nRegion=128, Step=32")
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
