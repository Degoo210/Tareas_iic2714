import matplotlib.pyplot as plt
from skimage import io, util, color
from ecualizacion_local import ecualizacion_local_malla

img = io.imread('./images/P1_IMG_2402.tif')

if img.ndim == 3:
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img_gray = color.rgb2gray(img)
else:
    img_gray = img

img = util.img_as_ubyte(img_gray)

r_large = 128
s_large = 64
res_L_256 = ecualizacion_local_malla(img, r_large, s_large, clip_limit=None, num_bins=256)
res_L_64  = ecualizacion_local_malla(img, r_large, s_large, clip_limit=None, num_bins=64)
res_L_16  = ecualizacion_local_malla(img, r_large, s_large, clip_limit=None, num_bins=16)

r_small = 32
s_small = 16
res_S_256 = ecualizacion_local_malla(img, r_small, s_small, clip_limit=None, num_bins=256)
res_S_64  = ecualizacion_local_malla(img, r_small, s_small, clip_limit=None, num_bins=64)
res_S_16  = ecualizacion_local_malla(img, r_small, s_small, clip_limit=None, num_bins=16)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Fila 1: Regiones Grandes
axes[0, 0].imshow(res_L_256, cmap='gray')
axes[0, 0].set_title("Región=128, Bins=256")
axes[0, 1].imshow(res_L_64, cmap='gray')
axes[0, 1].set_title("Región=128, Bins=64")
axes[0, 2].imshow(res_L_16, cmap='gray')
axes[0, 2].set_title("Región=128, Bins=16")

# Fila 2: Regiones Pequeñas
axes[1, 0].imshow(res_S_256, cmap='gray')
axes[1, 0].set_title("Región=32, Bins=256")
axes[1, 1].imshow(res_S_64, cmap='gray')
axes[1, 1].set_title("Región=32, Bins=64")
axes[1, 2].imshow(res_S_16, cmap='gray')
axes[1, 2].set_title("Región=32, Bins=16")

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()