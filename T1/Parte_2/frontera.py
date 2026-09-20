import matplotlib.pyplot as plt
from skimage import io, util, color
from ecualizacion_local import ecualizacion_local_malla

img = io.imread('./images/P3_IMG_2387_crop.tif')

if img.ndim == 3:
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img_gray = color.rgb2gray(img)
else:
    img_gray = img

img = util.img_as_ubyte(img_gray)

clip = 100
r_size = 64

no_overlap = ecualizacion_local_malla(img, r_size, r_size, clip_limit=clip)

overlap = ecualizacion_local_malla(img, r_size, r_size//2, clip_limit=clip)

y1, y2 = 120, 450
x1, x2 = 350, 680

crop_orig = img[y1:y2, x1:x2]
crop_no_over = no_overlap[y1:y2, x1:x2]
crop_over = overlap[y1:y2, x1:x2]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(crop_orig, cmap='gray')
axes[0].set_title("Recorte Original")
axes[0].axis('off')

axes[1].imshow(crop_no_over, cmap='gray')
axes[1].set_title("Sin Overlap")
axes[1].axis('off')

axes[2].imshow(crop_over, cmap='gray')
axes[2].set_title("Overlap")
axes[2].axis('off')

plt.tight_layout()
plt.show()