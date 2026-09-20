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

r_size = 64
s_size = 32

res_clip_25 = ecualizacion_local_malla(img, r_size, s_size, clip_limit=25)
res_clip_50 = ecualizacion_local_malla(img, r_size, s_size, clip_limit=50)
res_clip_150 = ecualizacion_local_malla(img, r_size, s_size, clip_limit=150)
res_no_clip = ecualizacion_local_malla(img, r_size, s_size, clip_limit=None)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(res_clip_25, cmap='gray')
axes[1].set_title("Restricción Fuerte\n(clip_limit=25)")
axes[1].axis('off')

axes[2].imshow(res_clip_50, cmap='gray')
axes[2].set_title("Restricción Media\n(clip_limit=50)")
axes[2].axis('off')

axes[3].imshow(res_clip_150, cmap='gray')
axes[3].set_title("Restricción Débil\n(clip_limit=150)")
axes[3].axis('off')

axes[4].imshow(res_no_clip, cmap='gray')
axes[4].set_title("Sin Restricción\n(clip_limit=None)")
axes[4].axis('off')

plt.tight_layout()
plt.show()