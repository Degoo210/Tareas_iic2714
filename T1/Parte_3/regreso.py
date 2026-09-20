import matplotlib.pyplot as plt
from skimage import data
from reescalado import reescalar_imagen

img = data.checkerboard()

img_red = reescalar_imagen(img, 0.5, 'bilineal')
img_rec = reescalar_imagen(img_red, 2.0, 'bilineal')

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img[50:200, 150:300], cmap='gray')
axes[0].set_title('Original (Pedazo)')
axes[0].axis('off')
axes[1].imshow(img_red[25:100, 75:150], cmap='gray')
axes[1].set_title('Reducida s=0.5')
axes[1].axis('off')
axes[2].imshow(img_rec[50:200, 150:300], cmap='gray')
axes[2].set_title('Recuperada s=2.0')
axes[2].axis('off')

plt.tight_layout()
plt.show()