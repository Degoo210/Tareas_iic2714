import matplotlib.pyplot as plt
from skimage import data
from reescalado import reescalar_imagen

img = data.checkerboard()

img_succ = reescalar_imagen(reescalar_imagen(img, 1.2, 'bilineal'), 1.2, 'bilineal')
img_sing = reescalar_imagen(img, 1.44, 'bilineal')

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(img_succ[150:300, 150:300], cmap='gray')
axes[0].set_title('Sucesivo (1.2 -> 1.2)')
axes[0].axis('off')
axes[1].imshow(img_sing[150:300, 150:300], cmap='gray')
axes[1].set_title('Único (1.44)')
axes[1].axis('off')

plt.tight_layout()
plt.show()