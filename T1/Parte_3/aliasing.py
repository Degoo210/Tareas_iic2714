import matplotlib.pyplot as plt
from skimage import data
from reescalado import reescalar_imagen

checker = data.checkerboard()
red_checker_v = reescalar_imagen(checker, 0.35, 'vecino')
red_checker_b = reescalar_imagen(checker, 0.35, 'bilineal')
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(checker, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(red_checker_v, cmap='gray')
axes[1].set_title('Vecino s=0.35 (Aliasing)')
axes[2].imshow(red_checker_b, cmap='gray')
axes[2].set_title('Bilineal s=0.35')

for ax in axes: ax.axis('off')

plt.tight_layout()
plt.show()