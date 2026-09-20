import matplotlib.pyplot as plt
from skimage import io
from color_saturation import ColorSaturation
import numpy as np

#prueba

img_dog =  io.imread('./images/dog.jpg')
  
pts_aumentar = [
    (0.0, 10), (0.1, 10),
    (0.25, 1.0), (0.8, 1.0),
    (1.0, 10)
]

pts_aumentar_2 = [
    (h * 2 * np.pi, m) for h, m in pts_aumentar
]

aumentar = ColorSaturation(img_dog, pts_aumentar, mode='LCH')

fig, axes = plt.subplots(2, 3, figsize=(100, 30))

# Perro
axes[0, 0].imshow(img_dog)
axes[0, 0].set_title("Original")

axes[0, 1].imshow(aumentar)
axes[0, 1].set_title("Aumentar rojos (LCH con m=3)")

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout(pad=4)
plt.show()