import matplotlib.pyplot as plt
from skimage import io
from color_saturation import ColorSaturation

#prueba

img_dog =  io.imread('./images/dog.jpg')
img_desierto =  io.imread('./images/desierto.webp')
  
pts_aumentar = [
    (0.0, 2.5), (0.1, 2.5),
    (0.25, 1.0), (0.8, 1.0),
    (1.0, 2.5)
]

pts_atenuar = [
    (0.0, 1.0), (0.2, 1.0),
    (0.4, 0.0), (0.7, 0.0),
    (0.9, 1.0)
]


pts_mixto = [
    (0.0, 2.0),
    (0.3, 0.2), (0.7, 0.2),
    (1.0, 2.0)      
]

aumentar = ColorSaturation(img_dog, pts_aumentar, mode='HSV')
atenuar = ColorSaturation(img_dog, pts_atenuar, mode='HSV')
mixto = ColorSaturation(img_dog, pts_mixto, mode='HSV')

aumentar_2 = ColorSaturation(img_desierto, pts_aumentar, mode='HSV')
atenuar_2 = ColorSaturation(img_desierto, pts_atenuar, mode='HSV')
mixto_2 = ColorSaturation(img_desierto, pts_mixto, mode='HSV')

fig, axes = plt.subplots(4, 4, figsize=(100, 30))

# Perro
axes[0, 0].imshow(img_dog)
axes[0, 0].set_title("Original")

axes[0, 1].imshow(aumentar)
axes[0, 1].set_title("A: Resaltar Rojos")

axes[0, 2].imshow(atenuar)
axes[0, 2].set_title("B: Atenuar Azules/Verdes")

axes[0, 3].imshow(mixto)
axes[0, 3].set_title("C: Mixto (Rojos +, Azules -)")

# Desierto

axes[1, 0].imshow(img_dog)
axes[1, 0].set_title("Original")

axes[1, 1].imshow(aumentar_2)
axes[1, 1].set_title("A: Resaltar Rojos")

axes[1, 2].imshow(atenuar_2)
axes[1, 2].set_title("B: Atenuar Azules/Verdes")

axes[1, 3].imshow(mixto_2)
axes[1, 3].set_title("C: Mixto (Rojos +, Azules -)")

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout(pad=4)
plt.show()