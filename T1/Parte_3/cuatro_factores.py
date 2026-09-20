import matplotlib.pyplot as plt
from skimage import io, data
from reescalado import reescalar_imagen

# Cargar imágenes
img1 = io.imread('./images/dog.jpg')
img_2 = data.checkerboard()

scales = [0.6, 0.8, 1.3, 1.7]

fig, axes = plt.subplots(4, 4, figsize=(14, 12))

for i, s in enumerate(scales):
    out_v1 = reescalar_imagen(img1, s, mode='vecino')
    out_b1 = reescalar_imagen(img1, s, mode='bilineal')
    
    h1, w1 = out_v1.shape[:2]
    crop_v1 = out_v1[int(h1*0.35):int(h1*0.55), int(w1*0.35):int(w1*0.55)]
    crop_b1 = out_b1[int(h1*0.35):int(h1*0.55), int(w1*0.35):int(w1*0.55)]
    
    out_v2 = reescalar_imagen(img_2, s, mode='vecino')
    out_b2 = reescalar_imagen(img_2, s, mode='bilineal')
    
    h2, w2 = out_v2.shape[:2]
    crop_v2 = out_v2[int(h2*0.35):int(h2*0.55), int(w2*0.35):int(w2*0.55)]
    crop_b2 = out_b2[int(h2*0.35):int(h2*0.55), int(w2*0.35):int(w2*0.55)]
    
    axes[i, 0].imshow(crop_v1)
    axes[i, 0].set_title(f'Vecino s={s}')
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(crop_b1)
    axes[i, 1].set_title(f'Bilineal s={s}')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(crop_v2, cmap='gray')
    axes[i, 2].set_title(f'Vecino s={s}')
    axes[i, 2].axis('off')
    
    axes[i, 3].imshow(crop_b2, cmap='gray')
    axes[i, 3].set_title(f'Bilineal s={s}')
    axes[i, 3].axis('off')

plt.tight_layout()
plt.show()