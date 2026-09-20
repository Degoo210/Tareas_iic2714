import numpy as np
from skimage import color, io

img_dog = io.imread('./images/dog.jpg')
pixel_rgb = img_dog[100, 150] 
print(f"RGB Original: {pixel_rgb}")

pixel_float = pixel_rgb.astype(float) / 255.0
pixel_hsv = color.rgb2hsv(pixel_float.reshape(1, 1, 3))[0, 0]
H = pixel_hsv[0]
S = pixel_hsv[1]
print(f"HSV -> Tono (H): {H}, Saturación original (S): {S}")

pts = [(0.0, 3.0), (0.1, 3.0), (0.25, 1.0), (0.8, 1.0), (1.0, 3.0)]
h_pts = np.array([p[0] for p in pts])
m_pts = np.array([p[1] for p in pts])

h_pad = np.concatenate(([h_pts[-1] - 1.0], h_pts, [h_pts[0] + 1.0]))
m_pad = np.concatenate(([m_pts[-1]], m_pts, [m_pts[0]]))
m_h = np.interp(H, h_pad, m_pad)
print(f"Valor interpolado: {m_h}")

new_S = S * m_h
new_S_clipped = np.clip(new_S, 0.0, 1.0)
print(f"Resultado de g_m: {new_S}")
print(f"Nueva Saturación: {new_S_clipped}")

pixel_hsv[1] = new_S_clipped
pixel_rgb_final = color.hsv2rgb(pixel_hsv.reshape(1, 1, 3))[0, 0]
pixel_rgb_out = (np.clip(pixel_rgb_final, 0.0, 1.0) * 255).astype(np.uint8)
print(f"Valor RGB final: {pixel_rgb_out}")