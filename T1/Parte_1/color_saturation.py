import numpy as np
from skimage import color

def ColorSaturation(img_rgb, control_points, mode='HSV'):
    mode = mode.strip().upper()

    if mode not in ['HSV', 'LCH']:
        ValueError("Modo no soportado. Usa 'HSV' o 'LCH'.")

    if img_rgb.shape[-1] == 4:
        img_rgb = img_rgb[:, :, :3]

    if img_rgb.dtype == np.uint8:
        img_float = img_rgb.astype(float) / 255.0
    else:
        img_float = img_rgb.copy()

    c_pts = sorted(control_points, key=lambda x: x[0])
    h_pts = np.array([p[0] for p in c_pts])
    m_pts = np.array([p[1] for p in c_pts])

    if mode == 'HSV':
        img_transformed = color.rgb2hsv(img_float)
        H = img_transformed[:, :, 0]
        S = img_transformed[:, :, 1]
        max_h = 1.0

    elif mode == 'LCH':
        img_lab = color.rgb2lab(img_float)
        img_transformed = color.lab2lch(img_lab)
        H = img_transformed[:, :, 2]
        S = img_transformed[:, :, 1] # croma
        max_h = 2 * np.pi


    h_pts_padded = np.concatenate(([h_pts[-1] - max_h], h_pts, [h_pts[0] + max_h]))
    m_pts_padded = np.concatenate(([m_pts[-1]], m_pts, [m_pts[0]]))
    

    M_map = np.interp(H, h_pts_padded, m_pts_padded)

    new_S = S * M_map
    
    if mode == 'HSV':
        img_transformed[:, :, 1] = np.clip(new_S, 0.0, 1.0)
        img_out = color.hsv2rgb(img_transformed)
    elif mode == 'LCH':

        img_transformed[:, :, 1] = np.clip(new_S, 0.0, None)
        img_lab_out = color.lch2lab(img_transformed)
        img_out = color.lab2rgb(img_lab_out)
        
    img_out = np.clip(img_out, 0.0, 1.0)
    return (img_out * 255).astype(np.uint8)