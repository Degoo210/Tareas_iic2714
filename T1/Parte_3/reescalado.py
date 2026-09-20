import numpy as np

def reescalar_imagen(imagen, s, mode='bilineal'):

    if imagen.ndim == 3:
        H, W, C = imagen.shape
    else:
        H, W = imagen.shape
        C = 1
        imagen = imagen.reshape((H, W, C))
        
    H_out = int(np.round(H * s))
    W_out = int(np.round(W * s))
    
    i_out, j_out = np.indices((H_out, W_out))
    
    u = (i_out + 0.5) / s - 0.5
    v = (j_out + 0.5) / s - 0.5
    
    u = np.clip(u, 0, H - 1)
    v = np.clip(v, 0, W - 1)
    
    salida = np.zeros((H_out, W_out, C), dtype=np.float32)
    
    if mode == 'vecino':
        u_nearest = np.round(u).astype(int)
        v_nearest = np.round(v).astype(int)
        
        for c in range(C):
            salida[:, :, c] = imagen[u_nearest, v_nearest, c]
            
    elif mode == 'bilineal':
        i0 = np.floor(u).astype(int)
        i1 = np.clip(i0 + 1, 0, H - 1)
        j0 = np.floor(v).astype(int)
        j1 = np.clip(j0 + 1, 0, W - 1)
        
        dy = u - i0
        dx = v - j0
        
        w00 = (1 - dy) * (1 - dx)
        w01 = (1 - dy) * dx
        w10 = dy * (1 - dx)
        w11 = dy * dx
        
        for c in range(C):
            I00 = imagen[i0, j0, c]
            I01 = imagen[i0, j1, c]
            I10 = imagen[i1, j0, c]
            I11 = imagen[i1, j1, c]
            
            salida[:, :, c] = w00 * I00 + w01 * I01 + w10 * I10 + w11 * I11
    else:
        raise ValueError("mode debe ser 'vecino' o 'bilineal'")
        
    if salida.shape[2] == 1:
        salida = salida.reshape((H_out, W_out))
        
    return np.clip(salida, 0, 255).astype(imagen.dtype)