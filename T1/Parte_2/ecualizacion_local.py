import numpy as np

def calcular_cdf_limitada(region, num_bins=256, clip_limit=None):

    hist, _ = np.histogram(region.flatten(), bins=num_bins, range=(0, 256))
    
    if clip_limit is not None:
        exceso = 0
        for i in range(num_bins):
            if hist[i] > clip_limit:
                exceso += hist[i] - clip_limit
                hist[i] = clip_limit
        
        redistribucion = exceso // num_bins
        hist = hist + redistribucion
        
        
    cdf = hist.cumsum()

    if cdf.max() == cdf.min():
        return np.zeros_like(cdf, dtype=np.uint8)
    
    cdf_normalizada = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    return cdf_normalizada.astype(np.uint8)

def ecualizacion_local_malla(img, region_size, step_size, clip_limit=None, num_bins=256):

    h, w = img.shape
    salida = np.zeros_like(img, dtype=np.float32)
    pesos_totales = np.zeros_like(img, dtype=np.float32)
    
    y_starts = np.arange(0, h - region_size + 1, step_size)
    x_starts = np.arange(0, w - region_size + 1, step_size)
    
    if len(y_starts) == 0:
        y_starts = np.array([0])

    if len(x_starts) == 0:
        x_starts = np.array([0])

    for y in y_starts:
        for x in x_starts:
            region = img[y:y+region_size, x:x+region_size]
            
            cdf_local = calcular_cdf_limitada(region, num_bins, clip_limit)

            factor_escala = num_bins / 256.0
            indices = (region * factor_escala).astype(int)
            indices = np.clip(indices, 0, num_bins - 1)
            
            region_ecualizada = cdf_local[indices]
            
            y_grid, x_grid = np.ogrid[0:region.shape[0], 0:region.shape[1]]
            peso_y = np.minimum(y_grid + 1, region.shape[0] - y_grid)
            peso_x = np.minimum(x_grid + 1, region.shape[1] - x_grid)
            matriz_pesos = peso_y * peso_x
            
            salida[y:y+region_size, x:x+region_size] += region_ecualizada * matriz_pesos
            pesos_totales[y:y+region_size, x:x+region_size] += matriz_pesos

    pesos_totales[pesos_totales == 0] = 1 
    
    salida_final = salida / pesos_totales
    
    return np.clip(salida_final, 0, 255).astype(np.uint8)