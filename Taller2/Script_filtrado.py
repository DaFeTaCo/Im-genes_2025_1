import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

# --- 1. Cargar la imagen anatómica ---
input_nifti_path = 'E:\ACTUAL\ACTUAL\P_D_Imagenes\Git-Hub\Imagenes_2025_1\DATA\data_taller2\BIDS\sub-0001/anat\sub-0001_T1w.nii.gz' # ¡Cambia esto por la ruta real de tu archivo!

try:
    img = nib.load(input_nifti_path)
    data = img.get_fdata() # Obtener los datos de la imagen como un array NumPy
    affine = img.affine      # Obtener la matriz afín (importante para guardar)
    header = img.header      # Obtener el encabezado (importante para guardar)
    print(f"Imagen cargada exitosamente. Dimensiones: {data.shape}")
except FileNotFoundError:
    print(f"Error: El archivo '{input_nifti_path}' no se encontró. Verifica la ruta.")
    exit()
except Exception as e:
    print(f"Error al cargar la imagen: {e}")
    exit()

# --- 2. Aplicar Filtro Gaussiano ---
# sigma: Desviación estándar del kernel gaussiano.
# Un valor mayor de sigma produce un suavizado más fuerte.
# Puedes probar diferentes valores para comparar efectos.
sigma_gaussian = 1.5 # Ejemplo: 1.5 mm de suavizado

print(f"Aplicando filtro gaussiano con sigma={sigma_gaussian}...")
filtered_gaussian_data = gaussian_filter(data, sigma=sigma_gaussian)
print("Filtro gaussiano aplicado.")

# Guardar la imagen con filtro gaussiano
output_gaussian_nifti_path = 'E:\ACTUAL\ACTUAL\P_D_Imagenes\Git-Hub\Imagenes_2025_1\DATA\data_taller2\Results\sub-0001_Gaussian' # Cambia esto
gaussian_img = nib.Nifti1Image(filtered_gaussian_data, affine, header)
nib.save(gaussian_img, output_gaussian_nifti_path)
print(f"Imagen gaussiana guardada en: {output_gaussian_nifti_path}")


# --- 3. Aplicar Filtro Anisotrópico (Perona-Malik) ---
# Implementación básica de difusión anisotrópica (Perona-Malik)
# Este filtro es más complejo y computacionalmente más costoso.
# Es iterativo y busca suavizar regiones homogéneas mientras preserva bordes.

def anisotropic_diffusion(img, niter=10, kappa=50, gamma=0.1, step=(1.,1.,1.), option=1):
    """
    Implementación básica de la difusión anisotrópica (Perona-Malik).
    Basado en https://en.wikipedia.org/wiki/Anisotropic_diffusion
    y adaptaciones comunes.

    Parameters:
    ----------
    img : ndarray
        La imagen 3D a filtrar.
    niter : int
        Número de iteraciones.
    kappa : float
        Parámetro de conductancia. Afecta la sensibilidad al gradiente.
        Valores más bajos preservan más los bordes.
    gamma : float
        Factor de paso de tiempo. Debe ser < 0.25 para estabilidad.
    step : tuple
        Tamaño de paso en cada dimensión (por ejemplo, (1,1,1) para isotrópico).
    option : int
        Función de conductancia:
        1: c(x) = exp(-(nablaI/kappa)^2) (favorece bordes suaves)
        2: c(x) = 1 / (1 + (nablaI/kappa)^2) (favorece bordes más marcados)

    Returns:
    -------
    ndarray
        La imagen filtrada.
    """
    img = img.astype(np.float64) # Asegurarse de que los datos sean flotantes
    
    # Inicialización de la imagen de salida
    filtered_img = np.copy(img)
    
    # Define la función de conductancia
    if option == 1:
        def c(delta_I, kappa):
            return np.exp(-(delta_I/kappa)**2)
    elif option == 2:
        def c(delta_I, kappa):
            return 1.0 / (1.0 + (delta_I/kappa)**2)
    else:
        raise ValueError("Opción de conductancia no válida. Usa 1 o 2.")

    # Calcular gradientes discretos (diferencias finitas)
    for _ in range(niter):
        # Gradientes en las 6 direcciones (arriba, abajo, izquierda, derecha, adelante, atrás)
        # Diferencias para calcular los gradientes
        delta_N = filtered_img[1:, :, :] - filtered_img[:-1, :, :]
        delta_S = filtered_img[:-1, :, :] - filtered_img[1:, :, :]
        delta_E = filtered_img[:, 1:, :] - filtered_img[:, :-1, :]
        delta_W = filtered_img[:, :-1, :] - filtered_img[:, 1:, :]
        delta_U = filtered_img[:, :, 1:] - filtered_img[:, :, :-1]
        delta_D = filtered_img[:, :, :-1] - filtered_img[:, :, 1:]

        # Calcular las funciones de conductancia para cada dirección
        c_N = c(delta_N, kappa)
        c_S = c(delta_S, kappa)
        c_E = c(delta_E, kappa)
        c_W = c(delta_W, kappa)
        c_U = c(delta_U, kappa)
        c_D = c(delta_D, kappa)

        # Aplicar la ecuación de difusión anisotrópica
        # Note: se usan slices para que las operaciones de suma/resta
        # coincidan en dimensiones.
        # Por ejemplo, c_N tiene una dimensión menos en el primer eje.
        # Necesitamos que la multiplicación con delta_N y la suma con el resto
        # se aplique a la región común.

        term_N = c_N * delta_N
        term_S = c_S * delta_S
        term_E = c_E * delta_E
        term_W = c_W * delta_W
        term_U = c_U * delta_U
        term_D = c_D * delta_D
        
        # Actualización iterativa:
        # Crea un array para los flujos de corriente (conducción)
        flux_N = np.zeros_like(filtered_img)
        flux_S = np.zeros_like(filtered_img)
        flux_E = np.zeros_like(filtered_img)
        flux_W = np.zeros_like(filtered_img)
        flux_U = np.zeros_like(filtered_img)
        flux_D = np.zeros_like(filtered_img)

        flux_N[:-1, :, :] = term_N
        flux_S[1:, :, :] = term_S # S se refiere a la dirección opuesta a N (hacia abajo)
        flux_E[:, :-1, :] = term_E
        flux_W[:, 1:, :] = term_W # W se refiere a la dirección opuesta a E (hacia la derecha)
        flux_U[:, :, :-1] = term_U
        flux_D[:, :, 1:] = term_D # D se refiere a la dirección opuesta a U (hacia la profundidad)

        # Suma de los flujos de "corriente" para actualizar la imagen
        # Este es el paso clave de la difusión: el valor de un voxel se actualiza
        # en función de la diferencia de intensidad con sus vecinos y la conductancia.
        filtered_img += gamma * (flux_N + flux_S + flux_E + flux_W + flux_U + flux_D)

    return filtered_img

# Parámetros para el filtro anisotrópico
n_iterations = 10     # Número de veces que se aplica el algoritmo
kappa_anisotropic = 30 # Valor que controla la sensibilidad al borde (ajustar según la imagen)
gamma_anisotropic = 0.05 # Tasa de aprendizaje (pequeña para estabilidad)

print(f"Aplicando filtro anisotrópico con {n_iterations} iteraciones, kappa={kappa_anisotropic}, gamma={gamma_anisotropic}...")
filtered_anisotropic_data = anisotropic_diffusion(data, 
                                                    niter=n_iterations, 
                                                    kappa=kappa_anisotropic, 
                                                    gamma=gamma_anisotropic)
print("Filtro anisotrópico aplicado.")

# Guardar la imagen con filtro anisotrópico
output_anisotropic_nifti_path = 'E:\ACTUAL\ACTUAL\P_D_Imagenes\Git-Hub\Imagenes_2025_1\DATA\data_taller2\Results\sub-0001_Anisotropic' # Cambia esto
anisotropic_img = nib.Nifti1Image(filtered_anisotropic_data, affine, header)
nib.save(anisotropic_img, output_anisotropic_nifti_path)
print(f"Imagen anisotrópica guardada en: {output_anisotropic_nifti_path}")

print("\nProceso completado. Revisa las imágenes generadas.")