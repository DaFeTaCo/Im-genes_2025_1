## Para la Difusión Anisotrópica
 #       output_anisotropic_filename = f"{base_filename}_desc-anisoDiffused.nii.gz"
 #       output_anisotropic_path = os.path.join(output_derivatives_dir, output_anisotropic_filename)
 #       processed_anisotropic_img = nib.Nifti1Image(anisotropic_diffused_data, affine, header=header)
 #       nib.save(processed_anisotropic_img, output_anisotropic_path)
 #       print(f"  Guardado Anisotrópico en: {os.path.basename(output_anisotropic_path)}")
#

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

#pip install bids
from bids import BIDSLayout
import nibabel as nib
import os

#Ruta del dataset BIDS
bids_path = r"C:\Users\Isabela\Documents\2025-1\Imagenes\Imagenes_2025_1\DATA\data_taller2\BIDS\BIDS"  # Cambia esto a la ruta de tu dataset BIDS

try:
    layout = BIDSLayout(bids_path, validate=True)
    print(f"BIDS dataset cargado y escaneado desde: {bids_path}")
    print(f"Número total de archivos en el layout: {len(layout.get_files())}")
except Exception as e:
    print(f"Error al cargar el dataset BIDS. Asegúrate de que la ruta sea correcta y el dataset sea conforme a BIDS.")
    print(f"Detalles del error: {e}")
    exit()

imagenes_para_filtrar = layout.get(extension=['.nii.gz', '.nii'], return_type='object')

if not imagenes_para_filtrar:
    print("No se encontraron imágenes NIfTI en la estructura BIDS que coincidan con los criterios.")
    exit()

print(f"\nEncontradas {len(imagenes_para_filtrar)} imágenes para procesar.")

#Ruta de salida para las imagenes procesadas con filtro Gaussiano
salida_dir_G = os.path.join(bids_path, 'ResultadosG', 'imagenes_filtradas')
os.makedirs(salida_dir_G, exist_ok=True)
print(f"Los resultados del filtro Gaussiano se guardarán en: {salida_dir_G}")

#Ruta de salida para las imagenes procesadas con filtro anisotrópico
salida_dir_A = os.path.join(bids_path, 'ResultadosA', 'imagenes_filtradas')
os.makedirs(salida_dir_A, exist_ok=True)
print(f"Los resultados del filtro Anisotrópico se guardarán en: {salida_dir_A}")

for bids_file in imagenes_para_filtrar:
    original_filepath = bids_file.path
    print(f"\nProcesando imagen: {os.path.basename(original_filepath)}")

    try:
        # Cargar la imagen con nibabel (pybids lo hace por ti con .get_image())
        img = bids_file.get_image()
        data = img.get_fdata()
        affine = img.affine
        header = img.header

        print(f"  Forma original: {data.shape}")

        #Filtro Gaussiano
        sigma_gaussian = [0.5, 1.0, 1.5]  # Ejemplo de diferentes valores de sigma
        for sigma in sigma_gaussian:
            print(f"  Aplicando filtro gaussiano con sigma={sigma}...")
            filtered_gaussian_data = gaussian_filter(data, sigma=sigma)
            gaussian_img = nib.Nifti1Image(filtered_gaussian_data, affine, header)
            print("  Filtro gaussiano aplicado.")
            # Guardar la imagen filtrada

            base_filename = os.path.basename(original_filepath).replace('.nii.gz', '').replace('.nii', '')
            # Crear un nombre de archivo para la imagen filtrada con el sigma aplicado
            # Ejemplo: sub-0001_T1w_desc-FGaussiano_sigma1.5.nii.gz
            nombre_salida_F_Gaussiano = f"{base_filename}_desc-FGaussiano_sigma{sigma}.nii.gz"
            path_salida_F_Gaussiano = os.path.join(salida_dir_G, nombre_salida_F_Gaussiano)
            processed_gaussian_img = nib.Nifti1Image(filtered_gaussian_data, affine, header=header)
            nib.save(processed_gaussian_img, path_salida_F_Gaussiano)
            print(f"  Guardado Gaussiano en: {os.path.basename(path_salida_F_Gaussiano)}")

        #Filtro Anisotrópico (Perona-Malik)
        
        def anisotropic_diffusion(img, niter=10, kappa=50, gamma=0.1, step=(1.,1.,1.), option=1):
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
        
        #Parámetros para el filtro anisotrópico
        n_iteraciones= [10, 20]  
        kappa = [30, 50]
        gamma = [0.5, 0.1]  # Valor constante para el paso de tiempo
        step = (1.0, 1.0, 1.0)  # Paso en cada dimensión
        option = 1  # Opción de Perona-Malik

        print(f"Aplicando filtro anisotrópico con {n_iteraciones[1]} iteraciones, kappa={kappa[1]}, gamma={gamma[1]}...")
        filtered_anisotropic_data = anisotropic_diffusion(data, niter=n_iteraciones[1], kappa=kappa[1], gamma=gamma[1], step=step, option=option)

        anisotropic_img = nib.Nifti1Image(filtered_anisotropic_data, affine, header)
        print("  Filtro anisotrópico aplicado.")

        # Guardar la imagen filtrada
        base_filename = os.path.basename(original_filepath).replace('.nii.gz', '').replace('.nii', '')
        # Crear un nombre de archivo para la imagen filtrada anisotrópica
        nombre_salida_F_Anisotropico = f"{base_filename}_desc-FAnisotropico_n{n_iteraciones[1]}_k{kappa[1]}_g{gamma[1]}.nii.gz"
        path_salida_F_Anisotropico = os.path.join(salida_dir_A, nombre_salida_F_Anisotropico)
        processed_anisotropic_img = nib.Nifti1Image(filtered_anisotropic_data, affine, header=header)
        nib.save(processed_anisotropic_img, path_salida_F_Anisotropico)
        print(f"  Guardado Anisotrópico en: {os.path.basename(path_salida_F_Anisotropico)}")

        #Variando número de iteraciones
        print(f"Aplicando filtro anisotrópico con {n_iteraciones[0]} iteraciones, kappa={kappa[1]}, gamma={gamma[1]}...")
        filtered_anisotropic_data = anisotropic_diffusion(data, niter=n_iteraciones[0], kappa=kappa[1], gamma=gamma[1], step=step, option=option)

        anisotropic_img = nib.Nifti1Image(filtered_anisotropic_data, affine, header)
        print("  Filtro anisotrópico aplicado.")

        # Guardar la imagen filtrada
        base_filename = os.path.basename(original_filepath).replace('.nii.gz', '').replace('.nii', '')
        # Crear un nombre de archivo para la imagen filtrada anisotrópica
        nombre_salida_F_Anisotropico = f"{base_filename}_desc-FAnisotropico_n{n_iteraciones[0]}_k{kappa[1]}_g{gamma[1]}.nii.gz"
        path_salida_F_Anisotropico = os.path.join(salida_dir_A, nombre_salida_F_Anisotropico)
        processed_anisotropic_img = nib.Nifti1Image(filtered_anisotropic_data, affine, header=header)
        nib.save(processed_anisotropic_img, path_salida_F_Anisotropico)
        print(f"  Guardado Anisotrópico en: {os.path.basename(path_salida_F_Anisotropico)}")

        #Variando kappa
        print(f"Aplicando filtro anisotrópico con {n_iteraciones[1]} iteraciones, kappa={kappa[0]}, gamma={gamma[1]}...")
        filtered_anisotropic_data = anisotropic_diffusion(data, niter=n_iteraciones[1], kappa=kappa[0], gamma=gamma[1], step=step, option=option)

        anisotropic_img = nib.Nifti1Image(filtered_anisotropic_data, affine, header)
        print("  Filtro anisotrópico aplicado.")

        # Guardar la imagen filtrada
        base_filename = os.path.basename(original_filepath).replace('.nii.gz', '').replace('.nii', '')
        # Crear un nombre de archivo para la imagen filtrada anisotrópica
        nombre_salida_F_Anisotropico = f"{base_filename}_desc-FAnisotropico_n{n_iteraciones[1]}_k{kappa[0]}_g{gamma[1]}.nii.gz"
        path_salida_F_Anisotropico = os.path.join(salida_dir_A, nombre_salida_F_Anisotropico)
        processed_anisotropic_img = nib.Nifti1Image(filtered_anisotropic_data, affine, header=header)
        nib.save(processed_anisotropic_img, path_salida_F_Anisotropico)
        print(f"  Guardado Anisotrópico en: {os.path.basename(path_salida_F_Anisotropico)}")     

        #Variando gamma
        #Variando kappa
        print(f"Aplicando filtro anisotrópico con {n_iteraciones[1]} iteraciones, kappa={kappa[1]}, gamma={gamma[0]}...")
        filtered_anisotropic_data = anisotropic_diffusion(data, niter=n_iteraciones[1], kappa=kappa[1], gamma=gamma[0], step=step, option=option)

        anisotropic_img = nib.Nifti1Image(filtered_anisotropic_data, affine, header)
        print("  Filtro anisotrópico aplicado.")

        # Guardar la imagen filtrada
        base_filename = os.path.basename(original_filepath).replace('.nii.gz', '').replace('.nii', '')
        # Crear un nombre de archivo para la imagen filtrada anisotrópica
        nombre_salida_F_Anisotropico = f"{base_filename}_desc-FAnisotropico_n{n_iteraciones[1]}_k{kappa[1]}_g{gamma[0]}.nii.gz"
        path_salida_F_Anisotropico = os.path.join(salida_dir_A, nombre_salida_F_Anisotropico)
        processed_anisotropic_img = nib.Nifti1Image(filtered_anisotropic_data, affine, header=header)
        nib.save(processed_anisotropic_img, path_salida_F_Anisotropico)
        print(f"  Guardado Anisotrópico en: {os.path.basename(path_salida_F_Anisotropico)}")       

    except Exception as e:
        print(f"  ERROR al procesar {os.path.basename(original_filepath)}: {e}")

print("\n¡Procesamiento de imágenes completado!")

#Disminuir numero de iteraciones: poner entre 5 y 10 (Se demora mucho con 20)