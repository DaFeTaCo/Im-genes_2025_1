import nibabel as nib
from bids.layout import BIDSLayout
import json
import pandas as pd # Usaremos pandas para una comparación más fácil

# Ruta a la raíz de dataset BIDS
bids_root_path = 'E:/ACTUAL/ACTUAL/P_D_Imagenes/Git-Hub/Imagenes_2025_1/DATA/data_taller2/BIDS' 

# Lista para almacenar los parámetros de todas las adquisiciones
all_acquisitions_data = []

# ---
## Procesamiento de Datos BIDS
# ---

try:
    # Inicializa el layout BIDS. validate=False puede ser útil si tu dataset no es 100% compliant.
    layout = BIDSLayout(bids_root_path, validate=False)

    # Define los sujetos que quieres procesar
    # Ejemplo: Sujetos '01' y '02' con dos adquisiciones (ej. T1w y flair)
    # Sujetos '03', '04', '05' con una adquisición (ej. solo T1w)
    # Ajusta esta lista según los sujetos y adquisiciones que realmente tengas.
    subjects_to_process = ['0001', '0002', '0003', '0004', '0005']
    acquisition_types_sub1_2 = ['T1w', 'FLAIR'] # Ejemplo de 2 adquisiciones
    acquisition_types_sub3_5 = ['T2w','T1w'] # Ejemplo de 1 adquisición

    for subject_id in subjects_to_process:
        print(f"\n--- Procesando Sujeto: {subject_id} ---")

        # Determinar qué tipos de adquisición buscar para el sujeto actual
        if subject_id in ['0001', '0002']:
            acquisitions_to_look_for = acquisition_types_sub1_2
        else:
            acquisitions_to_look_for = acquisition_types_sub3_5

        for acq_type in acquisitions_to_look_for:
            print(f"Buscando adquisición: {acq_type}")

            # Obtener archivos de RM para el sujeto y tipo de adquisición
            # Puedes añadir 'session' si tu dataset tiene sesiones (e.g., session='01')
            mri_files = layout.get(
                subject=subject_id,
                suffix=acq_type,
                extension=['.nii.gz'],
                return_type='object' # Asegura que devuelve objetos BIDSFile
            )

            if mri_files:
                for mri_file_obj in mri_files:
                    nifti_path = mri_file_obj.path
                    print(f"  Encontrado archivo: {nifti_path}")

                    try:
                        img = nib.load(nifti_path)
                        metadata = layout.get_metadata(nifti_path)

                        # Extraer resolución, dimensiones y FOV
                        voxel_dims = img.header['pixdim'][1:4]
                        image_dims = img.header['dim'][1:4]
                        fov_x = image_dims[0] * voxel_dims[0]
                        fov_y = image_dims[1] * voxel_dims[1]
                        fov_z = image_dims[2] * voxel_dims[2]
                        # Preparar los datos para esta adquisición
                        acquisition_data = {
                            'subject_id': subject_id,
                            'acquisition_type': acq_type,
                            'nifti_path': nifti_path,
                            'resolution_x_mm': float(f'{voxel_dims[0]:.2f}'),
                            'resolution_y_mm': float(f'{voxel_dims[1]:.2f}'),
                            'resolution_z_mm': float(f'{voxel_dims[2]:.2f}'),
                            'image_dims_x_voxels': int(image_dims[0]),
                            'image_dims_y_voxels': int(image_dims[1]),
                            'image_dims_z_voxels': int(image_dims[2]),
                            
                            'fov_x_mm': float(f'{fov_x:.2f}'),
                            'fov_y_mm': float(f'{fov_y:.2f}'),
                            'fov_z_mm': float(f'{fov_z:.2f}'),
                            # Parámetros de adquisición (del JSON)
                            'RepetitionTime': metadata.get('RepetitionTime'),
                            'EchoTime': metadata.get('EchoTime'),
                            'FlipAngle': metadata.get('FlipAngle'),
                            'SequenceType': metadata.get('SequenceType'),
                            'Manufacturer': metadata.get('Manufacturer'),
                            'MagneticFieldStrength': metadata.get('MagneticFieldStrength'),
                            'PhaseEncodingDirection': metadata.get('PhaseEncodingDirection'),
                            'SliceThickness': metadata.get('SliceThickness'), 
                            'SpacingBetweenSlices': metadata.get('SpacingBetweenSlices'), # Campo RECOMENDADO en BIDS
                            'PhotometricInterpretation': metadata.get('PhotometricInterpretation'), # Puede estar presente si tu conversor DICOM lo incluyó

                            # 'otros_param': metadata.get('OtroParametro')
                        }

                        # Añadir metadatos de calidad si existen (MRIQC u otros)
                        # Este es un ejemplo; los nombres de los campos varían.
                        # Puedes añadir más campos buscando en `metadata`.
                        if 'bids_mriqc_info' in metadata: # Ejemplo de un campo hipotético
                             acquisition_data['mriqc_snr'] = metadata['bids_mriqc_info'].get('snr_total')
                        # Puedes iterar sobre el diccionario metadata para encontrar otros campos relevantes de calidad
                        for key, value in metadata.items():
                            if "quality" in key.lower() or "qc" in key.lower() or "snr" in key.lower() or "cnr" in key.lower() or "fd_rms" in key.lower():
                                acquisition_data[f'json_qc_{key}'] = value

                        all_acquisitions_data.append(acquisition_data)

                    except FileNotFoundError:
                        print(f"  Error: Archivo NIfTI no encontrado en {nifti_path}")
                    except Exception as e:
                        print(f"  Ocurrió un error al procesar {nifti_path}: {e}")
            else:
                print(f"  No se encontraron archivos {acq_type} para el sujeto {subject_id}.")

except Exception as e:
    print(f"Ocurrió un error al inicializar BIDSLayout o durante la iteración de sujetos: {e}")
    print("Asegúrate de que 'bids_root_path' apunte a la raíz de tu dataset BIDS y que la estructura sea válida.")

# ---
## Comparación y Visualización de Datos
# ---

if all_acquisitions_data:
    # Convertir la lista de diccionarios a un DataFrame de pandas para una fácil comparación
    df = pd.DataFrame(all_acquisitions_data)

    print("\n--- Resumen de Datos Extraídos ---")
    #print(df.to_string()) # to_string() para ver todas las filas y columnas sin truncar
        # ---
    ## Ejemplos de Comparación
    # ---

    print("\n--- Comparaciones Específicas ---")

    #OTROS
    print("\nOTROS")
    print(df[['subject_id','Manufacturer','SliceThickness','SpacingBetweenSlices','PhotometricInterpretation']])
    
    print("\nDimensiones de la Imagen (voxels) :")
    print(df[['subject_id', 'acquisition_type', 'image_dims_x_voxels', 'image_dims_y_voxels', 'image_dims_z_voxels']])
    
    # Comparar la resolución para todas las adquisiciones
    print("\nResolución por Sujeto y Adquisición:")
    print(df[['subject_id', 'acquisition_type', 'resolution_x_mm', 'resolution_y_mm', 'resolution_z_mm']])

    # Comparar el FOV
    print("\nFOV por Sujeto y Adquisición:")
    print(df[['subject_id', 'acquisition_type', 'fov_x_mm', 'fov_y_mm', 'fov_z_mm']])

    # Comparar tiempos de adquisición (TR, TE)
    print("\nTiempos de Adquisición (TR, TE) por Sujeto y Adquisición:")
    print(df[['subject_id', 'acquisition_type', 'RepetitionTime', 'EchoTime']])

    # Comparar un parámetro específico (ej. FlipAngle)
    print("\nÁngulo de Flip por Sujeto y Adquisición:")
    print(df[['subject_id', 'acquisition_type', 'FlipAngle']])

    # Puedes agrupar y describir datos para ver estadísticas
    print("\nEstadísticas de Resolución (Agrupadas por tipo de adquisición):")
    print(df.groupby('acquisition_type')[['resolution_x_mm', 'resolution_y_mm', 'resolution_z_mm']].describe())

    # Filtrar y comparar adquisiciones específicas, por ejemplo, T1w de todos los sujetos
    print("\nParámetros de todas las adquisiciones T1w:")
    t1w_data = df[df['acquisition_type'] == 'T1w']
    print(t1w_data[['subject_id', 'RepetitionTime', 'EchoTime', 'FlipAngle', 'MagneticFieldStrength']])

    # Comparar la calidad (si se extrajeron métricas de QC)
    # Asume que 'json_qc_...' son las columnas para métricas de calidad.
    qc_columns = [col for col in df.columns if col.startswith('json_qc_')]
    if qc_columns:
        print("\nMétricas de Calidad (si están presentes):")
        print(df[['subject_id', 'acquisition_type'] + qc_columns])
    else:
        print("\nNo se encontraron métricas de calidad explícitas en los JSON.")


else:
    print("No se extrajeron datos de ninguna adquisición.")

#%%


# %%
