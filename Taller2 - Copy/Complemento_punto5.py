import pydicom
import os

dicom_file_example = 'E:\ACTUAL\ACTUAL\P_D_Imagenes\Git-Hub\Imagenes_2025_1\DATA\data_taller2\SORTER\Series0301_FLAIR_SAG_3D\Image (0003).dcm'
try:                    
    # Cargar el archivo DICOM
    ds = pydicom.dcmread(dicom_file_example)

    # Acceder al campo PhotometricInterpretation
    # El tag DICOM es (0028, 0004). Pydicom permite accederlo como atributo
    # si es un tag conocido, o usando .get((grupo, elemento))
    photometric_interpretation = ds.get((0x0028, 0x0004))
    # O de forma más directa, si está presente:
    # photometric_interpretation = ds.PhotometricInterpretation

    if photometric_interpretation:
        print(f"Photometric Interpretation (desde DICOM): {photometric_interpretation.value}")
    else:
        print("Photometric Interpretation no encontrado en la cabecera DICOM de este archivo.")

    # Puedes imprimir todo el dataset para ver otros tags si no estás seguro del nombre:
    # print("\n--- Cabecera DICOM completa ---")
    # print(ds)
    # print("-----------------------------")

except FileNotFoundError:
    print(f"Error: El archivo DICOM no se encontró en {dicom_file_example}")
except Exception as e:
    print(f"Ocurrió un error al leer el archivo DICOM: {e}")