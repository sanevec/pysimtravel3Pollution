import os
import numpy as np
import matplotlib.pyplot as plt

# Parámetros de ejemplo para el experimento
params = {
    'delta': 0.1,
    'gamma': 0.01,
    'times': 2000,
    'seed': 34,
    'buildings': True,
    'distributionCS': 1,
    'densityCars': 0.25,
    'densityEV': 0.35,
    'densityDiesel': 0.5,
    'windV': (0.1, 0),
    'pollutionRouting': False
}

def load_and_generate_maps(prefix, params):
    # Construir el nombre del archivo basado en los parámetros del experimento
    filename = f"{prefix}_delta_{params['delta']}_gamma_{params['gamma']}_times_{params['times']}_seed_{params['seed']}_buildings_{params['buildings']}_distributionCS_{params['distributionCS']}_densityCars_{params['densityCars']}_densityEV_{params['densityEV']}_densityDiesel_{params['densityDiesel']}_windV_{params['windV']}_pollutionRouting_{params['pollutionRouting']}.npz"
    
    filepath = os.path.join("simulationData32Last", filename)
    
    if not os.path.exists(filepath):
        print(f"Archivo no encontrado: {filepath}")
        return

    data = np.load(filepath)
    
    if prefix == 'A':
        output_dir = 'representaciones'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for key in data:
            # Crear un mapa de color para cada clave
            plt.figure(figsize=(10, 8))
            plt.imshow(np.transpose(data[key]), cmap='viridis')
            plt.colorbar()
            plt.title(f'{prefix} - {key}')
            
            # Guardar la figura
            output_filename = f"{prefix}_delta_{params['delta']}_gamma_{params['gamma']}_times_{params['times']}_seed_{params['seed']}_buildings_{params['buildings']}_distributionCS_{params['distributionCS']}_densityCars_{params['densityCars']}_densityEV_{params['densityEV']}_densityDiesel_{params['densityDiesel']}_windV_{params['windV']}_pollutionRouting_{params['pollutionRouting']}_{key}.png"
            output_filepath = os.path.join(output_dir, output_filename)
            plt.savefig(output_filepath)
            plt.close()

    print(f"Archivos guardados en la carpeta {output_dir}")

# Ejemplo de uso
load_and_generate_maps('A', params)
