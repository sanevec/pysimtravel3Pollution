import os
import numpy as np
from collections import defaultdict

# Parámetros de los experimentos
seeds = [12, 34, 56, 78]
buildings = [True, False]
distributionCS = [0, 1, 2]
densityCars = [0.05, 0.15, 0.25, 0.5]
densityEV = [0.05, 0.25, 0.45, 1, 0]
densityDiesel = [0.05, 0.35, 0.65, 1, 0]
windV = [
    (0, 0),
    (0.1, 0),
    (0.2, 0),
    (0.1, 0.1),
    (0.2, 0.1)
]
pollutionRouting = [False]#, True]

# Directorio donde se encuentran los archivos
directory = "simulationData32"

# Estructura para almacenar los resultados
experiments_count = defaultdict(int)
data_types = defaultdict(lambda: defaultdict(list))

# Función para verificar los tipos de datos en los archivos .npz
def check_data_types(filename, prefix):
    try:
        with np.load(filename) as data:
            for key in data.files:
                data_types[prefix][key].append(data[key].dtype)
    except Exception as e:
        print(f'Error al leer el archivo {filename}: {e}')

i=0
# Verificación de existencia de archivos y tipos de datos
for seed in seeds:
    for building in buildings:
        for distCS in distributionCS:
            for densCar in densityCars:
                for densEV in densityEV:
                    for densDiesel in densityDiesel:
                        for wind in windV:
                            for pollRoute in pollutionRouting:
                                if densEV + densDiesel > 1:
                                    continue
                                filenames = [
                                    os.path.join(directory, f'{prefix}_delta_{0.1}_gamma_{0.01}_times_{2000}_seed_{seed}_buildings_{building}_distributionCS_{distCS}_densityCars_{densCar}_densityEV_{densEV}_densityDiesel_{densDiesel}_windV_{wind}_pollutionRouting_{pollRoute}.npz')
                                    for prefix in ['P', 'G', 'A', 'C']
                                ]
                                aux = [os.path.exists(filename) for filename in filenames]
                                if any(aux) and not all(aux):
                                    print('Hay un error en ', filenames[0])
                                if all(aux):
                                    print(i)
                                    i+=1
                                    for filename, prefix in zip(filenames, ['P', 'G', 'A', 'C']):
                                        check_data_types(filename, prefix)
                                    experiments_count[f'seed_{seed}'] += 1
                                    experiments_count[f'buildings_{building}'] += 1
                                    experiments_count[f'distributionCS_{distCS}'] += 1
                                    experiments_count[f'densityCars_{densCar}'] += 1
                                    experiments_count[f'densityEV_{densEV}'] += 1
                                    experiments_count[f'densityDiesel_{densDiesel}'] += 1
                                    experiments_count[f'windV_{wind}'] += 1
                                    experiments_count[f'pollutionRouting_{pollRoute}'] += 1

# Verificar la consistencia de los tipos de datos
for prefix in ['P', 'G', 'A', 'C']:
    for key, types in data_types[prefix].items():
        if len(set(types)) > 1:
            print(f'Inconsistencia en el tipo de datos para {prefix} en la variable {key}: {set(types)}')

# Ordenar y imprimir resultados
params_order = [
    ('seed', seeds),
    ('buildings', buildings),
    ('distributionCS', distributionCS),
    ('densityCars', densityCars),
    ('densityEV', densityEV),
    ('densityDiesel', densityDiesel),
    ('windV', windV),
    ('pollutionRouting', pollutionRouting)
]

for param, values in params_order:
    for value in values:
        count = experiments_count[f'{param}_{value}']
        if count == 0:
            print(f'{param} = {value}: 0 experimentos (¡Advertencia: ningún experimento encontrado!)')
        else:
            print(f'{param} = {value}: {count} experimentos')
