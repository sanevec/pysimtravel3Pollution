import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

base_path = "simulationData32"
output_path = "representations"
if not os.path.exists(output_path):
    os.makedirs(output_path)

def load_file(prefix, params):
    # Build the filename based on the experiment parameters
    filename = (f"{prefix}_delta_{params['delta']}_gamma_{params['gamma']}_times_{params['times']}_seed_{params['seed']}_"
                f"buildings_{params['buildings']}_distributionCS_{params['distributionCS']}_densityCars_{params['densityCars']}_"
                f"densityEV_{params['densityEV']}_densityDiesel_{params['densityDiesel']}_windV_{params['windV']}_pollutionRouting_{params['pollutionRouting']}.npz")
    
    filepath = os.path.join(base_path, filename)
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    return np.load(filepath)

def calculate_total_pollution(data, pollutant, prefix, squares=False):
    total_pollution = np.zeros(list(data.values())[0].shape)
    if squares:
        sumWhere = np.zeros_like(total_pollution)
        l = list(range(0, 10)) + list(range(85, 106)) + list(range(181, 202)) + list(range(277, 287))
        for i in l:
            for j in l:
                sumWhere[i + 1, j + 1] = 1
    else:
        sumWhere = np.ones_like(total_pollution)
    for key in data.keys():
        if prefix == 'A':
            prefix2 = 'P'
            if pollutant in key and prefix2 in key:
                total_pollution += sumWhere * data[key] 
        else:           
            if pollutant in key:
                total_pollution += sumWhere * data[key]
    return total_pollution

def compute_metrics(data1, data2):
    # Flatten the data for comparison
    flat_data1 = data1.flatten()
    flat_data2 = data2.flatten()
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(flat_data1 - flat_data2))
    
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((flat_data1 - flat_data2) ** 2))
    
    # Pearson Correlation Coefficient
    correlation, _ = pearsonr(flat_data1, flat_data2)
    
    return mae, rmse, correlation

def process_experiments():
    seeds = [12, 34, 1, 78]
    buildings = [True, False]
    distributionCS = [0, 1, 2]
    densityCars = [0.05, 0.15, 0.25]
    densityEV = [0.05, 0.25, 0.45]
    densityDiesel = [0.05, 0.35, 0.65]
    windV = [
        (0, 0), 
        (np.float32(0.1), 0),
        (np.float32(0.2), 0),
        (np.float32(0.1), np.float32(0.1)),
        (np.float32(0.2), np.float32(0.1))
    ]
    pollutionRouting = [False]
    pollutants = ['CO2', 'NOx', 'VOC', 'PMexhaust', 'PMnonexhaust25', 'PMnonexhaust10']

    params = {
        'delta': 0.1,
        'gamma': 0.01,
        'times': 2000,
        'seed': 1,
        'buildings': True,
        'distributionCS': 1,
        'densityCars': 0.25,
        'densityEV': 0.45,
        'densityDiesel': 0.35,
        'windV': "(0, 0)",
        'pollutionRouting': False
    }
    for squares in [True, False]:
        for pollutant in pollutants:
            plt.figure()
            for dDiesel in densityDiesel:
                means = []
                stds = []
                for dEV in densityEV:
                    if dEV + dDiesel <= 1:
                        total_pollution = []
                        for seed in seeds:
                            params2 = params.copy()
                            params2['seed'] = seed
                            params2['densityEV'] = dEV
                            params2['densityDiesel'] = dDiesel
                            prefix = 'P'
                            data = load_file(prefix, params2)
                            if data is not None:
                                if prefix == 'A':
                                    total_pollution.append(np.sum(calculate_total_pollution(data, pollutant, prefix, squares=squares)))
                                else:
                                    total_pollution.append(np.sum(calculate_total_pollution(data, pollutant, prefix, squares=squares)[:, :, -1]))
                        if total_pollution:
                            means.append(np.mean(total_pollution))
                            stds.append(np.std(total_pollution))
                        else:
                            means.append(0)
                            stds.append(0)
                
                plt.errorbar([dd for dd in densityEV if dd + dDiesel<=1], means, yerr=stds, label=f'densityDiesel={dDiesel}', marker='o')

            plt.xlabel('Density EV')
            plt.ylabel('Total Pollution')
            plt.title(f"{pollutant} Total Pollution\n"
                      f"Squares: {squares}, DistributionCS: {params['distributionCS']}, "
                      f"DensityCars: {params['densityCars']}, WindV: {params['windV']}, Buildings: {params['buildings']}", wrap=True)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            output_filename = os.path.join(output_path, f"{prefix}_{pollutant}_squares_{squares}_averageSeed_distributionCS_{params['distributionCS']}_densityCars_{params['densityCars']}_windV_{params['windV']}_buildings_{params['buildings']}.png")
            plt.savefig(output_filename)
            plt.close()

process_experiments()
