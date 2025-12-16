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

def calculate_total_pollution(data, pollutant,prefix):
    total_pollution = np.zeros(list(data.values())[0].shape)
    squares=True
    if squares:
        sumWhere = np.zeros_like(total_pollution)
        l=list(range(0,10))+list(range(85,106))+list(range(181,202))+list(range(277,287))
        for i in l:
            for j in l:
                sumWhere[i+1,j+1]=1
    else:
        sumWhere = np.ones_like(total_pollution)
    for key in data.keys():
        if prefix == 'A':
            if pollutant in key and 'P' in key:
                total_pollution += sumWhere*data[key] 
        else:           
            if pollutant in key:
                total_pollution += sumWhere*data[key]
    return total_pollution

def generate_heatmap(total_pollution, output_filename):
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(total_pollution), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Total CO2 Heatmap")
    plt.savefig(output_filename)
    plt.close()

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
    buildings=[True,False]
    distributionCS=[0,1,2]
    densityCars=[0.05, 0.15, 0.25]
    densityEV=[0.05, 0.25, 0.45]
    densityDiesel=[0.05, 0.35, 0.65]
    windV = [
    (0, 0), 
    (np.float32(0.1), 0),
    (np.float32(0.2), 0),
    (np.float32(0.1), np.float32(0.1)),
    (np.float32(0.2), np.float32(0.1))
    ]
    pollutionRouting=[False]
    heatmaps = {}

    params = {
        'delta': 0.1,
        'gamma': 0.01,
        'times': 2000,
        'seed': 1,
        'buildings': True,
        'distributionCS': 0,
        'densityCars': 0.25,
        'densityEV': 0.45,
        'densityDiesel': 0.35,
        'windV': "(0, 0)",
        'pollutionRouting': False
    }
    for distr in distributionCS:
        total_heatmap = None
        count = 0
        for seed in seeds:
            params2 = params.copy()
            params2['seed'] = seed
            params2['distributionCS'] = distr
            prefix = 'P'
            data = load_file(prefix, params2)
            if data is not None:
                pollutant = 'CO2'
                if prefix == 'A':
                    total_pollution = calculate_total_pollution(data, pollutant,prefix)
                else:
                    total_pollution = calculate_total_pollution(data, pollutant,prefix)[:,:,-1]
                if total_heatmap is None:
                    total_heatmap = total_pollution
                else:
                    total_heatmap += total_pollution
                count += 1
        if total_heatmap is not None:
            average_heatmap = total_heatmap / count
            output_filename = os.path.join(output_path, f"{prefix}_{pollutant}_squares_averageSeed_distributionCS_{distr}_densityCars_{params2['densityCars']}_densityEV_{params2['densityEV']}_densityDiesel_{params2['densityDiesel']}_windV_{params2['windV']}_buildings_{params2['buildings']}.png")
            generate_heatmap(average_heatmap, output_filename)
            heatmaps[distr] = average_heatmap


    # Compare each pair of heatmaps
    results = []
    for i in range(len(heatmaps)):
        for j in range(i+1, len(heatmaps)):
            mae, rmse, correlation = compute_metrics(heatmaps[i], heatmaps[j])
            results.append((distributionCS[i], distributionCS[j], mae, rmse, correlation))
            print(f"Comparison between distributions {distributionCS[i]} and {distributionCS[j]}: MAE = {mae}, RMSE = {rmse}, Correlation = {correlation}, Means = {np.mean(heatmaps[i][heatmaps[i]!=0])}; {np.mean(heatmaps[j][heatmaps[j]!=0])}, Max = {np.max(heatmaps[i])}; {np.max(heatmaps[j])}, Relative error = {mae/np.mean(heatmaps[i])}")

process_experiments()
