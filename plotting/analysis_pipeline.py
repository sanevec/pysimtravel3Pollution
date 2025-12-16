import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

base_path = "simulationData32Last"
output_path = "representations2"
if not os.path.exists(output_path):
    os.makedirs(output_path)


pollutants = ['CO2', 'NOx', 'VOC', 'PMexhaust', 'PMnonexhaust25', 'PMnonexhaust10']

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
    for key in data.keys():
        if prefix == 'A':
            if pollutant in key and 'P' in key:
                total_pollution += data[key] 
        else:           
            if pollutant in key:
                total_pollution += data[key]
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
    print(len(flat_data1), len(flat_data2))
    
    # Identify nonzero elements in both data1 and data2
    #if ((flat_data1 !=0)!=(flat_data2!=0)).any():
    #    print(len(flat_data1 != 0)-len(flat_data2 != 0))
    nonzero_mask = (flat_data1 != 0) & (flat_data2 != 0)
    
    # Apply the mask to filter nonzero elements
    filtered_data1 = flat_data1[nonzero_mask]
    filtered_data2 = flat_data2[nonzero_mask]
    print(len(filtered_data1), len(filtered_data2))
    
    # Mean Absolute Error (MAE) for nonzero elements
    mae = np.mean(np.abs(filtered_data1 - filtered_data2))
    
    # Pearson Correlation Coefficient for nonzero elements
    correlation, _ = pearsonr(filtered_data1, filtered_data2)
    
    # Relative Error (relative to the mean of filtered_data1)
    mean_data1 = np.mean(filtered_data1)
    #relative_error = mae / mean_data1
    relative_error = np.mean(np.abs(filtered_data1 - filtered_data2)/filtered_data1)
    
    return relative_error, correlation

def process_experiments():
    seeds = [12, 34, 1, 78]
    heatmaps = []

    params = {
        'delta': 0.1,
        'gamma': 0.01,
        'times': 2000,
        'seed': 1,
        'buildings': False,
        'distributionCS': 0,
        'densityCars': 0.25,
        'densityEV': 0.35,
        'densityDiesel': 0.25,
        'windV': "(0.5, 0.3)",
        'pollutionRouting': False
    }
    heatmaps = {}
    for seed in seeds:
        params2 = params.copy()
        params2['seed'] = seed
        prefix='A'
        data = load_file(prefix, params2)
        if data is not None:
            #pollutant = 'CO2'
            for pollutant in pollutants:
                total_pollution = calculate_total_pollution(data, pollutant,prefix)#[:,:,-1]
                output_filename = os.path.join(output_path, f"{pollutant}_total_seed_{seed}.png")
                #generate_heatmap(total_pollution, output_filename)
                if len(heatmaps)==0:
                    heatmaps = {(seed, pollutant) : total_pollution}
                else:
                    heatmaps[(seed, pollutant)] = total_pollution

    # Compare each pair of heatmaps
    results = []
    for i1 in range(len(seeds)):
        for i2 in range(len(seeds)):
            for j in range(len(pollutants)):
                if i2>i1:
                    #mae, rmse, correlation = compute_metrics(heatmaps[(seeds[i1], pollutants[j])], heatmaps[(seeds[i2], pollutants[j])])
                    relative_error, correlation = compute_metrics(heatmaps[(seeds[i1], pollutants[j])], heatmaps[(seeds[i2], pollutants[j])])
                    #results.append((seeds[i], seeds[j], mae, rmse, correlation))
                    #print(f"Comparison between seed {seeds[i1]} and seed {seeds[i2]} in {pollutants[j]}: relative error = {relative_error}, Correlation = {correlation},")# Means = {np.mean(heatmaps[(seeds[i1],pollutants[j])][heatmaps[(seeds[i1],pollutants[j])]])}; {np.mean(heatmaps[(seeds[i2],pollutants[j])][heatmaps[(seeds[i2],pollutants[j])]])}, Max = {np.max(heatmaps[(seeds[i1],pollutants[j])])}; {np.max(heatmaps[(seeds[i2],pollutants[j])])}, Relative error = {mae/np.mean(heatmaps[(seeds[i1],pollutants[j])])}")
                    flat_data1 = heatmaps[(seeds[i1], pollutants[j])].flatten()
                    flat_data2 = heatmaps[(seeds[i2], pollutants[j])].flatten()
                    
                    # Identify nonzero elements in both data1 and data2
                    #if ((flat_data1 !=0)!=(flat_data2!=0)).any():
                    #    print(len(flat_data1 != 0)-len(flat_data2 != 0))
                    nonzero_mask = (flat_data1 != 0) & (flat_data2 != 0)
                    
                    # Apply the mask to filter nonzero elements
                    filtered_data1 = flat_data1[nonzero_mask]
                    filtered_data2 = flat_data2[nonzero_mask]
                    #print(f"Comparison between seed {seeds[i1]} and seed {seeds[i2]} in {pollutants[j]}: relative error = {relative_error}, Correlation = {correlation},")
                    print(f"Comparison between seed {seeds[i1]} and seed {seeds[i2]} in {pollutants[j]}: relative error = {100*np.abs(np.mean(filtered_data1)-np.mean(filtered_data2))/np.mean(filtered_data1)}, total relative error = {relative_error}, correlation = {correlation},")
                    #print(f"Comparison in {pollutants[j]}. Seed {seeds[i1]} maximum: {np.max(filtered_data1)}, and seed {seeds[i2]} maximum:{np.max(filtered_data2)}")
                    #print(f"Comparison in {pollutants[j]}. Seed {seeds[i1]} mean: {np.mean(filtered_data1)}, and seed {seeds[i2]} mean:{np.mean(filtered_data2)}")
process_experiments()


'''
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

delta=0.1
gamma=0.01
times=2000
squares=list(range(0,10))+list(range(85,106))+list(range(181,202))+list(range(277,287))


parameters = {
    'seed': [12, 34, 1, 78],
    'buildings': [True, False],
    'distributionCS': [0, 1, 2],
    'densityCars': [0.05, 0.15, 0.25],
    'densityEV': [0.05, 0.25, 0.45],
    'densityDiesel': [0.05, 0.35, 0.65],
    'windV': [(0, 0), (0.1, 0), (0.2, 0), (0.1, 0.1), (0.2, 0.1)],
    'pollutionRouting': [False]
}

prefixes = {'P', 'G', 'A', 'C'}

parameter_names = list(parameters.keys())
parameter_values = list(parameters.values())
tuples_list = list(itertools.product(*parameter_values))
tuples_list = [tup for tup in tuples_list if tup[parameter_names.index('densityEV')] + tup[parameter_names.index('densityDiesel')] <= 1]


def load_file(prefix, params):
    # Construir el nombre del archivo basado en los parámetros del experimento
    filename = f"{prefix}_delta_{params['delta']}_gamma_{params['gamma']}_times_{params['times']}_seed_{params['seed']}_buildings_{params['buildings']}_distributionCS_{params['distributionCS']}_densityCars_{params['densityCars']}_densityEV_{params['densityEV']}_densityDiesel_{params['densityDiesel']}_windV_{params['windV']}_pollutionRouting_{params['pollutionRouting']}.npz"
    
    filepath = os.path.join("simulationData32", filename)
    
    if not os.path.exists(filepath):
        print(f"Archivo no encontrado: {filepath}")
        return

    return np.load(filepath)

tuples_buildings =[tup for tup in tuples_list if tup[parameter_names.index('buildings')]]
tuples_no_buildings =[tup for tup in tuples_list if not tup[parameter_names.index('buildings')]]
def create_filename(t,prefix='P'):
    if prefix not in prefixes:
        raise ValueError(f"Invalid prefix: {prefix}")
    windV_str = f"({t[6][0]}, {t[6][1]})"
    return os.path.join("simulationData32", f"{prefix}_delta_{delta}_gamma_{gamma}_times_{times}_seed_{t[0]}_buildings_{t[1]}_distributionCS_{t[2]}_densityCars_{t[3]}_densityEV_{t[4]}_densityDiesel_{t[5]}_windV_{windV_str}_pollutionRouting_{t[7]}.npz")
'''
