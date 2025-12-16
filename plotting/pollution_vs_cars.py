import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

base_path = "simulationData32Last"
output_path = "representations2"
if not os.path.exists(output_path):
    os.makedirs(output_path)


pollutants = ['CO2', 'NOx', 'VOC', 'PMexhaust', 'PMnonexhaust25', 'PMnonexhaust10']
properties = ['positions', 'velocities', 'accelerations', 'braking']

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
        #if prefix == 'A':
            if pollutant in key and prefix in key:
                total_pollution += data[key] 
        #else:           
        #    if pollutant in key:
        #        total_pollution += data[key]
    return total_pollution

def calculate_total_occupation(data, carprop):
    total_pollution = np.zeros(list(data.values())[0].shape)
    for key in data.keys():
        if carprop in key:
            total_pollution += data[key]
    return total_pollution

def generate_heatmap(total_pollution, output_filename):
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(total_pollution), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Total CO2 Heatmap")
    plt.savefig(output_filename)
    plt.close()

def compute_metrics(data1, data2, data1err, data2err, pollutant, charac, prefix, posit = None):
    # Flatten the data for comparison
    flat_data1 = data1.flatten()
    flat_data2 = data2.flatten()
    flat_data1err = data1err.flatten()
    flat_data2err = data2err.flatten()
    print(len(flat_data1))
    
    # Identify nonzero elements in both data1 and data2
    #if ((flat_data1 !=0)!=(flat_data2!=0)).any():
    #    print(len(flat_data1 != 0)-len(flat_data2 != 0))
    nonzero_mask = (flat_data1 != 0) & (flat_data2 != 0)
    if charac == 'positions':
        nonzero_mask = nonzero_mask & (flat_data1 < np.max(flat_data1))
    
    # Apply the mask to filter nonzero elements
    filtered_data1 = flat_data1[nonzero_mask]
    filtered_data2 = flat_data2[nonzero_mask]
    filtered_data1err = flat_data1err[nonzero_mask]
    filtered_data2err = flat_data2err[nonzero_mask]
    print(len(filtered_data1))

    if posit is not None and charac!='positions':
        filtered_data1 = filtered_data1 / posit.flatten()[nonzero_mask]
    
    # Mean Absolute Error (MAE) for nonzero elements
    #mae = np.mean(np.abs(filtered_data1 - filtered_data2))
    
    # Pearson Correlation Coefficient for nonzero elements
    correlation, _ = pearsonr(filtered_data1, filtered_data2)
    
    # Relative Error (relative to the mean of filtered_data1)
    #mean_data1 = np.mean(filtered_data1)
    #relative_error = mae / mean_data1
    #relative_error = np.mean(np.abs(filtered_data1 - filtered_data2)/filtered_data1)


    plt.scatter(filtered_data1, filtered_data2, marker='o')
    #plt.errorbar(filtered_data1, filtered_data2, xerr = filtered_data1err, yerr = filtered_data2err, marker='o',fmt='none')
    plt.xlabel(f"Average {charac}")
    plt.ylabel(f'{prefix} - {pollutant}')

    if posit is None:
        output_filename = os.path.join(output_path, f"{prefix}_{pollutant}_vs_{charac}.png")
    else:
        output_filename = os.path.join(output_path, f"{prefix}_{pollutant}_vs_{charac}_corrected.png")
    plt.title(f'{prefix} - {pollutant} vs {charac}', wrap=True)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(output_filename, dpi=600)
    print(f"Saved plot to {output_filename}")
    plt.close()
    print(pollutant, charac, correlation)
    return

def process_experiments():
    seeds = [12, 34, 1, 78]
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

    for pollutant in pollutants:
        heatmapsP = []
        for charac in properties:
            heatmapsC = []
            for prefix in ['P', 'G']:
                for seed in seeds:
                    params2 = params.copy()
                    params2['seed'] = seed
                    data_aux = load_file('A', params2)
                    if data_aux is not None:
                        total_pollution = calculate_total_pollution(data_aux, pollutant,prefix)#[:,:,-1]
                        heatmapsP.append(total_pollution)
                        total_occupation = calculate_total_occupation(data_aux, charac)
                        heatmapsC.append(total_occupation)
                data2=np.mean(heatmapsP, axis=0)
                data2err=np.std(heatmapsP, axis=0)
                data1=np.mean(heatmapsC, axis=0)
                data1err=np.std(heatmapsC, axis=0)
                if charac == 'positions':
                    posit = data1
                compute_metrics(data1, data2, data1err, data2err, pollutant, charac, prefix)
    # Compare each pair of heatmaps


process_experiments()
