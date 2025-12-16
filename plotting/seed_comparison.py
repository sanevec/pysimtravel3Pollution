import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

base_path = "simulationData32Last"
output_path = "representationsSeeds"
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

def compute_metrics(data1, data2):
    # Flatten the data for comparison
    flat_data1 = data1.flatten()
    flat_data2 = data2.flatten()
    
    # Identify nonzero elements in both data1 and data2
    #if ((flat_data1 !=0)!=(flat_data2!=0)).any():
    #    print(len(flat_data1 != 0)-len(flat_data2 != 0))
    nonzero_mask = (flat_data1 != 0) & (flat_data2 != 0)
    
    # Apply the mask to filter nonzero elements
    filtered_data1 = flat_data1[nonzero_mask]
    filtered_data2 = flat_data2[nonzero_mask]
    
    # Mean Absolute Error (MAE) for nonzero elements
    mae = np.mean(np.abs(filtered_data1 - filtered_data2))
    
    # Pearson Correlation Coefficient for nonzero elements
    correlation, _ = pearsonr(filtered_data1, filtered_data2)
    
    # Relative Error (relative to the mean of filtered_data1)
    mean_data1 = np.mean(filtered_data1)
    #relative_error = mae / mean_data1
    relative_error = np.mean(np.abs(filtered_data1 - filtered_data2)/filtered_data1)
    
    return relative_error, correlation

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

def process_all_experiments():
    # Define the lists of parameters
    lists = {
        'delta': [0.1],
        'gamma': [0.01],
        'times': [2000],
        'seed': [12, 34, 1, 78],
        'buildings': [True, False],
        'distributionCS': [0, 1, 2],
        'densityCars': [0.05, 0.15, 0.25, 0.35],
        'densityEV': [0.05, 0.35, 0.65, 0.95],
        'densityDiesel': [0, 0.25, 0.5, 0.75, 1],
        'windV': ["(0, 0)", "(0.1, 0)", "(0.3, 0)", "(0.5, 0)", "(0.2, 0.2)", "(0.5, 0.3)", "(1.0, 0.5)"],
        'pollutionRouting': [False]
    }

    pollutants = ['CO2', 'NOx', 'VOC', 'PMexhaust', 'PMnonexhaust25', 'PMnonexhaust10']

    # Generate all combinations of parameters (excluding 'seed' and 'pollutant')
    param_combinations = list(itertools.product(
        lists['delta'], lists['gamma'], lists['times'],
        lists['buildings'], lists['distributionCS'],
        lists['densityCars'], lists['densityEV'],
        lists['densityDiesel'], lists['windV'],
        lists['pollutionRouting']
    ))

    # Initialize variables to accumulate results
    results = {(seed1, seed2, pollutant): {'relative_error_sum': 0, 'correlation_sum': 0, 'count': 0}
               for seed1, seed2 in itertools.combinations(lists['seed'], 2)
               for pollutant in pollutants}

    for params_comb in param_combinations:
        params_dict = {
            'delta': params_comb[0],
            'gamma': params_comb[1],
            'times': params_comb[2],
            'buildings': params_comb[3],
            'distributionCS': params_comb[4],
            'densityCars': params_comb[5],
            'densityEV': params_comb[6],
            'densityDiesel': params_comb[7],
            'windV': params_comb[8],
            'pollutionRouting': params_comb[9]
        }

        heatmaps = {}
        for seed in lists['seed']:
            params_dict['seed'] = seed
            prefix = 'A'
            data = load_file(prefix, params_dict)
            if data is not None:
                for pollutant in pollutants:
                    total_pollution = calculate_total_pollution(data, pollutant, prefix)
                    heatmaps[(seed, pollutant)] = total_pollution

        # Compare each pair of heatmaps
        for (seed1, seed2) in itertools.combinations(lists['seed'], 2):
            for pollutant in pollutants:
                if (seed1, pollutant) in heatmaps and (seed2, pollutant) in heatmaps:
                    relative_error, correlation = compute_metrics(heatmaps[(seed1, pollutant)], heatmaps[(seed2, pollutant)])
                    result_key = (seed1, seed2, pollutant)
                    results[result_key]['relative_error_sum'] += relative_error
                    results[result_key]['correlation_sum'] += correlation
                    results[result_key]['count'] += 1

    # Calculate averages and print results
    for key, result in results.items():
        if result['count'] > 0:
            avg_relative_error = result['relative_error_sum'] / result['count']
            avg_correlation = result['correlation_sum'] / result['count']
            seed1, seed2, pollutant = key
            print(f"Average comparison between seed {seed1} and seed {seed2} in {pollutant}: "
                  f"relative error = {avg_relative_error}, Correlation = {avg_correlation}")

process_all_experiments()
