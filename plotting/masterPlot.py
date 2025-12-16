## Imports and initializations
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

base_path = ["simulationData32Last"]
output_path = "masterRepresentationsLast"
if not os.path.exists(output_path):
    os.makedirs(output_path)

params = {
    'delta': 0.1,
    'gamma': 0.01,
    'times': 2000,
    'seed': 1,
    'buildings': True,
    'distributionCS': 1,
    'densityCars': 0.15,
    'densityEV': 0.25,
    'densityDiesel': 0.5,
    'windV': "(0.5, 0.3)",
    'pollutionRouting': False,
    'squares': False,
    'pollutant': 'CO2',
    'prefix': 'P',
    'vType': '1'
}

lists = {
    'delta': [0.1],
    'gamma': [0.01],
    'times': [2000],
    'seed': [12, 34, 1, 78],
    'buildings': [True, False],
    'distributionCS': [0, 1, 2],
    'densityCars': [0.05, 0.15, 0.25, 0.35],#[0.05, 0.15, 0.25, 0.35],
    'densityEV': [0.05, 0.35, 0.65, 0.95],
    'densityDiesel': [0, 0.25, 0.5, 0.75, 1],
    'windV': ["(0, 0)", "(0.1, 0)", "(0.3, 0)", "(0.5, 0)", "(0.2, 0.2)", "(0.5, 0.3)", "(1.0, 0.5)"],
    'pollutionRouting': [False],
    'squares': [False, True],
    'pollutant': ['CO2', 'NOx', 'VOC', 'PMexhaust', 'PMnonexhaust25', 'PMnonexhaust10'],
    'prefix': ['P', 'G'],
    'vType': ['0', '1', '2'],
    'carprop': ['positions', 'velocities', 'accelerations', 'braking']
}


## Fill this with the desired representations
prefixes = ['P', 'G']#['C']#[('A','C')]#,('A','G')]#['P', 'G']
averageIn = ['seed']  # Averaged in all
differentGraphs = ['densityCars']  # One by one
sameGraph = ['buildings', 'distributionCS', 'densityDiesel', 'windV']  # One by one
xVar = ['densityEV']
yVar = ['pollutant']
typeRepr = 'plot'# 'map' or 'plot'
sumvType = not 'vType' in (set(averageIn) | set(differentGraphs) | set(sameGraph) | set(xVar) | set(yVar))
print('sumvType = ', sumvType)
numdesv=3

## Just some checks
if typeRepr not in ['map', 'plot']:
    print('Wrong value for typeRepr')
    exit()
if prefixes not in [['C'],['P'], ['G'], ['P', 'G'], [('A','P'),('A','G')], [('A','P')], [('A','G')], [('A','C')]]:
    print('Error: the prefixes can only be P, G and tuples with A')
    exit()
allowed_elements = lists.keys()
sets_to_check = [set(averageIn), set(differentGraphs), set(sameGraph), set(xVar), set(yVar)]
no_common_elements = all(len(set1 & set2) == 0 for i, set1 in enumerate(sets_to_check) for set2 in sets_to_check[i + 1:])
all_elements_allowed = all(element in allowed_elements for s in sets_to_check for element in s)
if not no_common_elements:
    print('Error: there are common variables.')
    exit()
if not all_elements_allowed:
    print("Error: in the names of the variables.")
    exit()


## The code that makes it all possible
def load_file(prefix, params):
    # Build the filename based on the experiment parameters
    filename = (f"{prefix}_delta_{params['delta']}_gamma_{params['gamma']}_times_{params['times']}_seed_{params['seed']}_"
                f"buildings_{params['buildings']}_distributionCS_{params['distributionCS']}_densityCars_{params['densityCars']}_"
                f"densityEV_{params['densityEV']}_densityDiesel_{params['densityDiesel']}_windV_{params['windV']}_pollutionRouting_{params['pollutionRouting']}.npz")
    
    for path in base_path:
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            #print('Found: ', filename)
            return np.load(filepath)
    print(f"File not found: {filename}")
    return None


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

    # Mean of non-zero elements
    mean1 = np.mean(data1[data1 != 0])
    mean2 = np.mean(data2[data2 != 0])

    # Max value
    max1 = np.max(data1)
    max2 = np.max(data2)

    # Relative error
    relative_error = mae / mean1 if mean1 != 0 else float('inf')

    return mae, rmse, correlation, mean1, mean2, max1, max2, relative_error

def generate_heatmap(total_data, output_filename):
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(total_data), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Total CO2 Heatmap")
    plt.savefig(output_filename)
    plt.close()

def process_comparison(heatmaps, seeds):
    results = []
    for i in range(len(heatmaps)):
        for j in range(i + 1, len(heatmaps)):
            metrics = compute_metrics(heatmaps[i], heatmaps[j])
            mae, rmse, correlation, mean1, mean2, max1, max2, relative_error = metrics
            results.append((seeds[i], seeds[j], mae, rmse, correlation, mean1, mean2, max1, max2, relative_error))
            print(
                f"Comparison between seed {seeds[i]} and seed {seeds[j]}: MAE = {mae}, RMSE = {rmse}, Correlation = {correlation}, "
                f"Means = {mean1}; {mean2}, Max = {max1}; {max2}, Relative error = {relative_error}")


def calculate_total_data(data, subdata, prefix, prefix2 = 'P', squares=False, params2=None):
    total_data = np.zeros(list(data.values())[0].shape)
    if squares:
        sumWhere = np.zeros_like(total_data)
        l = list(range(0, 10)) + list(range(85, 106)) + list(range(181, 202)) + list(range(277, 287))
        for i in l:
            for j in l:
                sumWhere[i + 1, j + 1] = 1
    else:
        sumWhere = np.ones_like(total_data)
    for key in data.keys():
        if prefix == 'A':
            if subdata in key and prefix2 in key:
                if data[key] is None:
                    print('Data is none!')
                    print(params2)
                else:
                    total_data += sumWhere * data[key]
        else:
            if subdata in key:
                total_data += sumWhere * data[key]
    return total_data

def process_experiments():
    for prefix in prefixes:
        for difGraph in differentGraphs:
            for dGraph in lists[difGraph]:
                for x in xVar:
                    for y in yVar:
                        for yy in lists[y]:
                            for samGraph in sameGraph:
                                plt.figure()
                                for sGraph in lists[samGraph]:
                                    means = []
                                    stds = []
                                    for xx in lists[x]:
                                        for avIn in averageIn:
                                            for aIn in lists[avIn]:
                                                params2 = params.copy()
                                                params2['prefix'] = prefix
                                                params2[difGraph] = dGraph
                                                params2[x] = xx
                                                params2[y] = yy
                                                params2[samGraph] = sGraph
                                                params2[avIn] = aIn
                                                if params2['densityEV'] + params2['densityDiesel'] <= 2:#1:
                                                    total_data = []
                                                    if prefix in ['P', 'G', 'C']:
                                                        data = load_file(prefix, params2)
                                                    elif prefix in [('A','P'), ('A', 'G'), ('A', 'C')]:
                                                        data = load_file('A', params2)
                                                    else:
                                                        data=None
                                                    if data is not None:
                                                        if prefix in ['P', 'G', ('A','P'), ('A','G')]:
                                                            if sumvType:
                                                                if prefix == ('A','P'):
                                                                    total_data.append(np.sum(
                                                                        calculate_total_data(data, params2['pollutant'], 'A', prefix2 = 'P',
                                                                                                squares=params2['squares'], params2=params2)))
                                                                elif prefix == ('A','G'):
                                                                    total_data.append(np.sum(
                                                                        calculate_total_data(data, params2['pollutant'], 'A', prefix2 = 'G',
                                                                                                squares=params2['squares'], params2=params2))*params2['times'])
                                                                elif prefix in ['P', 'G']:
                                                                    total_data.append(np.sum(
                                                                        calculate_total_data(data, params2['pollutant'], prefix,
                                                                                                squares=params2['squares'], params2=params2)[:, :, -1]))
                                                                else:
                                                                    total_data = None
                                                            else:
                                                                total_data = data[f'{params2["vType"]}_{params2["pollutant"]}']
                                                        elif prefix == ('A','C'):
                                                            total_data = data[f'cars_{params2["vType"]}_{params2["carprop"]}']
                                                        elif prefix == 'C':
                                                            total_data = data[f'{params2["vType"]}_{params2["carprop"]}'][:,:,-1]
                                                            total_data = total_data[total_data>0]
                                                        else:
                                                            print('Still not implemented???')
                                                    else:
                                                        print(f"Data is None for params: {params2}")
                                        if total_data is not None:
                                            means.append(np.mean(total_data))
                                            stds.append(np.std(total_data)*numdesv)
                                        else:
                                            print('Error: in total_data')
                                            means.append(0)
                                            stds.append(0)
                                    #print(len(lists[x]))
                                    xplot = lists[x][:len(means)]
                                    #print(len(xplot))
                                    #print(len(means))
                                    #print(len(stds))
                                    plt.errorbar(xplot, means, yerr=stds, label=f'{samGraph}={params2[samGraph]}', marker='o')
                                plt.xlabel(f"{x}")
                                if sumvType:
                                    plt.ylabel(f"{prefix} - {yy} - total")
                                else:
                                    plt.ylabel(f'{prefix} - {yy} - {params2["vType"]}')
                                aux = [f"{a}={params2[a]}" for a in lists.keys() if a not in averageIn + xVar + yVar + [samGraph, "vType", "prefix", "pollutionRouting", "carprop"]]
                                if not sumvType:
                                    aux.append(f"{'vType'}={params2['vType']}")
                                
                                title = f"{prefix} {yy}"

                                #title += "".join([" " + a for a in aux])
                                filename = "".join(["_" + a for a in aux])
                                #title += " Averaged in: " + "".join([a + "," for a in averageIn])

                                output_filename = os.path.join(output_path,
                                                            f"{prefix}_{yy}_vs_{x}" + filename + "_averagedIn" + "".join(["_" + a for a in averageIn]) + ".png")
                                plt.title(title, wrap=True)
                                plt.grid(True)
                                plt.legend()
                                plt.tight_layout()
                                plt.subplots_adjust(top=0.85)
                                plt.savefig(output_filename, dpi=600)
                                print(f"Saved plot to {output_filename}")
                                plt.close()
                                


process_experiments()
