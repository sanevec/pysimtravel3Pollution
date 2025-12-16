import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import numpy as np
import os

base_path = "simulationData32"
output_path = "gifs"
if not os.path.exists(output_path):
    os.makedirs(output_path)

delta=0.1
gamma=0.01
times=2000
savedTimes=40
width=288
height=288

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

vehicle_types = ['0', '1', '2']

params = {
    'delta': 0.1,
    'gamma': 0.01,
    'times': 2000,
    'seed': 1,
    'buildings': True,
    'distributionCS': 1,
    'densityCars': 0.25,
    'densityEV': 0.05,
    'densityDiesel': 0.05,
    'windV': "(0, 0)",
    'pollutionRouting': False
}

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

def update_plot(frame_number, pollution, vehicle_positions, plot, fig, ax):
    for collection in ax.collections:
        collection.remove()
    total_pollution = np.sum(pollution[:, :, frame_number])
    plot.set_data(np.transpose(pollution[:, :, frame_number]))
    ax.set_title(f"$t={frame_number*times//savedTimes}$, $P_{{tot}}={total_pollution:.6f}$")
    colors = ['blue', 'green', 'red']
    for idx, vehicle in enumerate(vehicle_types):
        positions = vehicle_positions[vehicle][:, :, frame_number]
        numVeh = np.sum(positions)
        numVeh = int(numVeh)
        xx = [0]*numVeh
        yy = [0]*numVeh
        i=0
        for x in range(width):
            for y in range(height):
                if positions[x,y]:
                    xx[i]=x
                    yy[i]=y
                    i+=1
        ax.scatter(xx, yy, c=colors[idx], s=3, label=vehicle)
    ax.legend(loc='upper right')

pollutant='CO2'
squares=False

for seed in seeds:
    params2 = params.copy()
    params2['seed'] = seed
    prefix = 'G'
    data = load_file(prefix, params2)
    data2 = load_file('C', params2)
    vehicle_positions = {vehicle: np.zeros((data2['1_positions'].shape)) for vehicle in vehicle_types}
    
    if data is not None and data2 is not None:
        for vehicle in vehicle_types:
            key = f"{vehicle}_positions"
            if key in data2:
                vehicle_positions[vehicle] = data2[key]
    
        if prefix == 'P' or prefix == 'G':
            pollution=calculate_total_pollution(data, pollutant, prefix, squares=squares)

            max_color_value = np.max(pollution)
            print(seed, max_color_value)

            plt.plot(range(pollution.shape[2]),[np.sum(pollution[:,:,t]) for t in range(pollution.shape[2])])
            plt.xlabel('$t/1.8 s$')
            plt.ylabel('$P_{tot} (g)$')
            output_filename = os.path.join(output_path, f"{prefix}_{pollutant}_squares_{squares}_seed_{seed}_distributionCS_{params['distributionCS']}_densityCars_{params['densityCars']}_windV_{params['windV']}_buildings_{params['buildings']}.png")
            plt.savefig(output_filename)
            plt.close()

            fig, ax = plt.subplots()

            colorbar_scale = 'logarithmic'  # 'logarithmic' or 'linear'

            # Initial plot
            if colorbar_scale == 'logarithmic':
                norm = mcolors.LogNorm(vmin=0.01, vmax=max_color_value) # Avoid zero in log scale
            else:
                norm = mcolors.Normalize(vmin=0.01, vmax=max_color_value)

            # Define the colors for the custom colormap
            colors = ["green", "yellow", "red"]  # Define a list of colors
            n_bins = 1000  # Increase this number to make the transitions smoother
            cmap_name = "custom1"

            # Create the colormap
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

            plot = ax.imshow(pollution[:, :, 0], cmap=cm, interpolation="nearest", norm=norm)
            colorbar = fig.colorbar(plot, ax=ax, format='%.2f')

            # Creating the animation
            ani = animation.FuncAnimation(fig, update_plot, savedTimes+1, fargs=(pollution, vehicle_positions, plot, fig, ax), interval=200)

            # Save the animation as a gif
            output_filename = os.path.join(output_path, f"{prefix}_{pollutant}_squares_{squares}_seed_{seed}_distributionCS_{params['distributionCS']}_densityCars_{params['densityCars']}_windV_{params['windV']}_buildings_{params['buildings']}.gif")
            ani.save(output_filename, writer='imagemagick', fps=5)
            plt.close()
