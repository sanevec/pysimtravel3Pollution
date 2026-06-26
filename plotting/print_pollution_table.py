import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

plot_cars = True

filter_lower_than_vmin = True
vmin=1e-2

# Adjust base_path to be relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, '../data/testTime')

params = {
    'delta': 0.1,
    'gamma': 0.01,
    'times': 2000,
    'seed': 12,
    'buildings': True,
    'distributionCS': 1,
    'densityCars': 0.05,
    'densityEV': 0.05,
    'densityDiesel': 0,
    'windV': "(0, 0)",
    'pollutionRouting': False
}

def load_file(prefix, params2):
    filename = (f"{prefix}_delta_{params2['delta']}_gamma_{params2['gamma']}_times_{params2['times']}_seed_{params2['seed']}_"
                f"buildings_{params2['buildings']}_distributionCS_{params2['distributionCS']}_densityCars_{params2['densityCars']}_"
                f"densityEV_{params2['densityEV']}_densityDiesel_{params2['densityDiesel']}_windV_{params2['windV']}_pollutionRouting_{params2['pollutionRouting']}.npz")
    filepath = os.path.join(base_path, filename)
    if not os.path.exists(filepath):
        return None
    return np.load(filepath)

def calculate_total_pollution(data, pollutant, prefix):
    total_pollution = np.zeros(list(data.values())[0].shape)
    sumWhere = np.ones_like(total_pollution)
    for key in data.keys():
        if pollutant in key:
            total_pollution += sumWhere * data[key]
    return total_pollution

wind_configs = [
    "(0, 0)",
    "(np.float32(-0.3), np.float32(0.9))",
    "(np.float32(0.9), np.float32(0.3))"
]

pollutant = 'CO2'
prefix = 'P'

# print("| Time | No Wind (0, 0) | NW Wind (-0.3, 0.9) | SE Wind (0.9, 0.3) |")
# print("|---|---|---|---|")

data_by_wind = {}
pollutions_by_wind = {}
cars_by_wind = {}
max_color_value = 0
all_data = {}

for wind in wind_configs:
    params2 = params.copy()
    params2['windV'] = wind
    data = load_file(prefix, params2)
    all_data[wind] = data
    if plot_cars:
        cars_by_wind[wind] = load_file('C', params2)
    
    if data is not None:
        pollution = calculate_total_pollution(data, pollutant, prefix)
        pollutions_by_wind[wind] = pollution
        data_by_wind[wind] = [np.sum(pollution[:, :, t]) for t in range(pollution.shape[2])]
        if np.max(pollution) > max_color_value:
            max_color_value = np.max(pollution)
    else:
        data_by_wind[wind] = None

saved_times = len(data_by_wind["(0, 0)"]) if data_by_wind.get("(0, 0)") else 0

# for t in range(0, 2001, 250):
#     if saved_times > 0:
#         frame_idx = int(t * (saved_times - 1) / 2000)
#     else:
#         frame_idx = t
#         
#     row = [str(t)]
#     for wind in wind_configs:
#         if data_by_wind.get(wind) and frame_idx < len(data_by_wind[wind]):
#             row.append(f"{data_by_wind[wind][frame_idx]:.6f}")
#         else:
#             row.append("N/A")
#     print(f"| {' | '.join(row)} |")

# Now generate images for t = 500
target_t = 500
if saved_times > 0:
    frame_idx = int(target_t * (saved_times - 1) / 2000)
    # print('frame_id=',frame_idx)
    
    # Load road map to highlight accessible cells using the simulator's logic
    road_mask = None
    try:
        import json
        grid_path = os.path.join(script_dir, '../graph_grid.json')
        if os.path.exists(grid_path):
            with open(grid_path, 'r') as f:
                grid_data = json.load(f)
            n_width = grid_data['grid_info']['cols']
            n_height = grid_data['grid_info']['rows']
            
            acc = np.zeros((n_width+2, n_height+2))
            for cell in grid_data['cells']:
                r = cell['row']
                c = cell['col']
                if cell.get('is_road', False):
                    acc[c+1, r+1] = 1
            
            acc[0,:]=1
            acc[-1,:]=1
            acc[:,0]=1
            acc[:,-1]=1

            border_ring_width = 5
            acc[1:border_ring_width+1, :] = 1
            acc[-(border_ring_width+1):-1, :] = 1
            acc[:, 1:border_ring_width+1] = 1
            acc[:, -(border_ring_width+1):-1] = 1

            acc_neig_edge = (
                acc[0:-2, 1:-1] + acc[2:, 1:-1] +
                acc[1:-1, 0:-2] + acc[1:-1, 2:]
            )
            acc_neig_corner = (
                acc[0:-2, 0:-2] + acc[2:, 2:] +
                acc[2:, 0:-2] + acc[0:-2, 2:]
            )

            for i in range(1,n_width+1):
                for j in range(1,n_height+1):
                    if acc_neig_corner[i-1,j-1]+acc_neig_edge[i-1,j-1]:
                        if (i in range(2,n_width) and j in range(2,n_height)) or acc_neig_corner[i-1,j-1]+acc_neig_edge[i-1,j-1]>3:
                            acc[i,j]=1

            road_mask = acc[1:-1, 1:-1] == 1
    except Exception as e:
        print("Could not generate road mask:", e)

    num_cells = np.sum(road_mask) if road_mask is not None else 1

    # Print table with all pollutants
    print(f"\n| Pollutant | No Wind: Tot (Avg, Max) | NW Wind: Tot (Avg, Max) | SE Wind: Tot (Avg, Max) |")
    print("|---|---|---|---|")
    all_pollutants = ['CO2', 'NOx', 'VOC', 'PMexhaust', 'PMnonexhaust25', 'PMnonexhaust10']
    for p in all_pollutants:
        row_str = f"| {p} "
        for wind in wind_configs:
            if all_data.get(wind) is not None:
                p_matrix = calculate_total_pollution(all_data[wind], p, prefix)
                if frame_idx < p_matrix.shape[2]:
                    tot = np.sum(p_matrix[:, :, frame_idx])
                    mx = np.max(p_matrix[:, :, frame_idx])
                    avg_conc = (tot * 1e6) / (num_cells * 125)
                    row_str += rf"| {tot*1e6:.2e} $\mu g$ ({avg_conc:.2e} $\mu g/m^3$, {mx*1e6/125:.2e} $\mu g/m^3$) "
                else:
                    row_str += "| N/A "
            else:
                row_str += "| N/A "
        row_str += "|"
        print(row_str)
    print("\n")
    
    wind_names = ["No Wind", "NE Wind (-3, 9) km/h", "NW Wind (9, 3) km/h"]
    vehicle_types = ['0', '1', '2']
    vehicle_names = {'0': 'EV', '1': 'Petrol', '2': 'Diesel'}
    car_colors = ['cyan', 'black', 'magenta']

    for p in all_pollutants:
        # Calculate max_color_value for this specific pollutant
        max_color_value_p = 0
        poll_matrices_p = {}
        for wind in wind_configs:
            if all_data.get(wind) is not None:
                p_matrix = calculate_total_pollution(all_data[wind], p, prefix)
                poll_matrices_p[wind] = p_matrix
                if np.max(p_matrix) > max_color_value_p:
                    max_color_value_p = np.max(p_matrix)

        if max_color_value_p <= 0:
            max_color_value_p = 1e-10  # Prevent error if empty

        # Determine appropriate unit for the colorbar based on max_color_value_p
        if max_color_value_p >= 1:
            cbar_unit = "g"
            unit_mult = 1.0
        elif max_color_value_p >= 1e-3:
            cbar_unit = "mg"
            unit_mult = 1e3
        else:
            cbar_unit = r"$\mu g$"
            unit_mult = 1e6

        max_color_value_p_scaled = max_color_value_p * unit_mult
        vmin_p = max_color_value_p_scaled / 1e5#1000.0  # Dynamic vmin based on max

        # Setup colormap
        colors = ["green", "yellow", "red"]
        n_bins = 1000
        cmap_name = "custom1"
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        norm = mcolors.LogNorm(vmin=vmin_p, vmax=max_color_value_p_scaled)
        
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 4)
        ax0 = fig.add_subplot(gs[0, 1:3])
        ax1 = fig.add_subplot(gs[1, 0:2])
        ax2 = fig.add_subplot(gs[1, 2:4])
        axes = [ax0, ax1, ax2]
        
        for i, wind in enumerate(wind_configs):
            ax = axes[i]
            if wind in poll_matrices_p:
                pollution = poll_matrices_p[wind]
                if frame_idx < pollution.shape[2]:
                    poll_plot = pollution[:, :, frame_idx].copy() * unit_mult
                    
                    if road_mask is not None and filter_lower_than_vmin:
                        # Set accessible cells with pollution < vmin_p to exactly vmin_p
                        # so they appear with the lowest color (green)
                        poll_plot[(road_mask) & (poll_plot < vmin_p)] = vmin_p
                    
                    plot = ax.imshow(np.transpose(poll_plot), cmap=cm, interpolation="nearest", norm=norm, origin='lower')
                    total_p = np.sum(pollution[:, :, frame_idx])
                    
                    # Determine appropriate unit for total_p
                    if total_p >= 1e3:
                        tot_str = f"{total_p/1e3:.2f} kg"
                    elif total_p >= 1:
                        tot_str = f"{total_p:.2f} g"
                    elif total_p >= 1e-3:
                        tot_str = f"{total_p*1e3:.2f} mg"
                    else:
                        tot_str = rf"{total_p*1e6:.2f} $\mu g$"
                        
                    ax.set_title(rf"{wind_names[i]}" + "\n" + rf"$t={target_t/60*1.8:.0f}\, min$, $P_{{tot}}={tot_str}$")
                    # print(rf"Wind {wind_names[i]}: Total pollution = {total_p*1e6} $\mu g$")
                    # print(rf"Maximum pollution concentration: {np.max(poll_plot)/5**3} $\mu g/m^3$")
                    if plot_cars and wind in cars_by_wind and cars_by_wind[wind] is not None:
                        data_cars = cars_by_wind[wind]
                        for idx, vehicle in enumerate(vehicle_types):
                            pos_key = f"{vehicle}_positions"
                            if pos_key in data_cars:
                                positions = data_cars[pos_key][:, :, frame_idx]
                                xx, yy = np.where(positions)
                                if len(xx) > 0:
                                    ax.scatter(xx, yy, c=car_colors[idx], s=1.5, label=vehicle_names[vehicle] if i == 0 else "")
                    
                    # Draw wind arrow in the top right corner
                    if wind != "(0, 0)":
                        import ast
                        w_str = wind.replace("np.float32", "").replace("(", "").replace(")", "")
                        w_parts = w_str.split(",")
                        wx = float(w_parts[0].strip())
                        wy = -float(w_parts[1].strip())
                        
                        length = np.sqrt(wx**2 + wy**2)
                        dx = (wx / length) * 0.1
                        dy = (wy / length) * 0.1
                        
                        start_x = 0.6 - dx/2
                        start_y = 0.9 - dy/2
                        ax.annotate('', xy=(start_x + dx, start_y + dy), xytext=(start_x, start_y),
                                    xycoords='axes fraction', textcoords='axes fraction',
                                    arrowprops=dict(facecolor='black', width=3, headwidth=10, shrink=0))
                else:
                    ax.set_title(f"{wind_names[i]}\nData not available")
            else:
                ax.set_title(f"{wind_names[i]}\nData not available")
            
            ax.set_axis_off()
            if i == 0 and plot_cars:
                ax.legend(loc='upper left', fontsize='small')
        
        # Add a single colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cb = fig.colorbar(plot, cax=cbar_ax, format='%.2e')
        cb.ax.set_title(f"({cbar_unit})", pad=10)
        
        out_filename = os.path.join(script_dir, f"pollution_comparison_t500_{p}.pdf")
        plt.savefig(out_filename, bbox_inches='tight', format='pdf')
        plt.close(fig)
        print(f"Imagen comparativa generada para {p}: {out_filename}")

