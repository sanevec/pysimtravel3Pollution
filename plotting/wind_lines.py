import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import statsmodels.api as sm


base_path = ["simulationData32Last"]
output_path = "representationsLast"
if not os.path.exists(output_path):
    os.makedirs(output_path)

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


# Definir la función de ajuste
def func(x, alpha):
    return alpha * (1 - x)

# Extraer datos para el ajuste
def extract_data_for_fitting(prefix, params):
    data_points = []
    for x_val in lists['densityEV']:
        params['densityEV'] = x_val
        data = load_file(prefix, params)
        if data is not None:
            y_val = np.sum(calculate_total_data(data, params['pollutant'], prefix, squares=params['squares']))
            data_points.append((x_val, y_val))
    return np.array(data_points)

# Ajustar los datos y calcular alpha, su error y el coeficiente de correlación
def fit_lines_and_calculate_metrics(data_points):
    x_data = data_points[:, 0]
    y_data = data_points[:, 1]

    popt, pcov = curve_fit(func, x_data, y_data)
    alpha = popt[0]
    alpha_error = np.sqrt(np.diag(pcov))[0]
    
    # Calcular el coeficiente de correlación
    residuals = y_data - func(x_data, alpha)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return alpha, alpha_error, r_squared

# Parámetros específicos del caso
params = {
    'delta': 0.1,
    'gamma': 0.01,
    'times': 2000,
    'seed': 1,
    'buildings': True,
    'distributionCS': 1,
    'densityCars': 0.35,
    'densityEV': 0.25,  # Este valor será reemplazado en el bucle
    'densityDiesel': 0.5,
    'windV': "(0.5, 0.3)",  # Este valor será reemplazado en el bucle
    'pollutionRouting': False,
    'squares': False,
    'pollutant': 'CO2',
    'prefix': 'P',
    'vType': '1'
}

# Valores de windV a considerar
wind_values = ["(0, 0)", "(0.1, 0)", "(0.3, 0)", "(0.5, 0)", "(0.2, 0.2)", "(0.5, 0.3)", "(1.0, 0.5)"]

# Iterar sobre los valores de windV
results = []
wind_magnitudes = []
alphas = []
for wind in wind_values:
    params['windV'] = wind
    data_points = extract_data_for_fitting('P', params)
    alpha, alpha_error, r_squared = fit_lines_and_calculate_metrics(data_points)
    wind_magnitude = np.linalg.norm(eval(wind))  # Calcular el módulo de windV
    wind_magnitudes.append(wind_magnitude)
    alphas.append(alpha)
    results.append((wind, alpha, alpha_error, r_squared))
    #print(f"windV={wind}: alpha={alpha}, error={alpha_error}, R^2={r_squared}")

# Resultados finales
for result in results:
    wind, alpha, alpha_error, r_squared = result
    print(f"Para windV={wind}, alpha={alpha:.4f} ± {alpha_error:.4f}, relative error = {alpha_error/alpha :.4f}, R^2={r_squared:.4f}")

# Representar alpha frente al módulo de windV
plt.figure(figsize=(8, 6))
plt.plot(wind_magnitudes, alphas, 'o')
plt.xlabel('|w|')
plt.ylabel(r'$\alpha$')
plt.title(r'$\alpha$ vs |w|')
plt.grid(True)
# Guardar la imagen con buena calidad
output_filename = "alpha_vs_windV_magnitude.png"  # Nombre del archivo de salida
plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # dpi=300 para buena calidad


# Calcular la recta de regresión entre alpha y el módulo de windV
X = sm.add_constant(wind_magnitudes)  # Agregar la constante (intercepto)
model = sm.OLS(alphas, X)
results = model.fit()

# Extraer los valores necesarios
slope = results.params[1]
intercept = results.params[0]
slope_std_err = results.bse[1]
intercept_std_err = results.bse[0]
r_squared = results.rsquared

# Mostrar los resultados de la regresión
print(f"Recta de regresión: alpha = {slope:.4f} * |windV| + {intercept:.4f}")
print(f"Error estándar de la pendiente: {slope_std_err:.4f}")
print(f"Error estándar del intercepto: {intercept_std_err:.4f}")
print(f"Coeficiente de determinación R^2: {r_squared:.4f}")

# Añadir la recta de regresión a la gráfica
regression_line = np.array(wind_magnitudes) * slope + intercept
plt.plot(wind_magnitudes, regression_line, 'r-', label=f'Linear regression: r²={r_squared:.2f}')
plt.legend()

# Guardar la gráfica con la recta de regresión incluida
output_filename_with_regression = "alpha_vs_windV_magnitude_with_regression.png"
plt.savefig(output_filename_with_regression, dpi=300, bbox_inches='tight')


'''
# Calcular la recta de regresión entre alpha y el módulo de windV
slope, intercept, r_value, p_value, std_err = linregress(wind_magnitudes, alphas)

# Mostrar los resultados de la regresión
print(f"Recta de regresión: alpha = {slope:.4f} * |windV| + {intercept:.4f}")
print(f"Error estándar de la pendiente: {std_err:.4f}")
print(f"Coeficiente de determinación R^2: {r_value**2:.4f}")

# Añadir la recta de regresión a la gráfica
regression_line = np.array(wind_magnitudes) * slope + intercept
plt.plot(wind_magnitudes, regression_line, 'r-', label=f'Linear regression: R²={r_value**2:.4f}')
plt.legend()

# Guardar la gráfica con la recta de regresión incluida
output_filename_with_regression = "alpha_vs_windV_magnitude_with_regression.png"
plt.savefig(output_filename_with_regression, dpi=300, bbox_inches='tight')
'''