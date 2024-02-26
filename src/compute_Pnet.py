from utils import kb, c, h
import sys, os
# sys.path.append("C:\\Program Files\\Lumerical\\v232\\api\\python\\")
# sys.path.append("C:\\Program Files\\Lumerical\\v241\\api\\python\\")
from data_processing import DataProcessor
from fdtd_solver import FDTDSolver
from solar_index import SolarIndex
from scipy.integrate import trapz
import numpy as np
import pandas as pd

NmToM = 1e-9
MToNm = 1e9
UmtoM = 1e-6
MToUm = 1e6

class PnetSimulation:
    def __init__(self, layer_materials, layer_thicknesses, solver):
        # Constants
        self.kb = kb
        self.c = c
        self.h = h
        self.T_amb = 298
        self.T_struct = 298

        self.layer_materials = layer_materials
        self.layer_thicknesses = layer_thicknesses
        self.fdtd_solver = solver

        # Define wavelength ranges and frequency vectors
        self.lambda_start_cool = 4e-6
        self.lambda_stop_cool = 20e-6
        self.lambda_start_300nm_4um = 0.3e-6
        self.lambda_stop_300nm_4um = 4e-6
        self.num_cool_points = 1289
        self.num_300nm_4um_points = 1000
        self.f_vector_cool = np.linspace(c / self.lambda_stop_cool, c / self.lambda_start_cool, self.num_cool_points)
        self.f_vector_300nm_4um = np.linspace(c / self.lambda_stop_300nm_4um, c / self.lambda_start_300nm_4um, self.num_300nm_4um_points)


    def perform_stackrt_simulation(self, n_matrix, f_vector, theta):
        return self.fdtd_solver.stackrt(n_matrix, self.layer_thicknesses, f_vector, theta)

    def read_and_interpolate_AM1_5(self):
        AM1_5_data = DataProcessor.read_csv('AM1.5.csv')
        AM1_5_wavelengths = AM1_5_data.iloc[:, 0] * NmToM
        AM1_5_irradiance = AM1_5_data.iloc[:, 2]
        return DataProcessor.interpolate_data(AM1_5_wavelengths, AM1_5_irradiance, 
                                              self.c/self.f_vector_300nm_4um)

    def calculate_absorbed_power_density(self, RT_300nm_4um, AM1_5_interpolated):
        lambda_vector_300nm_4um = self.c/self.f_vector_300nm_4um

        R_300nm_4um_freq = (RT_300nm_4um['Rp'] + RT_300nm_4um['Rs']) / 2
        A_300nm_4um_freq = 1 - R_300nm_4um_freq
        A_300nm_4um_freq = A_300nm_4um_freq[:, 0]
        absorbed_power_density_300nm_4um = AM1_5_interpolated * A_300nm_4um_freq
        P_300nm_4um = -trapz(absorbed_power_density_300nm_4um, lambda_vector_300nm_4um * MToNm)
        return P_300nm_4um

    def calculate_cooling(self, n_cool):
        # Setup for angle and wavelength vectors
        num_angles = 91
        dtheta = np.pi / (2 * num_angles)
        theta_values = np.linspace(0, np.pi / 2, num_angles)
        lambda_vector_cool = self.c / self.f_vector_cool
        theta_values_deg = np.linspace(0, 90, num_angles)

        # Perform stackrt simulation for all angles at once
        RT_cool = self.fdtd_solver.stackrt(n_cool, self.layer_thicknesses, self.f_vector_cool, theta_values_deg)
        R_cool_freq = (RT_cool['Rp'] + RT_cool['Rs']) / 2
        A_cool_all_angles = 1 - R_cool_freq
        # print(A_cool_all_angles.shape)


        # Read atmospheric transmittance data and interpolate
        atm_data = DataProcessor.read_csv('modtran_4_20.csv')
        lambda_atm = atm_data.iloc[:, 0] * UmtoM
        transmittance_atm = atm_data.iloc[:, 1]
        transmittance_atm_interp = DataProcessor.interpolate_data(lambda_atm, transmittance_atm, lambda_vector_cool)
        transmittance_atm_interp_2d = transmittance_atm_interp[:, np.newaxis]
        # Calculate emissivity_atm_interp2
        emissivity_atm_interp = 1 - transmittance_atm_interp_2d ** (1 / np.cos(theta_values))

        # Calculate Blackbody radiation intensities
        I_BB_amb = ((2 * self.h * self.c**2) / (lambda_vector_cool**5) /
                    (np.exp((self.h * self.c) / (lambda_vector_cool * self.kb * self.T_amb)) - 1))
        I_BB_struct = ((2 * self.h * self.c**2) / (lambda_vector_cool**5) /
                       (np.exp((self.h * self.c) / (lambda_vector_cool * self.kb * self.T_struct)) - 1))

        # Calculate atmospheric power (P_atm)
        I_BB_amb_2d = I_BB_amb[:, np.newaxis]
        # Calculate angular factors for broadcasting
        angular_factors = np.cos(theta_values) * np.sin(theta_values) * 2 * np.pi
        angular_factors_2d = angular_factors[np.newaxis, :]  # Shape becomes (1, 90)
        # Calculate the integrand for atmospheric power
        integrand_atm = I_BB_amb_2d * emissivity_atm_interp * A_cool_all_angles * angular_factors_2d
        P_atm = -trapz(trapz(integrand_atm, lambda_vector_cool, axis=0), theta_values)

        # Calculate radiative power (P_rad)
        power_density_struct = I_BB_struct * A_cool_all_angles.T * np.cos(theta_values[:, np.newaxis]) * np.sin(theta_values[:, np.newaxis]) * 2 * np.pi
        P_rad = -trapz(trapz(power_density_struct, lambda_vector_cool, axis=1), theta_values)

        # Calculate net cooling power (P_cool) and net power (P_net)
        P_cool = P_rad - P_atm

        # print(f"P_atm: {P_atm} W/m^2")
        # print(f"P_rad: {P_rad} W/m^2")
        return P_cool

    def run_simulation(self):
        # Create refractive index matrices
        n_cool = SolarIndex.create_matrix(self.fdtd_solver, self.layer_materials, self.f_vector_cool)
        n_300nm_4um = SolarIndex.create_matrix(self.fdtd_solver, self.layer_materials, self.f_vector_300nm_4um)

        # Perform stackrt simulation for 300nm - 4um range
        RT_300nm_4um = self.perform_stackrt_simulation(n_300nm_4um, self.f_vector_300nm_4um, 0)

        # Read and interpolate AM1.5 data
        AM1_5_interpolated = self.read_and_interpolate_AM1_5()

        # Calculate absorbed power density
        P_300nm_4um = self.calculate_absorbed_power_density(RT_300nm_4um, AM1_5_interpolated)

        # Cooling calculations
        P_cool = self.calculate_cooling(n_cool)
        P_net = P_300nm_4um - P_cool

        # Output results
        # print(f"P_300nm_4um: {P_300nm_4um} W/m^2")
        # print(f"P_cool: {P_cool} W/m^2")
        # print(f"P_net: {P_net} W/m^2")
        return P_cool, P_net

    # ... Additional methods for each step in your simulation

def get_layers_from_csv(file_path, line_index):
    """
    Reads a CSV file and extracts layer materials and thicknesses from the specified line index.
    
    Parameters:
    - file_path: Path to the CSV file.
    - line_index: Index of the row (1-based) from which to extract the data.
    
    Returns:
    - layer_materials: List of materials for the layers.
    - layer_thicknesses: Array of thicknesses for the layers in meters.
    """
    df = pd.read_csv(file_path, header=None)  # No header in the CSV file
    
    # Subtract 1 from line_index because pandas uses 0-based indexing
    row = df.iloc[line_index - 1]
    
    # Assuming the first 10 columns are materials and the next 10 are thicknesses
    layer_materials = row[:10].tolist()
    layer_thicknesses = row[10:20].astype(float).values  # Convert to numpy array
    
    return layer_materials, layer_thicknesses

# Example usage
if __name__ == "__main__":
    # Define layer materials, thicknesses, and temperatures
    # Cold Structure
    layer_materials = ['Air', 'TiO2 (Titanium Dioxide) - Siefke', 'SiO2 (Glass) - Palik', 'Si (Silicon) - Palik',
                       'SiO2 (Glass) - Palik', 'Si (Silicon) - Palik', 'SiO2 (Glass) - Palik', 'Si (Silicon) - Palik',
                       'SiO2 (Glass) - Palik', 'Si (Silicon) - Palik']
    layer_thicknesses = np.array([0, 134e-9, 75e-9, 55e-9, 139e-9, 66e-9, 277e-9, 115e-9, 184e-9, 0])

    # Hot Structure
    # layer_materials = ['Air', 'TiO2 (Titanium Dioxide) - Siefke', 'SiO2 (Glass) - Palik',
    #                 'TiO2 (Titanium Dioxide) - Siefke', 'Cr (Chromium) - Palik', 'SiO2 (Glass) - Palik',
    #                   'Cr (Chromium) - Palik', 'Si (Silicon) - Palik']
    # layer_thicknesses = np.array([0, 24e-9, 141e-9, 153e-9, 18e-9, 135e-9, 164e-9, 0])

    # file_path = 'res/BO/PCOOL/pareto_frontier.csv'  # Update the path to your actual CSV file
    # line_index = 3  # For example, to read the 12th line
    
    # layer_materials_origin, layer_thicknesses_origin = get_layers_from_csv(file_path, line_index)
    # layer_materials = ['Air'] + layer_materials_origin + ['Air']
    # layer_thicknesses = np.pad(layer_thicknesses_origin, (1, 1), 'constant')

    # Initialize and run the simulation
    solver = FDTDSolver()
    simulation = PnetSimulation(layer_materials, layer_thicknesses, solver)
    simulation.run_simulation()
