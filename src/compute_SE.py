from fdtd_solver import FDTDSolver
from RF_index import RFIndex
import numpy as np

class SESimulation:
    def __init__(self, layer_materials, layer_thicknesses, solver):

        self.layer_materials = layer_materials
        self.layer_thicknesses = layer_thicknesses
        self.fdtd_solver = solver

        # Define frequency vectors for simulation
        self.f_vector = np.linspace(8e9, 18e9, 100)

    def perform_stackrt_simulation(self, n_matrix, f_vector):
        return self.fdtd_solver.stackrt(n_matrix, self.layer_thicknesses, f_vector, 0)

    def calculate_SE(self, RT):
        Ts = RT['Ts']
        Tp = RT['Tp']
        SE_Ts = -10 * np.log10(Ts)
        SE_Tp = -10 * np.log10(Tp)
        
        SE_Ts_mean = np.mean(SE_Ts)
        SE_Tp_mean = np.mean(SE_Tp)

        SE = (SE_Tp_mean + SE_Ts_mean) / 2
        return SE

    def run_simulation(self):
        # Create refractive index matrix
        n_matrix = RFIndex.create_matrix(self.fdtd_solver, self.layer_materials, self.f_vector)

        # Perform stackrt simulation
        RT = self.perform_stackrt_simulation(n_matrix, self.f_vector)

        # Calculate Spectral Efficiency (SE)
        SE = self.calculate_SE(RT)

        # Output results
        # print(f'stackrt: SE {SE:.4f}')
        return SE

# Example usage
if __name__ == "__main__":
    # Define layer materials and thicknesses
    layer_materials = ['Air', 'TiO2 (Titanium Dioxide) - Siefke', 'SiO2 (Glass) - Palik',
                        'TiO2 (Titanium Dioxide) - Siefke', 'Cr (Chromium) - CRC', 'SiO2 (Glass) - Palik',
                        'Cr (Chromium) - Palik', 'Air']
    layer_thicknesses = np.array([0, 24e-9, 141e-9, 153e-9, 18e-9, 135e-9, 164e-9, 0])

    # Initialize and run the simulation
    solver = FDTDSolver()
    simulation = SESimulation(layer_materials, layer_thicknesses, solver)
    simulation.run_simulation()
