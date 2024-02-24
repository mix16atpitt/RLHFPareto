import numpy as np
from utils import c
# for non metals, use the index at 20um because the end of solar band we simulate is 20um we are sure we have data here
# Frequency corresponding to 20um
freq_20um = c / 20e-6

# Update the calculation using specific conductivity values for each metal
# Constants
mu0 = 4 * np.pi * 1e-7  # H/m
sigma = 6.3e7  # S/m
Z_0 = 377  # Ohms, impedance of free space

# Frequency range
nu = np.linspace(8e9, 18e9, 11)  # From 8 GHz to 18 GHz
omega = 2 * np.pi * nu  # Angular frequency

# Provided conductivity values for each metal, from Wikipedia
Metals_sigma = {
    'Cu (Copper) - CRC': 5.96e7,
    'Cr (Chromium) - CRC': 7.74e6,
    'Ag (Silver) - CRC': 6.3e7,
    'Al (Aluminium) - CRC': 3.77e7,
    'Ni (Nickel) - Palik' : 1.43e7,
    'W (Tungsten) - CRC' : 1.79e7,
    'Ti (Titanium) - Palik' : 2.38e6,
    'Pd (Palladium) - Palik' : 9.52e6
}

# Empty dictionary to store updated values
Metals_nk_updated_specific_sigma = {}

# Iterate over each metal to calculate its specific n and k values using its conductivity
for metal, sigma in Metals_sigma.items():
    Z = np.sqrt(1j * omega * mu0 / sigma)  # Impedance of the material using specific sigma
    n_complex = Z_0 / Z  # Complex refractive index

    # Extract real and imaginary parts of the refractive index
    n_real = np.real(n_complex)
    k_imag = np.imag(n_complex)
    
    # Update the dictionary with the new values
    Metals_nk_updated_specific_sigma[metal] = {
        'freq_data': nu.tolist(),
        'n_data': n_real.tolist(),
        'k_data': k_imag.tolist()
    }

class RFIndex:
    @staticmethod
    def create_matrix(fdtd_solver, layer_materials, f_vector):
        num_layers = len(layer_materials)
        num_freqs = len(f_vector)
        n_matrix = np.zeros((num_layers, num_freqs), dtype=np.complex128)
        for i, material_name in enumerate(layer_materials):
            if material_name == 'Air':
                n_matrix[i, :] = 1
            elif material_name in Metals_nk_updated_specific_sigma:
                metal_data = Metals_nk_updated_specific_sigma[material_name]
                n = np.interp(f_vector, metal_data['freq_data'], metal_data['n_data'])
                k = np.interp(f_vector, metal_data['freq_data'], metal_data['k_data'])
                n_matrix[i, :] = n + k * 1j
            else:
                n_complex_20um = fdtd_solver.get_index(str(material_name), freq_20um)
                # Broadcast the 20um index across the frequency vector
                n_matrix[i, :] = np.full(num_freqs, n_complex_20um[0])
        return n_matrix
