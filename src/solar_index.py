import numpy as np

class SolarIndex:
    @staticmethod
    def create_matrix(fdtd_solver, layer_materials, f_vector):
        num_layers = len(layer_materials)
        num_freqs = len(f_vector)
        n_matrix = np.zeros((num_layers, num_freqs), dtype=np.complex128)
        for i, material_name in enumerate(layer_materials):
            if material_name == 'Air':
                n_matrix[i, :] = 1
            else:
                n_complex = fdtd_solver.get_index(str(material_name), f_vector)
                n_matrix[i, :] = n_complex[:, 0]
        return n_matrix
