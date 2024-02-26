import sys, os
sys.path.append("C:\\Program Files\\Lumerical\\v241\\api\\python\\")
sys.path.append("C:\\Program Files\\Lumerical\\v232\\api\\python\\")
sys.path.append("D:\\Program Files\\Lumerical\\v241\\api\\python\\")
import lumapi

class FDTDSolver:
    def __init__(self):
        self.fdtd = lumapi.FDTD(hide=True)

    def get_index(self, material_name, frequencies):
        # print(frequencies)
        # print("material_name: ", material_name)
        return self.fdtd.getindex(material_name, frequencies)

    def stackrt(self, n_matrix, layer_thicknesses, frequencies, theta):
        return self.fdtd.stackrt(n_matrix, layer_thicknesses, frequencies, theta)
