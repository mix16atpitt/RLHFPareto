import pandas as pd
from scipy.interpolate import interp1d

class DataProcessor:
    @staticmethod
    def read_csv(file_path):
        return pd.read_csv(file_path)

    @staticmethod
    def interpolate_data(x, y, new_x):
        interp_func = interp1d(x, y, kind='linear', fill_value="extrapolate")
        return interp_func(new_x)
