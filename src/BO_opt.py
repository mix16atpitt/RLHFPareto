import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from fdtd_solver import FDTDSolver
import random
import csv
import os
import torch
from botorch.models import FixedNoiseGP, ModelListGP, MixedSingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.models.transforms import Standardize
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf_mixed
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
import sys
from botorch import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
import warnings 
from linear_operator.utils.cholesky import NumericalWarning
from botorch.exceptions.warnings import InputDataWarning, OptimizationWarning, BotorchWarning
from linear_operator.utils.warnings import NumericalWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
from compute_SE import SESimulation
from compute_Pnet import PnetSimulation
from datetime import datetime
# Get today's date in the format YYYY-MM-DD
today_date = datetime.now().strftime("%Y-%m-%d")

OBJ = "PCOOL"
OBJ_COOL_SCALE_FACTOR = 100
MATERIALS = ['TiO2 (Titanium Dioxide) - Siefke', 'Ni (Nickel) - Palik', 'SiO2 (Glass) - Palik', 'Si3N4 (Silicon Nitride) - Luke', 'Pd (Palladium) - Palik', 'Cr (Chromium) - CRC', 'Ag (Silver) - CRC', 'Al (Aluminium) - CRC', 'Ti (Titanium) - Palik', 'W (Tungsten) - CRC', 'Al2O3 - Palik', 'TiN - Palik']
# MATERIALS = ['TiO2 (Titanium Dioxide) - Siefke', 'GaAs - Palik', 'SiO2 (Glass) - Palik', 'Si (Silicon) - Palik', 'Si3N4 (Silicon Nitride) - Luke', 'Cu (Copper) - CRC', 'Cr (Chromium) - CRC', 'Ag (Silver) - CRC', 'Al (Aluminium) - CRC']   # Replace with actual materials
# n_layers = 10  # Number of layers
n_materials = len(MATERIALS)  # Number of materials
# cont_bounds = torch.tensor([[15e-9] * n_layers, [400e-9] * n_layers])  # Continuous bounds
# cat_bounds = torch.tensor([[0] * n_layers, [n_materials - 1] * n_layers])  # Categorical bounds, treated as continuous for sampling
NOISE_SE = torch.tensor([1e-3, 1e-3])
MC_SAMPLES = 128
NUM_RESTARTS = 5
RAW_SAMPLES = 256

def encode_material(material, materials_list):
    """
    Encode a material string to an integer based on its position in the materials list.
    
    Parameters:
    - material: The material string to encode.
    - materials_list: The list of all possible materials.
    
    Returns:
    - The integer encoding of the material.
    """
    return materials_list.index(material)

def decode_material(index, materials_list):
    """
    Decode an integer back to its corresponding material string.
    
    Parameters:
    - index: The integer index to decode, which may come as a numpy.float32.
    - materials_list: The list of all possible materials.
    
    Returns:
    - The string representation of the material.
    """
    return materials_list[int(index)]


# Initialize a single FDTDSolver instance
fdtd_solver = FDTDSolver()

def objective_function(X, n_layers):
    # Split X into materials and thicknesses
    layer_materials_encoded = X[:n_layers]
    layer_materials = ['Air'] + [decode_material(index, MATERIALS) for index in layer_materials_encoded] + ['Si (Silicon) - Palik']
    layer_thicknesses = np.array([0] + list(X[n_layers:]) + [0])

    # print("layer_materials:", layer_materials)
    # print("layer_thicknesses:", layer_thicknesses)

    se_simulation = SESimulation(layer_materials, layer_thicknesses, fdtd_solver)
    SE = se_simulation.run_simulation()

    pnet_simulation = PnetSimulation(layer_materials, layer_thicknesses, fdtd_solver)
    Pcool, P_net = pnet_simulation.run_simulation()
    if OBJ == "PCOOL":
        return [SE, OBJ_COOL_SCALE_FACTOR * Pcool]
    else:
        return [SE, -P_net]
    

class BayesianOptimization:
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.cat_bounds = torch.tensor([[0] * n_layers, [n_materials - 1] * n_layers])
        self.cont_bounds = torch.tensor([[15e-9] * n_layers, [400e-9] * n_layers])
        self.ref_point = torch.tensor([0, 0])
        self.cat_dim = [i for i in range(n_layers)]
        self.num_objectives = 2
        self.noise_se = NOISE_SE
        self.train_x = None
        self.cat_x = None
        self.cont_x = None
        self.train_obj = None
        self.model = None
        self.mll = None
        # self.max_evaluations = max_evaluations

    def generate_initial_data(self, n = 10):
        # Draw Sobol samples for continuous variables
        cont_samples = draw_sobol_samples(bounds= self.cont_bounds, n=n, q=1).squeeze(1)
        # Directly sample indices for categorical variables
        cat_samples = torch.randint(low=0, high=n_materials, size=(n, self.n_layers))
        # Combine categorical and continuous samples, with categorical samples first
        train_x = torch.cat((cat_samples, cont_samples), dim=1)

        # Evaluate the objective function on each sample
        train_obj = torch.tensor([objective_function(sample.numpy(), self.n_layers) for sample in train_x])

        # print("Shape of train_x:", train_x.shape)  # Print shape of train_x
        # print("Shape of train_obj:", train_obj.shape)  # Print shape of train_obj
        self.cat_x = cat_samples
        self.cont_x = cont_samples
        self.train_x = train_x
        self.train_obj = train_obj
        # return cat_samples, cont_samples, train_obj

    def initialize_model(self):
        # Normalize train_x
        cont_train_x_normalized = normalize(self.cont_x, bounds=self.cont_bounds)
        # Combine back with categorical data
        train_x_normalized = torch.cat([self.cat_x, cont_train_x_normalized], dim=1)
        
        # Convert train_x_normalized to the same dtype as train_obj for consistency
        train_x_normalized = train_x_normalized.to(dtype=torch.float64)
        
        # print("Shape of normalized train_x:", train_x_normalized.shape)

        models = []
        for i in range(self.num_objectives):
            # Extracting the i-th objective
            train_y = self.train_obj[..., i : i + 1]
            
            # Ensure train_y is of type torch.float64
            train_y = train_y.to(dtype=torch.float64)
            
            # print(f"Shape of train_y for objective {i}:", train_y.shape)

            # Setting the noise level for the i-th objective
            train_yvar = torch.full_like(train_y, self.noise_se[i] ** 2)
            
            # Ensure train_yvar is of type torch.float64
            train_yvar = train_yvar.to(dtype=torch.float64)
            
            print(f"Shape of train_yvar (noise variance) for objective {i}:", train_yvar.shape)

            # Creating a FixedNoiseGP model for the i-th objective
            gp_model = MixedSingleTaskGP(
                train_x_normalized, train_y, self.cat_dim, train_yvar, outcome_transform=Standardize(m=1)
            )
            models.append(gp_model)

        # Combining all GP models into a ModelListGP
        self.model = ModelListGP(*models)

        # Creating the Marginal Log Likelihood for the combined model
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)

    def optimize_qnehvi_and_get_observation(self, sampler):
        """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
        # partition non-dominated space into disjoint rectangles
        standard_bounds = torch.zeros(2, self.train_x.shape[-1])
        standard_bounds[1, :self.n_layers] = torch.tensor(len(MATERIALS) - 1)
        standard_bounds[1, self.n_layers:] = 1

        normalized_cont_variables = normalize(self.train_x[:, self.n_layers:], bounds=self.cont_bounds)
        # Concatenate the unnormalized categorical variables with the normalized continuous variables
        X_baseline = torch.cat([self.train_x[:, :self.n_layers], normalized_cont_variables], dim=1)

        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model = self.model,
            ref_point = self.ref_point.tolist(),  # use known reference point
            X_baseline = X_baseline,
            prune_baseline = True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler = sampler,
        )
            # Generate initial conditions
        candidates, _ = optimize_acqf_mixed(
            acq_function = acq_func,
            bounds = standard_bounds,
            q = 1,
            num_restarts = NUM_RESTARTS,
            raw_samples = RAW_SAMPLES,  # used for intialization heuristic
            # options={"batch_limit": 12, "maxiter": 200},
            fixed_features_list = [{}],
        )
        # observe new values
        # Splitting the candidates into continuous and categorical parts
        categorical_candidates = candidates[..., :len(self.cat_dim)]
        continuous_candidates = candidates[..., len(self.cat_dim):]

        # Unnormalize only the continuous part
        continuous_candidates_unnorm = unnormalize(continuous_candidates.detach(), bounds=self.cont_bounds)

        # Recombine the continuous and categorical parts
        new_x = torch.cat([categorical_candidates, continuous_candidates_unnorm], dim=-1)
        self.cat_x = torch.cat([self.cat_x, categorical_candidates])
        self.cont_x = torch.cat([self.cont_x, continuous_candidates_unnorm])
        self.train_x = torch.cat([self.train_x, new_x])
        return new_x

    def train_iteration(self, iterations = 50):
        for _ in range(1, iterations + 1):
            # fit the models
            fit_gpytorch_mll(self.mll)

            # define the qEI and qNEI acquisition modules using a QMC sampler
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

            # optimize acquisition functions and get new observations
            new_train_x = self.optimize_qnehvi_and_get_observation(sampler)

            # Evaluate the objective function on each sample
            new_train_obj = torch.tensor([objective_function(sample.numpy(), self.n_layers) for sample in new_train_x])
            self.train_obj = torch.cat([self.train_obj, new_train_obj])
            # reinitialize the models so they are ready for fitting on next iteration
            # Note: we find improved performance from not warm starting the model hyperparameters
            # using the hyperparameters from the previous iteration
            self.initialize_model()

def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the Pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of individuals on the Pareto front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each individual
    for i in range(population_size):
        # Compare scores against the rest
        for j in range(population_size):
            # Check if our 'i' point is dominated by 'j'
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # If it is, then this point cannot be on the Pareto front
                pareto_front[i] = 0
                break
    # Return ids of those on the Pareto front
    return population_ids[pareto_front]

def main(num_layers):
    n_layers=num_layers
    bo = BayesianOptimization(n_layers)
    bo.generate_initial_data()
    bo.initialize_model()
    bo.train_iteration(500)

    # Identify Pareto front
    pareto_indices = identify_pareto(bo.train_obj.numpy())
    pareto_parameters = bo.train_x[pareto_indices]
    pareto_objectives = bo.train_obj[pareto_indices]
    # Negate the P_net values to convert them back to positive for the plot
    if OBJ == "PNET":
        pareto_objectives[:, 1] = -pareto_objectives[:, 1]
        bo.train_obj[:, 1] = -bo.train_obj[:, 1]

    decoded_materials = [[decode_material(index.item(), MATERIALS) for index in row[:n_layers]] for row in pareto_parameters]
    thicknesses = pareto_parameters[:, n_layers:].tolist()

    # Plot the Pareto frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(bo.train_obj[:, 0], bo.train_obj[:, 1] / OBJ_COOL_SCALE_FACTOR, c="blue", label="All Points")
    plt.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1] / OBJ_COOL_SCALE_FACTOR, c="red", label="Pareto Frontier")
    plt.title("Pareto Frontier")
    plt.xlabel("Objective 1 (SE)")
    if OBJ == "PCOOL":
        plt.ylabel("Objective 2 (P_cool)")
    else:
        plt.ylabel("Objective 2 (P_net)")
    plt.legend()
    plt.grid(True)

    # Save the plot
    if OBJ == "PCOOL":
        plot_path = f'res/BO_new_material/{today_date}/{num_layers}layers/PCOOL/pareto_frontier_plot_{today_date}.png'
    elif OBJ == "PNET":
        plot_path = f'res/BO_new_material/{today_date}/{num_layers}layers/PNET/pareto_frontier_plot_{today_date}.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)

    # Saving the Pareto frontier in a CSV file
    if OBJ == "PCOOL":
        csv_path = f'res/BO_new_material/{today_date}/{num_layers}layers/PCOOL/pareto_frontier_{today_date}.csv'
    elif OBJ == "PNET":
        csv_path = f'res/BO_new_material/{today_date}/{num_layers}layers/PNET/pareto_frontier_{today_date}.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Writing headers
        if OBJ == "PCOOL":
            headers = [f'Material{i}' for i in range(1, n_layers + 1)] + \
                [f'Thickness{i}' for i in range(1, n_layers + 1)] + \
                ['Objective1 (SE)', 'Objective2 (P_cool)']
        if OBJ == "PNET":
            headers = [f'Material{i}' for i in range(1, n_layers + 1)] + \
                [f'Thickness{i}' for i in range(1, n_layers + 1)] + \
                ['Objective1 (SE)', 'Objective2 (P_net)']
        writer.writerow(headers)

        # Writing data rows
        for materials, thickness, objectives in zip(decoded_materials, thicknesses, pareto_objectives.tolist()):
            if OBJ == "PCOOL":
                objectives[1] /= OBJ_COOL_SCALE_FACTOR  # Assuming the second objective was scaled
            row = materials + thickness + objectives
            writer.writerow(row)

    all_data_csv_path = os.path.join(os.path.dirname(csv_path), 'all_bo_data.csv')
    with open(all_data_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Headers for all BO data
        headers = [f'Material{i}' for i in range(1, n_layers + 1)] + \
            [f'Thickness{i}' for i in range(1, n_layers + 1)] + \
            ['Objective1 (SE)', 'Objective2']
        if OBJ == "PCOOL":
            headers[-1] += ' (P_cool)'
        elif OBJ == "PNET":
            headers[-1] += ' (P_net)'
        writer.writerow(headers)

        # Writing all BO data rows
        for i in range(bo.train_x.shape[0]):
            materials = [decode_material(index.item(), MATERIALS) for index in bo.train_x[i, :n_layers]]
            thicknesses = bo.train_x[i, n_layers:].tolist()
            objectives = bo.train_obj[i].tolist()
            if OBJ == "PCOOL":
                objectives[1] /= OBJ_COOL_SCALE_FACTOR  # Assuming the second objective was scaled
            row = materials + thicknesses + objectives
            writer.writerow(row)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_layers = int(sys.argv[1])  # Convert the argument to an integer
        main(num_layers)
    else:
        print("Please provide the number of layers as an argument.")
