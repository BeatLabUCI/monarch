import numpy as np
import os
from .solvers import rk4_wrapper, initialize_solvers_volumes
from .heart import set_total_wall_volumes_areas, unloaded_heart_volume, get_wall_thickness
from .import_export import store_converged_sol, export_growth_sim, import_beat_sim, export_beat_sim, import_pars
from .utils import get_outputs, change_pars
from .growth import initialize, grow, update_circ_heart, store
from pathlib import Path


class Monarch(object):
    """
    Main class, initializing and linking all main functions
    :param model_pars: Input file containing model parameters
    """

    def __init__(self, model_pars):

        # Preset parameter classes
        self.model_pars = model_pars
        self.circulation = None
        self.capacitances = None
        self.resistances = None
        self.solver = None
        self.heart = None
        self.pericardium = None
        self.growth = None
        self.time = None
        self.volumes = None
        self.pressures = None
        self.outputs = None

        # Import self parameters and assign to self class
        self.import_pars(model_pars)

        # Initialize volumes and pressures
        self.volumes = np.zeros((self.solver.n_inc, 8))
        self.pressures = np.zeros((self.solver.n_inc, 8))

        # Set compartment names
        self.compartments = {'PV': 0, 'LA': 1, 'LV': 2, 'SA': 3, ' SV': 4, 'RA': 5, 'RV': 6, 'PA': 7}
        self.compartment_names = list(self.compartments.keys())

        # Set walls
        self.heart.rm = np.zeros((self.solver.n_inc, 5))  # For each wall, ventricles and atria
        self.heart.xm = np.zeros((self.solver.n_inc, 3))  # Only for ventricles
        self.heart.ys_store = np.zeros(self.solver.n_inc)  # Septal height
        self.heart.walls = ['LFW', 'RFW', 'SW', 'LA', 'RA']

        # patches
        self.heart.n_patches_tot = self.heart.patches.size
        self.heart.lab_f = np.ones((self.solver.n_inc, self.heart.n_patches_tot))
        self.heart.sig_f = np.zeros((self.solver.n_inc, self.heart.n_patches_tot))

        # Find ventricular and atrial patches
        self.heart.i_ventricles = np.logical_or.reduce(
            (self.heart.patches == 0, self.heart.patches == 1, self.heart.patches == 2))
        self.heart.i_atria = np.logical_or(self.heart.patches == 3, self.heart.patches == 4)

        # Pericardium
        self.pericardium.lab_f = np.zeros(self.solver.n_inc)
        self.pericardium.pressure = np.zeros(self.solver.n_inc)

        # Calculate total heart wall volume and midwall reference area
        set_total_wall_volumes_areas(self)

        # Compute reference volume of heart for pericardial mechanics computations [mm^3]
        self.heart.v_tot_0 = unloaded_heart_volume(self.heart.am_ref_w, self.heart.vw_w)

        # Start at increment 0
        self.solver.inc = 0

        # Set converged solution file
        self.converged_file = Path(__file__).absolute().parent / "converged_solution.json"

    def change_pars(self, pars):
        """Change model parameters"""
        change_pars(self, pars)

    def import_beat_sim(self, filepath, filename):
        """Import results from a previous simulation and reload class"""
        import_beat_sim(self, filepath, filename)

    def export_beat_sim(self, filepath, filename):
        """Export results from a previous simulation and reload class"""
        export_beat_sim(self, filepath, filename)

    def import_pars(self, model_pars):
        """Import model parameters"""
        import_pars(self, model_pars)

    def clear_converged_sol(self):
        """
        Remove last converged solution, this file could cause problems if the inputs are too different from the last run
        """
        os.remove(self.converged_file)

    def just_beat_it(self, is_growth=False, print_solve=False, file_path=None, file_name=None, use_converged=False):
        """
        Main model function to run simulated heart beat until convergence is reached
        :param self: The heart model to be used in calculation
        :param is_growth: Whether this is part of continuous growth (don't store) or not (store) (default False)
        :param print_solve: Whether to print successful logs (default True)
        :param file_path: Path to the file to export results into (default None)
        :param file_name: Name of the file to export results into (default None)
        :param use_converged: Import and export converged solution, if existing (default True)
        """

        # Check if neither nor both filename and filepath are passed as input
        if file_name and not file_path:
            raise Exception("Filepath was not specified")
        elif file_path and not file_name:
            raise Exception("Filename was not specified")

        # Set state variables and initial guesses. Load from previous converged solution if existing. If simulating
        # growth, this information is already stored in the model class
        if not is_growth:
            initialize_solvers_volumes(self, use_converged)

        # Initial estimate: set final volumes equal to initial volumes
        self.growth.is_growth = is_growth
        self.volumes[-1, :] = self.volumes[0, :]

        # Initiate rK4 loop: first solver iteration and cardiac cycle time increment
        self.solver.iter = 0
        is_transient_state = True

        while is_transient_state and (self.solver.iter < self.solver.iter_max):

            # Determining initial volumes from ending volumes in order to reach steady-state (i.e. volumes at beginning
            # of  cardiac cycle should be the same as those at the end of the cycle for every compartment if we're at
            # steady state). Fix "Volume leakage" by ensuring sum is equal to SBV
            self.volumes[0, :] = self.volumes[-1, :] * self.circulation.sbv / np.sum(self.volumes[-1, :])

            # Main part of the solver, estimate compartmental pressures and volumes
            # for each time point (starting from second point) throughout the cardiac cycle
            # This code is compiled in machine code by the numba package. To debug the code, you need to comment out the
            # @njit decorator, restart the jupyter notebook and run the code again. The numba accerated code is 25 times
            # faster than the pure python implementation.
            rk4_wrapper(self)

            # Relative errors for each compartment
            errors = (np.abs(self.volumes[-1, :] - self.volumes[0, :])) / self.volumes[0, :]

            # Check for convergence
            is_transient_state = any(errors > self.solver.cutoff)

            if not is_growth and print_solve:
                errors_str = np.array2string(errors, formatter={'float_kind': lambda errors: "%.2e" % errors})
                print("Iteration " + str(self.solver.iter) + ":\t" + errors_str[1:-1])

            # Increase solver iteration counter
            self.solver.iter += 1

        # Throw warnings for model malfunctions, only export solution if all checks are passed
        if np.isnan(self.volumes).any():
            if print_solve:
                print("Warning: NaNs were encountered during the simulation, model results were not saved")
        # Warn if convergence was not reached in the maximum allowed number of iterations
        elif self.solver.iter >= self.solver.iter_max:
            if print_solve:
                print("Warning: Maximum allowed number of iterations has been reached, model results were not saved")
        elif not is_growth:
            # Detect negative pressures
            if np.any(self.pressures < 0):
                is_neg = np.where((self.pressures < 0).any(axis=0))[0]
                neg_compartments = [self.compartment_names[i] for i in is_neg]
                if print_solve:
                    print("Warning: Negative pressures were encountered in compartments " + str(neg_compartments))
            # Detect negative volumes
            if np.any(self.volumes < 0):
                is_neg = np.where((self.volumes < 0).any(axis=0))[0]
                neg_compartments = [self.compartment_names[i] for i in is_neg]
                if print_solve:
                    print("Warning: Negative volumes were encountered in compartments " + str(neg_compartments))
            if print_solve:
                print("Steady-state circulation established")

            # Store converged solution
            store_converged_sol(self)

            # Calculate model readouts (EDV, ESV, SBP, etc.)
            self.outputs = get_outputs(self)

            # Export result if desired
            if file_name and file_path:
                self.export_beat_sim(file_path, file_name)

    def let_it_grow(self, file_path=None, file_name=None, use_converged=False, print_solve=True):
        """
        Main growth function to run growth simulation
        :param self: The heart model to be used in calculation
        :param file_path: Path to the file to export results into (default None)
        :param file_name: Name of the file to export results into (default None)
        :param use_converged: Import and export converged solution, if existing (default True)
        :param print_solve: Whether to print successful logs (default True)
        :param growth_type: Type of growth to be simulated (default isotropic)
        """

        # Check if neither or both filename and filepath are passed as input
        if file_name and not file_path:
            raise Exception("Filepath was not specified")
        elif file_path and not file_name:
            raise Exception("Filename was not specified")

        # Initialize growth simulation
        initialize(self, use_converged)

        # for i_g in tqdm(range(0, model.growth.n_g), desc='Growing', ncols=80):
        for i_g in range(self.growth.n_g):
            self.growth.i_g = i_g

            # Grow after acute time step
            if i_g > 1:
                grow(self)

            # Update circulation and heart
            update_circ_heart(self)

            # Simulation heart beat
            self.just_beat_it(is_growth=True, print_solve=print_solve)

            # Store growth
            store(self)

        # Export growth simulation results
        if file_path and file_name:
            if np.isnan(self.volumes).any():
                print("Warning: NaNs were encountered during the simulation, model results were not saved")
            else:
                export_growth_sim(self, filepath=file_path, filename=file_name)

