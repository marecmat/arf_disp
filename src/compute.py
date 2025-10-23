import os
import sys
import numpy as np
import warnings
from traceback import format_exc

from muller import muller 
from utils import list_of_lists_to_array

if os.uname()[1] == 'helmholtz':
    sys.path.append("/export/home/laum/s170098/")

try: 
    from phd_lib import remote
    use_remote_tools = True
except ModuleNotFoundError:
    use_remote_tools = False

try: 
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = False

class compute_dispersion:

    def __init__(self, method, muller_params, args={}, directory='',
            det_type='det', nb_sol=0, max_recompute=3, map_size=(20, 20), 
            grid_interval=((-100, 100), (-100, 100)), URL='',
            mode_shapes=False, verbose=2):
        
        self.method = method
        self.method_name = str(method)[1:].split(' ')[0].split('.')[-1]
        self.muller_params = muller_params
        self.args = args
        self.nb_sol = nb_sol
        self.roots = []
        self.max_recompute = max_recompute
        self.len_kr = map_size[0]
        self.len_ki = map_size[1]
        self.kr_interval = grid_interval[0]
        self.ki_interval = grid_interval[1]
        self.mode_shapes = mode_shapes
        self.directory = directory
        self.verbose = verbose
        self.URL = URL
        if verbose >= 1 and type(tqdm) is type(os): 
            self.indic = tqdm
        else: np.real
        
        if verbose < 2: 
            warnings.filterwarnings(action='ignore', category=RuntimeWarning)

        if not self.directory == '' and not os.path.exists(directory): 
            os.makedirs(directory) 

        # Select the function that will be used as the main function call
        if det_type == 'det':
            def matrix_det(k, method):
                matrix = method.get_matrix(k)
                return np.linalg.det(matrix)
            self.det_type = matrix_det
            
        elif det_type == 'neumann': 
            p = self.method.nb_bloch_waves
            mbmt_slice = np.s_[4*(2*p + 1):6*(2*p + 1), 1*(2*p + 1):5*(2*p + 1)]
            remove_first_last_col = np.s_[:, (2*p + 1):3*(2*p + 1)]
        
            def matrix_det_neum(k, method):
                matrix = method.get_matrix(k)
                matrix = matrix[mbmt_slice][remove_first_last_col]
                return 1/np.linalg.det(matrix)

            self.det_type = matrix_det_neum

        elif det_type == 'expr':
            def scal_det(k, method):
                return method.get_matrix(k)
            self.det_type = scal_det

        else: 
            raise NameError('Invalid type of det')

        return None
    

    def get_output(self):
        # helps outputting a npy file for a given iteration
        # basically storing in a dict every class attribute
        output = {}
        for attribute, value in self.__dict__.items():
            if not callable(value) and attribute != "method":
                output[attribute] = value
        try:
            output['nb_bloch'] = self.method.nb_bloch_waves
            output['order'] = self.method.order
            output['termination'] = self.method.termination
        
        except AttributeError: 
            pass

        return output 

    def compute_map_minima(self):
        carte = np.zeros((self.len_kr, self.len_ki), dtype=complex)
        kr = np.linspace(*self.kr_interval, self.len_kr)
        ki = np.linspace(*self.ki_interval, self.len_ki)

        for i, ikr in enumerate(kr):
            for j, iki in enumerate(ki):
                k = ikr + 1j*iki
                carte[i, j] = self.det_type(k, self.method)

        # Computing map local minimas
        KR, KI = np.meshgrid(kr, ki)
        grid, carte = KR + 1j*KI, np.abs(carte.T)
        mapmin = ((carte <= np.roll(carte,  1, 0)) & (carte <= np.roll(carte, -1, 0)) &
                  (carte <= np.roll(carte,  1, 1)) & (carte <= np.roll(carte, -1, 1)))
        
        self.kr = kr
        self.kr = ki
        self.map = carte 
        return grid[mapmin]
    

    def compute_iteration(self, map):
        if len(self.muller_params['guesses']) == 0 or map == True:
            self.minima = self.compute_map_minima()
            if len(self.minima) == 0:
                raise RuntimeError('no minima found for this map !')
            else:
                self.muller_params['guesses'] += [*self.minima]
        
        self.reduce_guesses()
        if self.verbose >= 2: print('init muller with guesses', self.muller_params['guesses'])
            
        root, iter = muller(self.det_type, args=(self.method), **self.muller_params)
        return root, iter

    def reduce_guesses(self):
        if len(self.muller_params['guesses']) >= 0:
            self.muller_params['guesses'] = list(np.array(self.muller_params['guesses'])\
                [np.unique(np.round(np.array(self.muller_params['guesses']), 0), return_index=True)[1]])
        return None

    def compute_one_freq(self, ii, roots, iters):
        try:
            if len(self.roots) >= self.nb_sol:
                self.muller_params['guesses'] = []

            self.freq = self.method.frequencies[ii]
            self.method.update_frequency(2*np.pi*self.method.frequencies[ii])
            # self.muller_params['guesses'] 
            
            if ii == 0:  
                self.muller_params['prevRoots'] = np.array([], dtype=complex)
            else: 
                self.muller_params['guesses'] += [*roots[-1], *self.minima]
                self.muller_params['prevRoots'] = self.roots

            root, iter = self.compute_iteration(map=False)
            self.roots = root
            self.iters = iter

            if len(self.roots) < self.nb_sol:
                # Re-run the program if missing solution, with a computation of the map
                if self.verbose > 1:
                    print(f'{self.method_name}, ({self.directory}): found {len(self.roots)} solutions, \n {self.roots} \n running map')
                self.muller_params['guesses'] += self.roots

                # redo the calculation with a map computation
                root, iter = self.compute_iteration(map=True)
                if len(self.roots) > 0:
                    for jj, r in enumerate(root):
                        if np.min(np.abs(r - self.roots)) > self.muller_params['eps2']:
                            self.roots.append(r)
                            self.iters.append(iter[jj])
                else: 
                    self.roots += root
                    self.iters += iter
                    
                self.muller_params['prevRoots'] = np.concatenate((self.muller_params['prevRoots'], np.array(root)))

            if self.verbose > 1:
                print(f'{self.method_name}, ({self.directory}): found roots at f={self.method.frequencies[ii]},\n {self.roots} \n {self.iters}')
            
            roots.append(self.roots)
            iters.append(self.iters)

            if self.mode_shapes:
                # Compute the mode shape if possible
                try:
                    self.mode_shape = {'computed':True, 'data':[], 'out_amplitudes':[], 'matrix_amplitudes':[]}
                    for r in self.roots:
                        self.method.get_mode_shape(self.method.get_matrix(kx=r))
                        self.mode_shape['matrix_amplitudes'].append(self.method.matrix_amplitudes)
                        self.mode_shape['data'].append((self.method.vtk_points, self.method.vtk_triangle, self.method.scalars))
                        self.mode_shape['out_amplitudes'].append(self.method.out_amplitudes)
                except AttributeError:
                    # unable to compute the mode shapes due to missing attributes of the called function
                    self.mode_shapes = 0

            if not self.directory == '':
                np.save(f"{self.directory}/out_{ii}.npy", self.get_output())
                if self.verbose > 1: 
                    print(f"{self.method_name}, ({self.directory}): saved file {ii}/{len(self.method.frequencies) - 1}")
                if os.uname()[1] == 'helmholtz' and self.URL != '' and use_remote_tools:
                    remote.send_data_somewhere(f'{self.directory}', URL=self.URL, file_list=[f"out_{ii}.npy"])
                if self.method.frequencies[ii] == self.method.frequencies[-1] and use_remote_tools: 
                    remote.send_discord_notification(f'{os.uname()[1]} | {self.method_name} | ({self.directory})', f"{self.args}")
            
        except Exception as e:
            if os.uname()[1] == 'helmholtz' and use_remote_tools:
                remote.send_discord_notification(f'{self.method_name}, ({self.directory}):  exception {e}', f"{self.args, self.directory} threw: \n{format_exc()}")
            else: 
                print(e, format_exc())
            raise RuntimeError

        return roots, iters

    def compute(self):
        roots, iters = [], []

        for ii, self.freq in enumerate(self.indic(self.method.frequencies)):
            roots, iters = self.compute_one_freq(ii, roots, iters)

        roots = list_of_lists_to_array(list(roots), dtype=complex)
        return roots[np.isfinite(roots)], np.real(np.tile(self.method.frequencies, (roots.shape[1], 1)).T[np.isfinite(roots)])