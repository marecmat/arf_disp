[![DOI](https://zenodo.org/badge/1081885293.svg)](https://doi.org/10.5281/zenodo.17425638)

# a root-finding method for wave dispersion problems

This code has been developed during my PhD thesis and has been used to evaluate full dispersion relations of waves in various configurations. 

## Dependancies 
- numpy
- (not necessary) tqdm: for progress bars
- (not necessary) my messy library of python tools: [phd_lib](https://github.com/marecmat/phd_lib)

## Use the code 
The code has been originally designed to interface the 'DispersionPwProblem' from [pyPLANES](https://github.com/OlivierDAZEL/pyPLANES/tree/eTMM), but it can be used if the problem is declared using a similar layout. 

### Formatting of your problem
The code interfaces a class that should describe the problem for which you want to solve dispersion relation. Snippet for the code structure should be as,

```python
from compute import compute_dispersion

class some_method: 
    def __init__(self, **params):
        # initialise whatever you need here
    
    def update_frequency(self, omega):
        # Update the frequency-dependant values here
        return 
    
    def get_matrix(self, k):
        # Here, create the matrix for a given k
        # Note that get_matrix can also generate a single expression. If so, 
        # make use of the self.det_type accordingly in the declaration of 
        # `compute_dispersion` as either 'det' or 'expr"
        return matrix

    def any_other_useful_method(self, ...):
        pass


if __name__ == '__main__':
    muller_params = {
        "dx": 1+1j, 
        "eps1": 1e-16, 
        "eps2": 1e-8, 
        "guesses": [], 
        "nbIter": 200, 
        "interval": [[0, 158], [-80, 0]], "deflated": 0
    },

    method = some_method('any parameter')
    compute_dispersion(method, muller_params, directory='/your/path/here', 
            nb_sol=3, map_size=(20, 20), grid_interval=muller_params['interval'], verbose=2)


``` 

## References
This code has been used in the following papers: 

<!-- - M. Maréchal, O.Dazel,  V. Romero-García and J.-P. Groby, "Dispersion relations of leaky guided waves in metaporous layers", *in preparation for Phys. Rev. B* (2025) -->

- M. Maréchal, A. Geslain, J.-P. Groby, V. Romero-García, O. Dazel, A general spectral collocation method for computing the dispersion relations of guided acoustic waves in multilayer dissipative structures. [**J. Appl. Phys.**](https://doi.org/10.1063/5.0242405) 137 (10) 104902. 2025