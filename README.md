# a root-finding method for wave dispersion problems

This code has been developed during my PhD thesis and has been used to evaluate full dispersion relations of waves in various configurations. 

## Dependancies 
TODO

## Use the code 

### Formatting of your problem
The code interfaces a class that should describe the problem for which you want to solve dispersion relation. Let this class be called `some_method`, the code should be structured as, 

```python

class some_method: 
    def __init__(self, **params):
        # initialise w/e here
    
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
``` 

## References

## TODO