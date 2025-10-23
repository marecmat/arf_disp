import os
import numpy as np

def list_of_lists_to_array(liste, pad_value=None, dtype=float):
    """
    https://stackoverflow.com/questions/43146266/convert-list-of-lists-with-different-lengths-to-a-numpy-array
    """
    return np.array([[*i, *[pad_value]*(len(max(liste, key=len)) - len(i))] for i in list(liste)], dtype=dtype)

def sort_branches(data, sort_by=None, tol=10):
    # necessite qq corrections manuelles mais marche pas trop mal

    def argsort_with_func(array, func):
        arg = np.argsort(func(array))
        return array[arg]

    if type(data) == tuple:
        freqs = data[0]
        k1 = data[1]

    elif data.shape[1] == 2:
        # Formatted as classical output 2 columns : 
        k1    = data[:, 0]
        freqs = data[:, 1]
    
    else: 
        return None
    
    if sort_by == None : 
        func = lambda x: -np.abs(np.imag(x)) 
    else: 
        func = sort_by

    rev_freqs = np.unique(freqs)
    shapes = np.array([k1[rev_freqs[i] == freqs].shape[0] for i in range(len(rev_freqs))])    
    # am i dumb ? max_len = shapes[np.argmin(1/shapes)] 
    max_len = np.max(shapes)

    initial_points = argsort_with_func(k1[rev_freqs[0] == freqs], func)
    initial_points = np.pad(initial_points, (0, max_len - len(initial_points)), 'constant', constant_values=np.array([0]))
    branches = np.zeros((len(rev_freqs), max_len), dtype=complex)

    branches[0, :] = initial_points
    for i, f in enumerate(rev_freqs[1:]):
        solutions = argsort_with_func(k1[freqs == f], func)
        for j, k in enumerate(branches[i, :]):
            if len(solutions) > 0: 
                diff = np.abs(func(k) - func(solutions))
                if np.min(diff) < tol:
                    # look for minimum distance while keeping sign !!!, real and imag part infos
                    argmin = np.argmin(np.sqrt((np.real(k) - np.real(solutions))**2 \
                                            + (np.imag(k) - np.imag(solutions))**2))
                    sol = solutions[argmin]
                    solutions = np.delete(solutions, np.argwhere(solutions == sol))
                    branches[i+1, j] = sol

        remaining_zeros = np.where(branches[i+1, :] == 0)
        if solutions.shape[0] > 0 and len(remaining_zeros) > 0:
            branches[i+1, remaining_zeros] = solutions
    branches[np.where(branches == 0)] = None
    # first col with freqs and rest is solutions
    return np.vstack((rev_freqs, branches.T)).T


def run_batch(func, params, nb_threads=None):
    """
    runs multiple processes in parallel making use of multiprocessing.Pool
    input: func(params) on which the pool maps every simulation to run
    !!: no output, write the computed data to a file 
    """
    from multiprocessing import Pool, cpu_count

    if os.uname()[1] == 'helmholtz':
        # if running on laum server, do not use all cores
        max_cpu = int(cpu_count()/2)
    else: 
        max_cpu = cpu_count()

    nb_instances = len(params)
    if nb_threads == None:
        if  max_cpu > nb_instances:
            nb_threads = nb_instances
        else:
            nb_threads = max_cpu
    
    with Pool(processes=nb_threads) as p:
        p.map(func, params)
        p.close()

    return None 