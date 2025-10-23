import numpy as np 

"""
Core of the muller method with the iteration procedure and basic checks for valid roots
"""

def muller(
        f, args, guesses, interval, dx=1, eps1=1e-5, eps2=1e-8, eps3=1e6, nbIter=100, prevRoots=[],
        deflated=False, verbose=False):
    
    """
    Parameters
    --------
    f : function(x, args)
        complex-valued function to search roots of z on
    args : tuple (N, )
        additional arguments to use with the function

    guesses: list of guesses 

    'interval': interval where roots should be sought

    dx : float, optional
        calculation step. Muller method uses 3 points to perform 
        each iteration. The initial points are xr-dx, xr, xr+dx.
    eps1 : float 
        Stopping criterion
    eps2 : float 
        Tolerance for 2 sorting roots located too close to each other 
    eps3: float
        Allowed max variance for the evaluated x_i (too high means that the iteration process is diverging)

    nbIter : int, optional 
        max number of iterations to be performed
    
    Returns
    -------
    roots : array(dtype=complex)
        computed roots
    iters : ndarray 
        array with same shape as roots, iteration number for each converged root
    """
    
    
    def mullerIteration(p0, p1, p2, prevRoots, deflated=False):
        """
        Parameters
        ----------
        p0 : complex
            initial point 1 of the search
        p1 : complex
            initial point 2 of the search
        p2 : complex
            initial point 3 of the search
        prevRoots: array of complex
            previously found roots in order to compute the deflated function
        deflated : boolean, optional
            compute using deflated function fp or function f. Any way, 
            fp3 is always computed as it's a convergence criterion
        conv_warn : boolean, optional
            print a warning in the output is the root does not converge.
            Moslty for debugging purposes

        Returns
        -------
        p3 : complex
            the computed root value. If the method doesnt converge, 
            the last iteration is returned. It can be checked, since 
            in this case, i > nbIter
        """

        conv = False
        if verbose:
            print('pi', p0, p1, p2)
        f0, f1, f2 =  f(p0, args),  f(p1, args),  f(p2, args)
        # print('f', f0, f1, f2)

        for i in range(1,  nbIter+1):
            # Compute the coefficients of the classical Muller algorithm
            a = ((p1 - p2)*(f0 - f2) - (p0 - p2)*(f1 - f2))/((p0 - p2)*(p1 - p2)*(p0 - p1))
            b = ((p0 - p2)**2 *(f1 - f2)-(p1 - p2)**2*(f0 - f2))/((p0 - p2)*(p1 - p2)*(p0 - p1))
            c = f2

            p3 = p2 - 2*c/(b + np.sign(b)*np.sqrt(b**2 - 4*a*c))

            # First stopping criterion
            if np.abs((p3 - p2)/p3) < eps1:
                conv = True
                break 

            elif np.isnan([p0, p1, p2, p3]).any():
                # Keep the loop from making function calls until the end of the loop 
                # if the guess was computed as None 
                conv = False
                break       

            # Second stopping criterion: 
            elif np.var([p0, p1, p2, p3]) > eps3:
                # Test if the path taken will diverge
                conv = False
                break       

            # Swap values for the next iteration
            p0, p1, p2 = p1, p2, p3
            f0, f1, f2 = f(p0, args), f(p1, args), f(p2, args)
            if deflated: 
                f0 /= np.prod(p0 - prevRoots)
                f1 /= np.prod(p1 - prevRoots)
                f2 /= np.prod(p2 - prevRoots)
        
        # # ax.semilogy(np.abs(abs_var))
        # if warnings: print(f"no root found after {nbIter} it., f(p3={p3})={f3}")
        return p3, i, conv

    def validRoot(root, roots, conv):
        return all((
            conv == True,
            np.min(np.abs(root - roots)) > eps2,
            (np.real(root) > interval[0][0]),
            (np.real(root) < interval[0][1]),
            (np.imag(root) > interval[1][0]),
            (np.imag(root) < interval[1][1])
        ))
    
    r = 0


    roots, iters = [0+0j], [0]
    if verbose:
        print(f"Muller \t: iters./(total found roots)/total iters.")
    for jj, xr in enumerate(guesses): 
        if verbose:
            print(f"Muller \t: {jj}({len(roots)})/{len(guesses)}")
        root, it, conv = mullerIteration(xr - dx, xr, xr + dx, np.array([*roots[:r], *prevRoots]), deflated=deflated)
        if validRoot(root, roots, conv) and r < len(guesses):
            roots.append(root)
            iters.append(it)

    return roots[1:], iters[1:]