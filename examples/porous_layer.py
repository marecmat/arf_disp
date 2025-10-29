import sys
import numpy as np

sys.path.extend(['./src/', '../src/'])
from porous import JCA

def fluid_eq_fluid_coupled(k, params, d=True, source=False, return_ks=False):
    om, rho1, keq, rhof, cf, H = params

    k21 = np.sqrt(keq**2 - k**2)
    k2f = np.sqrt((om/cf)**2 - k**2)
    
    e1p = np.exp(1j*k21*H)
    a1 = k21/rho1
    af = k2f/rhof

    m = np.array([
        [     1,   1*e1p,   0, -1 ],
        [    a1, -a1*e1p,   0, af ],
        [   e1p,       1,  -1,  0 ],
        [a1*e1p,     -a1, -af,  0 ]
    ])

    if return_ks: return k2f, k21
    # if d: return det(m)
    elif source: return m, np.array([1, af, 0, 0])
    else: return m


def analytical_expr_fluid(k, params, losses=True, one_side=False, compute_k2=False):
    om, rho1, keq, rhof, cf, H, _, _ = params
    k0 = om/cf
    # fluid layer
    # c1 = cf/2
    # rho1 = rhof/5 #*(1-0.1j)
    # keq = om/c1
    # K_eq = (rho1*c1**2)*(1+0j)
    # keq = om*np.sqrt(rho1/K_eq)
    # lossless porous
    # keq = np.real(keq) + 0j
    # rho1 = np.real(rho1) + 0j

    if compute_k2:
        k2f = k 
        k1 = np.sqrt(k0**2 - k2f**2)
        # if np.imag(k2f) < 0:
        #     k1 *= -1
        k21 = -np.sqrt(keq**2 - k1**2)
    else: 
        k1 = k 
        k21 = np.sqrt(keq**2 - k1**2)
        k2f = -np.sqrt(k0**2 - k1**2) #*-1
        # if np.imag(k1) > np.imag(keq):
        #     k2f *= -1


    if one_side:
        expr = k2f/rhof*np.cos(k21*H) + 1j*k21/rho1*np.sin(k21*H)
    else: 
        expr = 2*(k21*k2f)/(rho1*rhof)*np.cos(k21*H) + 1j*((k21/rho1)**2 + (k2f/rhof)**2)*np.sin(k21*H)
    return expr


def analytical_det_fluid(k, params, compute_k2=True):
    om, rho1, keq, rhof, cf, H, _, _ = params
    k0 = om/cf

    if compute_k2:
        k2f = k 
        k1 = np.sqrt(k0**2 - k2f**2)
        k21 = np.sqrt(keq**2 - k1**2)
    else: 
        k1 = k 
        k21 = np.sqrt(keq**2 - k1**2)
        k2f = -np.sqrt(k0**2 - k1**2)
    
    eta1 = k21/rho1
    eta0 = k2f/rhof
    exp = np.exp(-1j*k21*H)

    matrix = np.array([
        [eta1*exp, -eta1, 0], 
        [1, exp, -1], 
        [eta1, -eta1*exp, eta0]
    ])


    return matrix

class analytical_method:
    def __init__(self, frequencies, fluid, porous, h, conv='+', boundary='coupled'):

        self.fluid = fluid
        self.porous = porous
        self.h = h
        # self.k_x = 0
        self.conv = conv
        self.boundary = boundary
        self.frequencies = frequencies
        
        self.update_frequency(2*np.pi*frequencies[0])
        self.keq = self.params[2]
        self.cf  = self.params[4]
        return None

    def update_frequency(self, om):
        self.om = om
        self.params = list(self.get_fluideq_params(self.om, conv=self.conv))
        # self.params[1] = np.real(self.params[1])
        # self.params[2] = np.real(self.params[2])
        return self.params[2], self.om/self.params[4]

    def get_matrix(self, kx):
        # if self.boundary == 'coupled':
        #     return fluid_eq_fluid_coupled(kx, self.params, d=False)
        if self.boundary == 'expr_one_side':
            return analytical_expr_fluid(kx, self.params, one_side=True)
        elif self.boundary == 'expr_sym':
            return analytical_expr_fluid(kx, self.params, one_side=False)
        # elif self.boundary == 'love-modes':
        #     return love_modes_expr(kx, self.params, one_side=False)
        elif self.boundary == 'det_one_side':
            return analytical_det_fluid(kx, self.params)

        # elif self.boundary == 'p=0':
        #     return free_fluid(kx, self.params, d=False)
        # elif self.boundary == 'u=0':
        #     return rigid_fluid(kx, self.params, d=False)
        else: 
            raise ValueError('invalid boundary type')

    # def get_k2(self, k):
    #     return np.sqrt((self.om/self.cf)**2 - k**2), \
    #            np.sqrt(self.keq**2 - k**2)    

    def get_mode_shape(self, test):
        self.vtk_points = []
        self.vtk_triangle = []
        self.scalars = []
        self.out_amplitudes = []


    def get_fluideq_params(self, om, conv='+'):
        rhof, cf = self.fluid['rho'], self.fluid['cf']
        K_eq, rho_eq = JCA(om, self.porous, saturating_fluid=self.fluid, conv=conv)
        if self.porous['medium_type'] == 'eqf_lossless':
            K_eq = np.real(K_eq) + 1j*np.sign(np.imag(K_eq))*self.porous['loss_amount']*np.abs(np.imag(K_eq))
            rho_eq = np.real(rho_eq) + 1j*np.sign(np.imag(rho_eq))*self.porous['loss_amount']*np.abs(np.imag(rho_eq))
        k_eq = om*np.sqrt(rho_eq/K_eq)
        
        # k_eq = om/342
        # rho_eq = 1.213
        params = (om, rho_eq, k_eq, rhof, cf, self.h, rho_eq, K_eq)
        return params