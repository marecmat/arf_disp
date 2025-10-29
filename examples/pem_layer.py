import sys
import numpy as np 
from numpy import sqrt
from numpy.linalg import det

sys.path.extend(['./src/', '../src/'])
from porous import biot, JCA

# sys.path.insert(0, "/Users/mat/These/porous_dispersion_scm/simulations/src/spectral/")
# from funs import cheb_nodes, chebyshev, quadraticVarChange


class analytical_pem:
    def __init__(self, h, frequencies, porous, fluid, case='transmission', mediapack=True):
        self.porous = porous
        self.fluid = fluid
        self.case = case # ['rigid', 'transmission']
        self.h = h
        porous['N'] = porous['E']/(2*(1+porous['nu']))*(1-1j*porous['eta'])
        porous['Kb'] = 2*porous['N']*(1+porous['nu'])/(3*(1-2*porous['nu']))# (porous['E']*porous['nu'])/((1+porous['nu'])*(1-2*porous['nu']))*(1-1j*porous['eta'])
        self.N = porous['N']
        self.phi = porous['phi']
        self.rho0 = fluid['rho'] 
        self.c0 = fluid['cf']
        self.use_mediapack = mediapack 
        self.frequencies = frequencies  

        if self.use_mediapack:
            sys.path.append('../..')
            from mediapack import pem
            self.pem = pem.PEM()
            self.pem.phi = porous['phi']
            self.pem.sigma = porous['sigma']
            self.pem.alpha = porous['alpha']
            self.pem.Lambda_prime = porous['Lambda_prime']
            self.pem.Lambda = porous['Lambda']
            self.pem.tort = porous['tort']
            self.pem.rho_1 = porous['rho_1']
            self.pem.nu = porous['nu']
            self.pem.E = porous['E']
            self.pem.eta = porous['eta']
            self.pem.loss_type = porous['loss_type']
            self.pem._compute_missing()

    def update_frequency(self, om):
        if self.use_mediapack: 
            self.pem.update_frequency(om)
            self.data = {
                'Kf' : self.pem.K_eq_til,
                'rho_eq' : self.pem.rho_eq_til,
                'mu1' : self.pem.mu_1,
                'mu2' : self.pem.mu_2,
                'mu3' : self.pem.mu_3,
                'k1' : sqrt(self.pem.delta_1),
                'k2' : sqrt(self.pem.delta_2),
                'k3' : sqrt(self.pem.delta_3),
                'P' : self.pem.P_til,
                'Q' : self.pem.Q_til,
                'R' : self.pem.R_til,
                'rho11t' : self.pem.rho_11_til,
                'rho12t' : self.pem.rho_12_til,
                'rho22t' : self.pem.rho_22_til,
            }

            # biot_params = list(np.conj(np.array(list(data.values()))))
            self.biot_params = list(self.data.values())

        else: 
            self.biot_params = biot(om=om, porous_material=self.porous, saturating_fluid=self.fluid, 
                            fullBiotcoefficients=False, conv='-')
        self.params = (om, self.phi, self.rho0, self.c0, *self.biot_params[2:11], self.N, self.h)

    def get_matrix(self, kx):
        if self.case == 'rigid':
            matrix = poroelastique_fond_rigide(k1=kx, params=self.params, d=False)
        if self.case == 'transmission':
            matrix = poroelastique_transmission(k1=kx, params=self.params, d=False)
        return matrix

    def get_wavenb(self):
        if self.use_mediapack == False:
            return self.biot_params[5:8]
        else:
            return self.data['k1'], self.data['k2'], self.data['k3']


def poroelastique_fond_rigide(k1, params, d=True, conj=False):
    """
    Rigidly-backed poroelastic layer with fluid loading on top.
    Adapted directly from Weisser et al. 2016, Appendix A
    """
    om, phi, rho0, c0, muP, muA, muS, kP, kA, kS, P, Q, R, N, H = params

    M = np.zeros((7, 7), dtype=np.complex128)
    
    k0 = om/c0
    k20 = sqrt(k0**2 - k1**2)
    k2P = sqrt(kP**2 - k1**2)         
    k2A = sqrt(kA**2 - k1**2)     
    k2S = sqrt(kS**2 - k1**2)

    M[0, 0] = 2*k1*k2P*np.exp(-1j*k2P*H)
    M[0, 1] = -2*k1*k2P*np.exp(1j*k2P*H)
    M[0, 2] = 2*k1*k2A*np.exp(-1j*k2A*H)
    M[0, 3] = -2*k1*k2A*np.exp(1j*k2A*H)
    M[0, 4] = (k1**2 - k2S**2)*np.exp(-1j*k2S*H)
    M[0, 5] = (k1**2 - k2S**2)*np.exp(1j*k2S*H)

    M[1, 0] = (-(P - 2*N + Q*(1 + muP) + muP*R)*kP**2 - 2*N*k2P**2)*np.exp(-1j*k2P*H)
    M[1, 1] = (-(P - 2*N + Q*(1 + muP) + muP*R)*kP**2 - 2*N*k2P**2)*np.exp(1j*k2P*H)
    M[1, 2] = (-(P - 2*N + Q*(1 + muA) + muA*R)*kA**2 - 2*N*k2A**2)*np.exp(-1j*k2A*H)
    M[1, 3] = (-(P - 2*N + Q*(1 + muA) + muA*R)*kA**2 - 2*N*k2A**2)*np.exp(1j*k2A*H)
    M[1, 4] = -2*N*k1*k2S*np.exp(-1j*k2S*H)
    M[1, 5] = 2*N*k1*k2S*np.exp(1j*k2S*H)
    M[1, 6] = 1

    M[2, 0] = -(Q + muP*R)*kP**2*np.exp(-1j*k2P*H)
    M[2, 1] = -(Q + muP*R)*kP**2*np.exp(1j*k2P*H)
    M[2, 2] = -(Q + muA*R)*kA**2*np.exp(-1j*k2A*H)
    M[2, 3] = -(Q + muA*R)*kA**2*np.exp(1j*k2A*H)
    M[2, 6] = phi

    M[3, 0] = -1j*k2P*(1 - phi + muP*phi)*np.exp(-1j*k2P*H)
    M[3, 1] =  1j*k2P*(1 - phi + muP*phi)*np.exp(1j*k2P*H)
    M[3, 2] = -1j*k2A*(1 - phi + muA*phi)*np.exp(-1j*k2A*H)
    M[3, 3] =  1j*k2A*(1 - phi + muA*phi)*np.exp(1j*k2A*H)
    M[3, 4] = -1j*k1*(1 - phi + muS*phi)*np.exp(-1j*k2S*H)
    M[3, 5] = -1j*k1*(1 - phi + muS*phi)*np.exp(1j*k2S*H)
    M[3, 6] = -1j*k20/(rho0*om**2)

    M[4, 0:4] = 1j*k1
    M[4, 4] = -1j*k2S
    M[4, 5] =  1j*k2S

    M[5, 0] = -1j*k2P
    M[5, 1] =  1j*k2P
    M[5, 2] = -1j*k2A
    M[5, 3] =  1j*k2A
    M[5, 4] = -1j*k1 
    M[5, 5] = -1j*k1 

    M[6, 0] = -1j*muP*k2P
    M[6, 1] =  1j*muP*k2P
    M[6, 2] = -1j*muA*k2A
    M[6, 3] =  1j*muA*k2A
    M[6, 4] = -1j*muS*k1
    M[6, 5] = -1j*muS*k1

    if conj: M = M.conj()
    if d: return det(M) #/k0/k2Ps
    else: 
        return M

def poroelastique_transmission(k1, params, d=True):
    """
    Adapted from the rigid bottom version described in Weisser et al. 2016, Appendix A. 
    The 4 BC for x2 = H are duplicated in the case x2 = 0, where the exp term is exp(-1j*k2i*0)
    """
    om, phi, rho0, c0, muP, muA, muS, kP, kA, kS, P, Q, R, N, H = params

    M = np.zeros((8, 8), dtype=complex)
    
    k0 = om/c0
    k20 = sqrt(k0**2 - k1**2)
    k2P = -sqrt(kP**2 - k1**2)         
    k2A = -sqrt(kA**2 - k1**2)     
    k2S = -sqrt(kS**2 - k1**2)

    M[0, 0] = 2*k1*k2P*np.exp(-1j*k2P*H)
    M[0, 1] = -2*k1*k2P*np.exp(1j*k2P*H)
    M[0, 2] = 2*k1*k2A*np.exp(-1j*k2A*H)
    M[0, 3] = -2*k1*k2A*np.exp(1j*k2A*H)
    M[0, 4] = (k1**2 - k2S**2)*np.exp(-1j*k2S*H)
    M[0, 5] = (k1**2 - k2S**2)*np.exp(1j*k2S*H)

    M[1, 0] = (-(P - 2*N + Q*(1 + muP) + muP*R)*kP**2 - 2*N*k2P**2)*np.exp(-1j*k2P*H)
    M[1, 1] = (-(P - 2*N + Q*(1 + muP) + muP*R)*kP**2 - 2*N*k2P**2)*np.exp(1j*k2P*H)
    M[1, 2] = (-(P - 2*N + Q*(1 + muA) + muA*R)*kA**2 - 2*N*k2A**2)*np.exp(-1j*k2A*H)
    M[1, 3] = (-(P - 2*N + Q*(1 + muA) + muA*R)*kA**2 - 2*N*k2A**2)*np.exp(1j*k2A*H)
    M[1, 4] = -2*N*k1*k2S*np.exp(-1j*k2S*H)
    M[1, 5] = 2*N*k1*k2S*np.exp(1j*k2S*H)
    M[1, 6] = 1

    M[2, 0] = -(Q + muP*R)*kP**2*np.exp(-1j*k2P*H)
    M[2, 1] = -(Q + muP*R)*kP**2*np.exp(1j*k2P*H)
    M[2, 2] = -(Q + muA*R)*kA**2*np.exp(-1j*k2A*H)
    M[2, 3] = -(Q + muA*R)*kA**2*np.exp(1j*k2A*H)
    M[2, 6] = phi

    M[3, 0] = -1j*k2P*(1 - phi + muP*phi)*np.exp(-1j*k2P*H)
    M[3, 1] =  1j*k2P*(1 - phi + muP*phi)*np.exp(1j*k2P*H)
    M[3, 2] = -1j*k2A*(1 - phi + muA*phi)*np.exp(-1j*k2A*H)
    M[3, 3] =  1j*k2A*(1 - phi + muA*phi)*np.exp(1j*k2A*H)
    M[3, 4] = -1j*k1*(1 - phi + muS*phi)*np.exp(-1j*k2S*H)
    M[3, 5] = -1j*k1*(1 - phi + muS*phi)*np.exp(1j*k2S*H)
    M[3, 6] = -1j*k20/(rho0*om**2)

    M[4, 0] = 2*k1*k2P
    M[4, 1] = -2*k1*k2P
    M[4, 2] = 2*k1*k2A
    M[4, 3] = -2*k1*k2A
    M[4, 4] = (k1**2 - k2S**2)
    M[4, 5] = (k1**2 - k2S**2)

    M[5, 0] = (-(P - 2*N + Q*(1 + muP) + muP*R)*kP**2 - 2*N*k2P**2)
    M[5, 1] = (-(P - 2*N + Q*(1 + muP) + muP*R)*kP**2 - 2*N*k2P**2)
    M[5, 2] = (-(P - 2*N + Q*(1 + muA) + muA*R)*kA**2 - 2*N*k2A**2)
    M[5, 3] = (-(P - 2*N + Q*(1 + muA) + muA*R)*kA**2 - 2*N*k2A**2)
    M[5, 4] = -2*N*k1*k2S
    M[5, 5] = 2*N*k1*k2S
    M[5, 7] = 1

    M[6, 0] = -(Q + muP*R)*kP**2
    M[6, 1] = -(Q + muP*R)*kP**2
    M[6, 2] = -(Q + muA*R)*kA**2
    M[6, 3] = -(Q + muA*R)*kA**2
    M[6, 7] = phi

    M[7, 0] = -1j*k2P*(1 - phi + muP*phi)
    M[7, 1] =  1j*k2P*(1 - phi + muP*phi)
    M[7, 2] = -1j*k2A*(1 - phi + muA*phi)
    M[7, 3] =  1j*k2A*(1 - phi + muA*phi)
    M[7, 4] = -1j*k1*(1 - phi + muS*phi)
    M[7, 5] = -1j*k1*(1 - phi + muS*phi)
    M[7, 7] = 1j*k20/(rho0*om**2)

    if d: return det(M)#/k0 # /k2P
    else: return M

def poroelastique_rigid_rigid(k1, params, d=True):
    om, phi, rho0, c0, muP, muA, muS, kP, kA, kS, P, Q, R, N, H = params

    M = np.zeros((6, 6), dtype=np.complex128)
    
    k0 = om/c0
    k20 = sqrt(k0**2 - k1**2)
    k2P = sqrt(kP**2 - k1**2)         
    k2A = sqrt(kA**2 - k1**2)     
    k2S = sqrt(kS**2 - k1**2)

    M[0, 0] = 1j*k1*np.exp(-1j*k2P*H)
    M[0, 1] = 1j*k1*np.exp(1j*k2P*H)
    M[0, 2] = 1j*k1*np.exp(-1j*k2A*H)
    M[0, 3] = 1j*k1*np.exp(1j*k2A*H)
    M[0, 4] = -1j*k2S*np.exp(-1j*k2S*H)
    M[0, 5] =  1j*k2S*np.exp(1j*k2S*H)

    M[1, 0] = -1j*k2P*np.exp(-1j*k2P*H)
    M[1, 1] =  1j*k2P*np.exp(1j*k2P*H)
    M[1, 2] = -1j*k2A*np.exp(-1j*k2A*H)
    M[1, 3] =  1j*k2A*np.exp(1j*k2A*H)
    M[1, 4] = -1j*k1 *np.exp(-1j*k2S*H)
    M[1, 5] = -1j*k1 *np.exp(1j*k2S*H)

    M[2, 0] = -1j*muP*k2P*np.exp(-1j*k2P*H)
    M[2, 1] =  1j*muP*k2P*np.exp(1j*k2P*H)
    M[2, 2] = -1j*muA*k2A*np.exp(-1j*k2A*H)
    M[2, 3] =  1j*muA*k2A*np.exp(1j*k2A*H)
    M[2, 4] = -1j*muS*k1 *np.exp(-1j*k2S*H)
    M[2, 5] = -1j*muS*k1 *np.exp(1j*k2S*H)


    M[3, 0:4] = 1j*k1
    M[3, 4] = -1j*k2S
    M[3, 5] =  1j*k2S

    M[4, 0] = -1j*k2P
    M[4, 1] =  1j*k2P
    M[4, 2] = -1j*k2A
    M[4, 3] =  1j*k2A
    M[4, 4] = -1j*k1 
    M[4, 5] = -1j*k1 

    M[5, 0] = -1j*muP*k2P
    M[5, 1] =  1j*muP*k2P
    M[5, 2] = -1j*muA*k2A
    M[5, 3] =  1j*muA*k2A
    M[5, 4] = -1j*muS*k1
    M[5, 5] = -1j*muS*k1

    if d: return det(M) #/k0/k2Ps
    else: 
        return M