from numpy.lib.scimath import sqrt
from numpy import conj, pi

"""
This script is a copy of my phd_lib.porous module, to avoid dependency issues.
It contains functions to compute Biot poroelastic parameters and the JCA equivalent fluid parameters
Original is at: https://github.com/marecmat/phd_lib/blob/main/porous.py
"""

def JCA(om, porous_material, saturating_fluid, return_F=False, return_G=False, conv='+'):

    p = porous_material;        f = saturating_fluid

    nu = f['eta']/f['rho'];                 q0 = f['eta']/p['sigma']
    nu_p = f['eta']/(f['rho']*f['Pr']);     q0_p = 1/8*p['phi']*p['ldp']**2

    F = (1 - 1j*om/nu*(2*p['tort']*q0/(p['phi']*p['ld']))**2)**(1/2)
    G = (1 - 1j*om/nu_p*(p['ldp']/4)**2)**(1/2)

    rho_eq = f['rho']/p['phi']*(p['tort'] - F*(nu*p['phi'])/(1j*om*q0))
    K_eq = f['gamma']*f['p0']/p['phi']/(f['gamma'] - (f['gamma'] - 1)/(1 - G*(nu_p*p['phi'])/(1j*om*q0_p)))
    
    if conv == '+': 
        F = conj(F)
        G = conj(G)
        K_eq = conj(K_eq)
        rho_eq = conj(rho_eq)
    out = [K_eq, rho_eq]

    if return_F: out.append(F)
    if return_G: out.append(G)
    
    return out


def  biot(om, porous_material, saturating_fluid, 
          fullBiotcoefficients=True, dict_out=False, 
          u_p_formulation=False, conv='-'):
    
    """
    Computes the Biot coefficient, equivalent densities and wavenumbers for poroelastic materials

    Parameters
    ----------
    om: float or array
        The frequency(ies) at which the biotStuff will be computed
    porous_material: dict
        Porous material properties
    saturating_fluid: dict
        Saturating fluid properties
    fullBiotcoefficients: bool
        Whether to use the complete expressions for Biot coefficients 
        or the rigid approximations
        
    Returns
    -------
        
    parameters: tuple or tuple of lists
        The parameters calculated. The order is:
        Kf, rho_eq, mu1, mu2, mu3, k1, k2, k3, P, Q, R, rho11t, rho12t, rho22t
    """
    p = porous_material;     f = saturating_fluid


    Kf, rho_eq  = JCA(om, p, f, conv=conv)

    try:
        loss_factor = 1+1j*p['eta'] if conv == '+' else 1-1j*p['eta']
        p['N'] = p['E']/(2*(1+p['nu']))*loss_factor
        p['Kb'] = 2*p['N']*(1+p['nu'])/(3*(1-2*p['nu']))
    except KeyError:
        pass


    if fullBiotcoefficients:

        P = ((1 - p['phi'])*(1 - p['phi'] - p['Kb']/p['Ks'])*p['Ks'] + p['phi']*(p['Ks']/Kf)*p['Kb'])/ \
                            (1 - p['phi'] - p['Kb']/p['Ks'] + p['phi']*p['Ks']/Kf) + 4/3*p['N']

        Q = (1 - p['phi'] - p['Kb']/p['Ks'])*p['phi']*p['Ks']/ \
                            (1 - p['phi'] - p['Kb']/p['Ks'] + p['phi']*p['Ks']/Kf)

        R = p['phi']**2*p['Ks']/ \
                            (1 - p['phi'] - p['Kb']/p['Ks'] + p['phi']*p['Ks']/Kf)

    else: 
        # P = p['Kb'] + 4/3*p['N'] + (1 - p['phi']**2)/p['phi']*Kf
        # # P = 2*p['N'] + p['Kb'] + (1 - p['phi'])**2*Kf
        # Q = Kf*(1 - p['phi'])
        # R = p['phi']*Kf
        
        # # # mediapack expressions
        A_hat = p['Kb'] - 2/3*p['N']
        P_hat = A_hat + 2*p['N']
        R = Kf*p['phi']**2
        Q = ((1-p['phi'])/p['phi'])*R
        P = P_hat+Q**2/R

    # Equivalent densities
    rho11t = p['rho1'] - p['phi']*f['rho'] + p['phi']**2*rho_eq
    rho12t = p['phi']*f['rho'] - p['phi']**2*rho_eq
    rho22t = p['phi']**2*rho_eq

    Delta = (P*rho22t + R*rho11t - 2*Q*rho12t)**2 - 4*(P*R - Q**2)*(rho11t*rho22t - rho12t**2)
    delta1_sq = om**2/(2*(P*R - Q**2))*(P*rho22t + R*rho11t - 2*Q*rho12t - sqrt(Delta))
    delta2_sq = om**2/(2*(P*R - Q**2))*(P*rho22t + R*rho11t - 2*Q*rho12t + sqrt(Delta))
    delta3_sq = om**2/p['N']*(rho11t*rho22t - rho12t**2)/rho22t

    mu1 = (P*delta1_sq - om**2*rho11t)/(om**2*rho12t - Q*delta1_sq)
    mu2 = (P*delta2_sq - om**2*rho11t)/(om**2*rho12t - Q*delta2_sq)
    # mu3 = (p['N']*delta3_sq - om**2*rho11t)/(om**2*rho22t)
    mu3 = -rho12t/rho22t

    k1, k2, k3 = sqrt(delta1_sq), sqrt(delta2_sq), sqrt(delta3_sq)

    if dict_out:
        out_dict = {
            'Kf':Kf, 'rho_eq':rho_eq, 'mu1':mu1, 'mu2':mu2, 'mu3':mu3, 'k1':k1, 'k2':k2, 'k3':k3, 
            'P':P, 'Q':Q, 'R':R, 'rho11t':rho11t, 'rho12t':rho12t, 'rho22t':rho22t
        }
        if u_p_formulation: 
            out_dict.update({
                'gamma_t' : p['phi']*(rho12t/rho22t - Q/R),
                'rho_t'   : rho11t - rho12t**2/rho22t, 
                'p_hat'   : P - Q**2/R,
                'a_hat'   : P - Q**2/R - 2*p['N']
            })
        return out_dict
    
    else:
        out_list = [Kf, rho_eq, mu1, mu2, mu3, k1, k2, k3, P, Q, R, rho11t, rho12t, rho22t]
    #              0        1   2    3    4    5   6   7  8  9  10    11       12      13
        if u_p_formulation: out_list.extend([
            p['phi']*(rho12t/rho22t - Q/R), rho11t - rho12t**2/rho22t, 
            P - Q**2/R, P - Q**2/R - 2*p['N']])
    
        return out_list


def compute_biot_coefs_mediapack(biot_pem, freq, porous):
    porous['N'] = porous['N']*(1+1j*porous['eta'])# porous['E']/(2*(1+porous['nu']))

    porous['Kb'] = 2*porous['N']*(1+porous['nu'])/(3*(1-2*porous['nu']))
    biot_pem.phi = porous['phi']
    biot_pem.sigma = porous['sigma']
    biot_pem.alpha = porous['alpha']
    biot_pem.Lambda_prime = porous['Lambda_prime']
    biot_pem.Lambda = porous['Lambda']
    biot_pem.tort = porous['tort']
    biot_pem.rho_1 = porous['rho_1']
    biot_pem.nu = porous['nu']
    biot_pem.E = porous['E']
    biot_pem.eta = porous['eta']
    biot_pem.loss_type = porous['loss_type']
    data = {
        'Kf': [], 'rho_eq': [], 'mu1': [],'mu2': [],'mu3': [],'k1': [],'k2': [],'k3': [], 
        'P':[], 'Q':[], 'R':[], 'rho11t':[], 'rho12t': [], 'rho22t': []}
    for ii, f in enumerate(freq):
        biot_pem._compute_missing()
        biot_pem.update_frequency(2*pi*f)
        data['Kf'].append(biot_pem.K_eq_til)
        data['rho_eq'].append(biot_pem.rho_eq_til)
        data['mu1'].append(biot_pem.mu_2)
        data['mu2'].append(biot_pem.mu_2)
        data['mu3'].append(biot_pem.mu_3)
        data['k1'].append(biot_pem.delta_2)
        data['k2'].append(biot_pem.delta_1)
        data['k3'].append(biot_pem.delta_3)
        data['P'].append(biot_pem.P_til)
        data['Q'].append(biot_pem.Q_til)
        data['R'].append(biot_pem.R_til)
        data['rho11t'].append(biot_pem.rho_11_til)
        data['rho12t'].append(biot_pem.rho_12_til)
        data['rho22t'].append(biot_pem.rho_22_til)
    
    if len(freq) == 1 or freq.shape[0] == 1:
        for i in data.keys():
            data[i] = data[i][0]

    return data
