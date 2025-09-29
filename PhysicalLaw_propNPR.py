import numpy as np

import torch


from torch.autograd import Variable

import ML_modulus_propNPR
import NumericalMthod
from scipy import stats

def initial_dislocation_density(tau0,alp,mu,b,gs,bet):
    """
    Parameters:
    tau0 (float):Resolved yield shear stress:Pa
    alp (float): Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).
    bet (float): Coefficient beta in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)). 
    mu (float): Shear modulus:Pa
    b (float): Burger's vector: m
    gs (float): grain size: m

    Returns:
    float: dislocation density 
    """
    rho0 = []
    if isinstance(tau0,(float, np.float64)):
        tau0 = [tau0]
        bet = [bet]
        alp = [alp]
        gs = [gs]

    for kk in range(0,len(tau0)):
        # equation becomes:  (sqrt(\rho)^2) - tau0/mu/(alpha *b) *sqrt(rho)  + beta / d/(alpha *b) == 0
        A = bet[kk]/(gs[kk])
        B = alp[kk]*(b)
        tau_bar = tau0[kk]/mu
        if (tau_bar/B)**2 - 4*A/B > 0:
            rho0.append(0.25*(tau_bar/B + np.sqrt((tau_bar/B)**2 - 4*A/B))**2)
        else:
        # cannot solve the equation, use the minimum value for the initial dislocation density 
            print("The initial dislocation density cannot be solved, use the minimum value for the initial dislocation density")
            rho0.append(0.25*(tau_bar/B)**2)
        
    return np.array(rho0)

def sigma_elastic(eps, rho_grain,E_modulus):
    """
    calculate elastic stress
    Parameters:
    eps (float): strain level 
    rho_grain (float): initial dislocation density 
    E_modulus (float): Young's modulus 

    Returns:
    float, float : elastic stress, initial dislocation density
    """
    sigma_ee = E_modulus * eps
    density = rho_grain
    return sigma_ee, density

def sigma_plastic(material_type, eps, schimd_factor_grain, gs, alp, bet, strainrate, materials, rho0, trained_model, \
                  transformer_strain, transformer_gs, transformer_strainrate,  scaler_density,scaler_YP,\
                  scaler_ShearModulus):
    """
    calculate plastic stress at one section 
    Parameters:
    material_type: string, can be "Ni", "Al" or "Cu"
    eps(float): strain level;
    schimd_factor_grain: a [n * 12] array, with the schmid factor on all 12 systems
    gs (array ): grain size: m
    alp (float): Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).
    bet (float): Coefficient beta in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)). 
    strainrate: strainrate
    materials: dictionary for this material
    rho0: initial dislocation density 
    trained_model: trained ML model 
    transformer_strain: strain transformer
    transformer_gs: grain size transformer 
    transformer_strainrate: strainrate transformer
    scaler_YP: Yield point transformer 
    scaler_ShearModulus: Shear modulus transformer 
    scaler_LatticeConst:Lattice constant transoformer 
    scaler_PoissonRatio:PR transformer 
    Returns:
    float: plastic stress and predicted density 
    """
    mu = materials["ShearModulus"]*1e9
    b = np.array(materials['LatticeConst'])/np.sqrt(2)*1e-10
   
    SFactor = np.max(schimd_factor_grain, axis=1)
    yieldpoints =density_to_stress(rho0, alp, mu, b, gs, bet) # Yield 
    
    ShearModulus  = mu * np.ones_like(gs)
    LatticeConst  = np.array(materials['LatticeConst'])*1e-10 * np.ones_like(gs)
    PoissonRatio = materials["PoissonRatio"]*  np.ones_like(gs)
    strains = eps*np.ones_like(gs) 
    strainrates = strainrate*np.ones_like(gs)
    df_test = ML_modulus_propNPR.construct_input(gs,strains,strainrates,ShearModulus,yieldpoints)
    #(GS,strains,strainrates,ShearModulus,LatticeConstant,PossionRatio,Yieldpoint):
    df_test = ML_modulus_propNPR.normalizeInput(df=df_test,transformer_strain=transformer_strain, transformer_gs=transformer_gs,\
                             transformer_strainrate=transformer_strainrate, scaler_YP=scaler_YP,\
                             scaler_ShearModulus = scaler_ShearModulus)

    df_test = df_test.astype('float32')

    X = torch.from_numpy(df_test.values)
    x_test_variable = Variable(X)
    pi_den, sigma_den, mean_den = trained_model(x_test_variable)
    
    mean_stress, dev_stress, mean_den, dev_den = distribution_stress(pi_den,sigma_den,mean_den,gs,b,alp,bet,mu, scaler_density)


    sigma_ep = np.array(mean_stress).reshape(len(gs),1) /SFactor.reshape(len(gs),1)
    sigma_ep_upper = np.array(np.array(mean_stress) + 2* np.array( dev_stress)).reshape(len(gs),1) /SFactor.reshape(len(gs),1)
    sigma_ep_lower = np.array(np.array(mean_stress)  - 2*  np.array( dev_stress)).reshape(len(gs),1) /SFactor.reshape(len(gs),1)
    upper_density = np.array(mean_den )+  1* np.array(dev_den)
    lower_density = np.array(mean_den )-  1* np.array(dev_den)
    # 

    # sigma_ep = np.where(sigma_ep > E_modulus * eps_cr.reshape(sigma_ep.shape), sigma_ep, E_modulus * eps_cr.reshape(sigma_ep.shape))
    return sigma_ep,  np.array(mean_den), sigma_ep_upper,upper_density, sigma_ep_lower,lower_density

def solve_strain(prediction_type, material_type, Schimd_factor, gs, alp, bet, strainrate, materials, rho0, trained_model,  \
            transformer_strain, transformer_gs, transformer_strainrate, scaler_density,  scaler_YP,weight,  sigma_z,scaler_ShearModulus
           ):
    """
    calculate strain at a stress level 
    Parameters:
    prediction_type: can be a string "average", "upper" or "lower"
    material_type: string, can be "Ni", "Al" or "Cu"
    sigma_z (float): stress level Pa;
    schimd_factor_grain: a [n * 12] array, with the schmid factor on all 12 systems
    gs (array ): grain size: m
    alp (float): Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).
    bet (float): Coefficient beta in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)). 
    strainrate: strainrate
    materials: dictionary for this material
    rho0: initial dislocation density 
    trained_model: trained ML model 
    transformer_strain: strain transformer
    transformer_gs: grain size transformer 
    transformer_strainrate: strainrate transformer
    weight: weight of grains on one section
    scaler_YP: Yield point transformer 

    Returns:
    float: plastic stress and predicted density 
    """
    mu = materials["ShearModulus"]*1e9
    E_modulus = 2*mu*(1+materials["PoissonRatio"])
    def stress(eps):
        #  strainrate,possion_ratio, LatticeConst,Yieldpoints, StackingFaultE, trained_model,  AtomicVolume, transformer_strain, transformer_gs, transformer_strainrate, scaler_LC, scaler_PR, scaler_SF, scaler_SM, scaler_V, scaler_YP, scaler_density
        [s_grain_average, ave_density, s_grain_p_upper,upper_density, s_grain_p_lower,lower_density]\
        = sigma_plastic(material_type, eps, Schimd_factor, gs, alp, bet, strainrate, materials, rho0, trained_model, \
                        transformer_strain, transformer_gs, transformer_strainrate,scaler_density,scaler_YP,\
                       scaler_ShearModulus)
        s_grain_e = sigma_elastic(eps * np.ones_like(gs), rho0,E_modulus)[0]
        if prediction_type == "average":
                s_grain_p = s_grain_average 
        if prediction_type == "upper":
                s_grain_p = s_grain_p_upper
        if prediction_type == "lower":
                s_grain_p = s_grain_p_lower 
               
        s_grain_p = s_grain_p
        s_grain =  np.where((s_grain_e.reshape(np.shape(s_grain_e)) > s_grain_p.reshape(np.shape(s_grain_e))), s_grain_p.reshape(np.shape(s_grain_e)), s_grain_e.reshape(np.shape(s_grain_e)))
        # eps_p = eps - s_grain/np.max(Schimd_factor, axis=1).reshape(rho0.shape)/E_modulus
        s0 = np.sum(s_grain * weight) 
        return s0
    
    def stress_equation(eps):
        s0 = stress(eps)
        return s0-sigma_z

   
   
    eps_sss, found = NumericalMthod.bisect_root(stress,0.000, 1, 5000,sigma_z)
    if not found:
        eps_sss, found = NumericalMthod.bisect_root(stress,0.000, 0.5, 5000,sigma_z)
        if not found :
            eps_sss, found = NumericalMthod.bisect_root(stress,0.000, 0.15, 5000,sigma_z)        
            if not found :
                        eps_sss, found = NumericalMthod.bisect_root(stress,0.000, 0.1, 5000,sigma_z)
   
    
    return eps_sss

    

def density_to_stress(dislocation_den,alp,mu,b,gs,bet):
    """
    convert density to strain 
    Parameters:
    dislocation_den (float): dislocation density:1/m^2
    alp (float): Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).
    bet (float): Coefficient beta in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)). 
    mu (float): Shear modulus:Pa
    b (float): Burger's vector: m
    gs (float): grain size: m

    Returns:
    float: resolved shear stress(Pa)
    """
    tau = mu *(alp.reshape(alp.shape)*b*np.sqrt(dislocation_den).reshape(alp.shape) + bet.reshape(alp.shape)/gs.reshape(alp.shape)/np.sqrt(dislocation_den).reshape(alp.shape))
    return tau



def sample_from_mdn(means, std_devs, mix_coeffs, num_samples=1):
    # Choose components for each sample
    components = np.random.choice(len(mix_coeffs), size=num_samples, p=mix_coeffs)

    # Generate samples for each component
    samples = np.random.normal(means[components], std_devs[components])

    return samples
        
def distribution_stress(pi_variable,sigma_variable,mu_variable,gs_list,b_v,alp_list,bet_list,shear_modulus,scaler_density):# unit is shear modulus
    '''
    Distribution of  stress 
    # Parameters for the distribution of A and the constants C1, C2
    pi_variable,sigma_variable,mu_variable: model output
    gs_list: grain size list (m)
    b_v: Burgers vector list
    alp_list:coeff for equation between density and stress
    bet_list: coeff for equation between density and stress
    shear_modulus: Shear modulus 
    scaler_density: change normalized density to density
    '''  
    pi_data = pi_variable.data.numpy()
    sigma_data = sigma_variable.data.numpy()
    mu_data = mu_variable.data.numpy()
    
    # calculate the mean and variance for dislocation density with inverse_transform 
    # However, for density, we firstly re-scale to [ln\rho_min,ln_\rho_max]. It means \bar\rho = (\ln\rho - S0)/L so we can easily know, if the variance is sigma and the mean is mu for non-dimensional output
    # the dimensional output for one bound is upper_bar =  L(mu + sigma) + S0
    # the dimensional output for mean is mu_list =  L(mu) + S0,
    # therefore, L * sigma = upper_bar -  mu_list， which is the variance after re-scaling

    mean_stress = []
    dev_stress = []
    mean_den = []
    dev_den = []
    
    for n in range(len(gs_list)):
        
        alp = alp_list[n]
        bet = bet_list[n]
        gs = gs_list[n]
        n_samples = 1000  # Number of samples
        
        # for k in range(len(pi_data[0])): # how  many different distribution
        mu = mu_data[n]
        sigma = sigma_data[n]
        pi = pi_data[n]
        
        normalized_logrho = sample_from_mdn(mu, sigma, pi, n_samples)
        logden_samples = scaler_density.inverse_transform(normalized_logrho.reshape(-1,1))
        den_samples = np.exp(logden_samples)
        # get statistical value fpr stress
        stress_samples =density_to_stress(den_samples,alp*np.ones_like(den_samples),shear_modulus*np.ones_like(den_samples),b_v*np.ones_like(den_samples),
        gs*np.ones_like(den_samples),bet*np.ones_like(den_samples))
        
        mean_stress.append(np.mean(stress_samples))
        dev_stress.append(np.std(stress_samples))
        
    return mean_stress, dev_stress, mean_den, dev_den

