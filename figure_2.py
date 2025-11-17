# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:47:03 2024

@author: luoji
"""




from spyder_kernels.utils.iofuncs import load_dictionary
import csv
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import random
import math


#%% pre training process 
randseed = 1995
np.random.seed(randseed)
random.seed(randseed)

AveSchmidFactor = 2.2
oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalization factor for Gaussians
test_size = 0.33




#%% Physics 
def initial_dislocation_density(tau0,alp,mu,b,gs,bet):    
    '''
    Parameters
    ----------
    tau0 (float):Resolved yield shear stress:Pa
    alp (float): Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).
    bet (float): Coefficient beta in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)). 
    mu (float): Shear modulus:Pa
    b (float): Burger's vector: m
    gs (float): grain size: m

    Returns
    -------
    rho0 intial dislocation density 
        array of float 
    found : bool
        if we found the solution
    '''
    rho0 = []
    found = []
    for kk in range(0,len(tau0)):
        C0 = bet/(gs[kk])
        C2 = alp*(b[kk])
        C1 = - tau0[kk]/mu[kk]
        rho0_kk = np.roots([C2,C1,C0])# find roots for equation between density and stress 
        if np.isreal(rho0_kk[0]):           
            rho0.append(max(rho0_kk)**2)
            found.append(1)
        else:
            temp =- 1/(2*C2)*(C1)
            rho0.append(temp**2)
            found.append(0)      
    return np.array(rho0),found


def dislocation_density(tau_list,alp,mu,b,tau0,gs,bet,material):
    '''
    Convert resolved shear stress to density     

    Parameters
    ----------
    tau_list : float array
        (Resolved yield shear stress:Pa
    alp :  float 
        Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).
    mu : float 
        Shear modulus
    b : float 
        Burgers vector : m
    tau0 : float array 
        initial dislocation density 1/m^2
    gs : float 
        grain size  grain size 
    bet : float 
        Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).

    Returns
    -------
    float array 
        list of density 

    '''
    #304L ：73MPa* 0.3 = 20
    #316: 80MPa * 0.3 = 24
    rho0, found =  initial_dislocation_density(tau0,alp,mu,b,gs,bet)
    tau_cr = alp*mu*b*np.sqrt(rho0) + bet/(gs)/np.sqrt(rho0)*mu # minimum stress for yielding 
    tau_cr = tau_cr - np.array(material == '304L')*20e6 - np.array(material == '304')*20e6  -  (np.array(material == '316L'))*24e6 -(np.array(material == '316'))*24e6
    # if material == '304L'or material == '304':
    #     tau_cr = tau_cr - 20e6
    # if material == '316L'or material == '316':
    #     tau_cr = tau_cr - 24e6    
    dislocation_den = []
    for kk in range(0,len(tau_list)):
        if np.array(tau_list)[kk] < tau_cr[kk] :
            dislocation_den.append(rho0[kk]) # if small than rho_cr, it is rho_0
        else:
            A = bet/(gs[kk])
            B = alp*(b[kk])
            tau_x = tau_list[kk]
            if material[kk] == '304L'or material[kk] == '304':
                tau_x = tau_x - 20e6
            if material[kk] == '316L'or material[kk] == '316':
                tau_x = tau_x - 24e6
            tau_bar = tau_x/mu[kk]
            dislocation_den.append(0.25*(tau_bar/B + np.sqrt((tau_bar/B)**2 - 4*A/B))**2) # otherwise it is the solution 
    return np.array(dislocation_den)   



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
    tau = mu *(alp*b*np.sqrt(dislocation_den) + bet/gs/np.sqrt(dislocation_den))
    return tau

def VisualizationData_hist(df,ax):    
    features = df.drop('stresses', axis=1) # get the remaining except stress as features 
    target = df['stresses'] # target is the stress 
    ShearModulus = np.array(features['ShearModuluses'])*1e9  # change the shear modulus unit to be Pa
    strain = np.array(features['strains'])# change unit to m
    Yieldpoint = np.array(features['Yieldpoint'])*1e6 # change normal stress to shear stress 
    # features['Yieldpoint'] = Yieldpoint 
    Tau0 = Yieldpoint/AveSchmidFactor
    grainsize = np.array(features['grainsize'])*1e-6 # change unit to m
    alp = 0.57
    bet = 0.00176 
    material = features['Material']
    
    BurgersVector = np.array(features['LatticeConsts'])/np.sqrt(2) *1e-10# change unit to m
    density_y , found= initial_dislocation_density(Tau0,alp,ShearModulus,BurgersVector,grainsize,bet)
    
    
    # fig, ax = plt.subplots(1,1,figsize=(4,4), facecolor='white')
    style = {'facecolor':'#DBE7C9', 'edgecolor':'#294B29', 'linewidth': 3}
    ax[0].hist(np.log10(density_y),bins=np.arange(12,math.ceil(max(np.log10(density_y))),0.2),**style)
    ax[0].set_xlim([11.9,16])
    # ax.set_xticks([]fontsize =18)
    # ax.set_yticks(fontsize = 18)
    ax[0].set_xlabel(r'log$_{10}$($\rho_0$)')
    ax[0].set_ylabel('Count')
    # plt.legend(fontsize = 25)
    # plt.ylim([0,300])
    ax[0].set_yticks([0,100,200,300])
    ax[0].set_xticks([12,13,14,15])
    # ax[0].tick_params(axis='both', labelsize=18)
    # ax.text(-0.28, 1.0,'(d)',transform=ax.transAxes,fontsize=18,color='black', ha='right', va='top',bbox=dict(facecolor='white', edgecolor='white',boxstyle="round,pad=0.2") )
    # ax.tight_layout()
    # plt.show()
    
   
    

    
    
    print("Finished!")
    # new_datag = np.array(newdataGood)
    # new_datab = np.array(newdataBad)
    # fig, ax = plt.subplots(1,1,figsize=(4,4), facecolor='white')
    style = {'facecolor':'#B9B4C7','edgecolor':'#352F44', 'linewidth': 3}

    ax[1].hist(Yieldpoint/ShearModulus/0.002,bins=20,**style)
    ax[1].tick_params(axis='both')
    # plt.ylim([0,700])
    # ax[1].yticks([0,200,400,600])
    # plt.xticks([0,2,4,6,8])
    ax[1].set_ylabel('Count')
    ax[1].set_xlabel(r'Yield Stress (0.2%$\mu$)')
    # plt.text(-0.28, 1.0,'(e)',transform=ax.transAxes,fontsize=18,color='black', ha='right', va='top',bbox=dict(facecolor='white', edgecolor='white',boxstyle="round,pad=0.2") )
    # fig.tight_layout()
    # plt.show()
    
    # fig, ax = plt.subplots(1,1,figsize=(4,4), facecolor='white')
    style = {'facecolor':'#A5C9CA', 'edgecolor':'#395B64', 'linewidth': 3}
    ax[2].hist(grainsize/1e-6,bins=20,**style)
    ax[2].set_ylabel('Count')
    # ax[2].tick_params(axis='both')

    ax[2].set_xlabel(r'$d_\text{ave}$ (μm)')
    # plt.text(-0.4, 1.0,'(f)',transform=ax.transAxes,fontsize=18,color='black', ha='right', va='top',bbox=dict(facecolor='white', edgecolor='white',boxstyle="round,pad=0.2") )
    # fig.tight_layout()
    # plt.show()
    
    # fig, ax = plt.subplots(1,1,figsize=(6,6), facecolor='white')
    # return hexbin_data
    Materials = features["Material"]
    tau_list = pd.Series(np.array(target)*1e6/AveSchmidFactor)# change normal stress to shear stress
   
    density_train = dislocation_density(tau_list,alp,ShearModulus,BurgersVector,Yieldpoint,grainsize, bet,Materials)
    # fig, ax = plt.subplots(1,1,figsize=(4,4), facecolor='white')
    # plt.ylim([0,400])
    ax[3].set_yticks([0,100,200,300,400])
    ax[3].set_xticks([12,13,14,15,16])
    style = {'facecolor':'#F2A07B', 'edgecolor':'#7D0633', 'linewidth': 3}
    ax[3].hist(np.log10(density_train),bins=20,**style)
    # ax[3].set_ylabel('count',fontsize = 18)
    # plt.tick_params(axis='both', labelsize=18)
    ax[3].set_xlim([12,16])
    ax[3].set_xlabel(r'log$_{10}$($\rho$)')
    ax[3].set_ylabel('Count')
    # plt.text(-0.32, 1.0,'(g)',transform=ax.transAxes,fontsize=18,color='black', ha='right', va='top',bbox=dict(facecolor='white', edgecolor='white',boxstyle="round,pad=0.2") )
    # fig.tight_layout()
    # plt.show()
    
    # colors = {'Cu': 'b',
    #           'Al':'r',
    #           'Ni':'g',
    #           '304L':'purple',
    #           '316L':'purple',
    #           '304':'purple',
    #           '316':'purple',
    #           '316SS':'purple',
    #           }
    # # Markers = ['1','2','3','4','o','v','>','<','^']
    #     # if data["material"] == material:
    # liss_color = [colors[material_single] for material_single in material]  
    # for index in range(len(strain)):      
    #     plt.scatter(strain[index], np.log10(density_train[index]),color=liss_color[index] ,s = 50, alpha = 0.7)
    # plt.xlim([0,0.2])
    # # plt.ylim([14,17])
    # plt.xlabel('Strain')
    # plt.ylabel(r'log$_{10}\rho$')
    # plt.show()
    
    
    # style = {'facecolor':'#86a6df', 'edgecolor':'#324e7b', 'linewidth': 3}
    # fig, ax = fig, ax = plt.subplots(1,1,figsize=(6,6), facecolor='white')
    # plt.xlim([0,0.5])
    # plt.hist(strain,bins=20,**style)
    # plt.tick_params(axis='both', labelsize=28)
    # plt.text(-0.26, 1.0,'(g)',transform=ax.transAxes,fontsize=18,color='black', ha='right', va='top',bbox=dict(facecolor='white', edgecolor='white',boxstyle="round,pad=0.2") )
    # fig.tight_layout()
    # # plt.ylim([0,800])
    # # plt.yticks([0,200,400,600,800])
    # plt.xticks([0,0.1,0.2,0.3,0.4,0.5])
    # plt.xlabel(r'Strain',fontsize = 25)
    # plt.show()
    # return hexbin_data
    # Materials = features["Material"]
    # tau_list = pd.Series(np.array(target)*1e6/AveSchmidFactor)# change normal stress to shear stress
   # tau_list,alp,mu,b,tau0,gs,bet,material
    # density_train = dislocation_density(tau_list,alp,ShearModulus,BurgersVector,Yieldpoint,grainsize, bet,Materials)

#%%
def VisualizationData(df,ax):    
    features = df.drop('stresses', axis=1) # get the remaining except stress as features 
    # target = df['stresses'] # target is the stress 
    ShearModulus = np.array(features['ShearModuluses'])*1e9  # change the shear modulus unit to be Pa
    # grainsize = np.array(features['grainsize'])*1e-6 # change unit to m
    Yieldpoint = np.array(features['Yieldpoint'])*1e6 # change normal stress to shear stress 
    # features['Yieldpoint'] = Yieldpoint 
    Tau0 = Yieldpoint/AveSchmidFactor
    grainsize = np.array(features['grainsize'])*1e-6 # change unit to m
    alp = 0.57
    bet = 0.00176 
    BurgersVector = np.array(features['LatticeConsts'])/np.sqrt(2) *1e-10# change unit to m
    density_y , found= initial_dislocation_density(Tau0,alp,ShearModulus,BurgersVector,grainsize,bet)
    # plt.figure(figsize=(10, 12), facecolor='white')

    hexbin_data = ax.hexbin( grainsize*1e6,Yieldpoint/1e6, gridsize=(40,10), cmap='BuPu')
    # ax.set_xticks(np.arange(0,151,30))
    # ax.set_yticks(np.arange(0,900,200))
    
    # Adding labels and title
    ax.set_xlabel(r'$d_\text{ave}$ (μm)',fontsize = 18)
    ax.set_xlim([0,150])
    # ax.set_ylim([0,800])
    ax.set_ylabel('Yield Stress (MPa)',fontsize = 18)
    # ax.set_title('2D Histogram for grain sizeand yield \n strength in training data(Pure metals)\n', fontsize = 20)
    ax.tick_params(axis='both', labelsize=18)
    # Adding colorbar

    # plt.colorbar()
    # plt.scatter(new_datag[0,:],new_datag[1,:],color='green',s = 240, marker='s',label = "good results for NiCoCr")
    # plt.scatter(new_datab[0,:],new_datab[1,:],color='red',s = 240, marker='s',label = "bad results for NiCoCr")
    # plt.legend(fontsize = 24)
    # Display the plot
    # plt.show()
    return hexbin_data

#%%
def VisualizationData_pure(df,ax):    
    features = df.drop('stresses', axis=1) # get the remaining except stress as features 
    # target = df['stresses'] # target is the stress 
    ShearModulus = np.array(features['ShearModuluses'])*1e9  # change the shear modulus unit to be Pa
    # grainsize = np.array(features['grainsize'])*1e-6 # change unit to m
    Yieldpoint = np.array(features['Yieldpoint'])*1e6 # change normal stress to shear stress 
    # features['Yieldpoint'] = Yieldpoint 
    Tau0 = Yieldpoint/AveSchmidFactor
    grainsize = np.array(features['grainsize'])*1e-6 # change unit to m
    alp = 0.57
    bet = 0.00176 
    BurgersVector = np.array(features['LatticeConsts'])/np.sqrt(2) *1e-10# change unit to m
    density_y , found= initial_dislocation_density(Tau0,alp,ShearModulus,BurgersVector,grainsize,bet)
    # plt.figure(figsize=(10, 12), facecolor='white')

    hexbin_data = ax.hexbin( grainsize*1e6,Yieldpoint/1e6, gridsize=(60,5), cmap='Blues')
    # ax.set_xticks(np.arange(0,151,30))
    # ax.set_yticks(np.arange(0,900,200))
    
    # Adding labels and title
    ax.set_xlabel(r'$d_\text{ave}$ (μm)',fontsize = 18)
    ax.set_xlim([0,150])
    # ax.set_ylim([0,300])
    ax.set_ylabel('Yield stress (MPa)',fontsize = 18)
    # ax.set_title('2D Histogram for grain sizeand yield \n strength in training data(Pure metals)\n', fontsize = 20)
    ax.tick_params(axis='both', labelsize=18)
    # Adding colorbar

    # plt.colorbar()
    # plt.scatter(new_datag[0,:],new_datag[1,:],color='green',s = 240, marker='s',label = "good results for NiCoCr")
    # plt.scatter(new_datab[0,:],new_datab[1,:],color='red',s = 240, marker='s',label = "bad results for NiCoCr")
    # plt.legend(fontsize = 24)
    # Display the plot
    # plt.show()
    return hexbin_data
#%%
def VisualizationData_SS(df,ax):    
    features = df.drop('stresses', axis=1) # get the remaining except stress as features 
    # target = df['stresses'] # target is the stress 
    ShearModulus = np.array(features['ShearModuluses'])*1e9  # change the shear modulus unit to be Pa
    # grainsize = np.array(features['grainsize'])*1e-6 # change unit to m
    Yieldpoint = np.array(features['Yieldpoint'])*1e6 # change normal stress to shear stress 
    # features['Yieldpoint'] = Yieldpoint 
    Tau0 = Yieldpoint/AveSchmidFactor
    grainsize = np.array(features['grainsize'])*1e-6 # change unit to m
    alp = 0.57
    bet = 0.00176 
    BurgersVector = np.array(features['LatticeConsts'])/np.sqrt(2) *1e-10# change unit to m
    density_y , found= initial_dislocation_density(Tau0,alp,ShearModulus,BurgersVector,grainsize,bet)
    # plt.figure(figsize=(10, 12), facecolor='white')

    hexbin_data = ax.hexbin(Yieldpoint/1e6, grainsize*1e6, gridsize=(20,5), cmap='Purples')
    # ax.set_xticks(np.arange(0,151,30))
    # ax.set_yticks(np.arange(0,900,200))
    
    # Adding labels and title
    ax.set_ylabel(r'$d_\text{ave}$ (μm)',fontsize = 18)
    # ax.set_xlim([0,150])
    # ax.set_ylim([0,300])
    ax.set_xlabel('Yield Stress (MPa)',fontsize = 18)
    # ax.set_title('2D Histogram for grain sizeand yield \n strength in training data(Pure metals)\n', fontsize = 20)
    ax.tick_params(axis='both', labelsize=18)
    # Adding colorbar

    # plt.colorbar()
    # plt.scatter(new_datag[0,:],new_datag[1,:],color='green',s = 240, marker='s',label = "good results for NiCoCr")
    # plt.scatter(new_datab[0,:],new_datab[1,:],color='red',s = 240, marker='s',label = "bad results for NiCoCr")
    # plt.legend(fontsize = 24)
    # Display the plot
    # plt.show()
    return hexbin_data
#%%
def scatter_newdata(ax, data,color,marker,label):
    if marker == 'o':
        ax.scatter(data[0,:],data[1,:],color='none',edgecolors = color, linewidth=4, s = 200, marker=marker,label = label,alpha = 0.7)
    else:
        ax.scatter(data[0,:],data[1,:],color = color, linewidth=4, s = 200, marker=marker,label = label,alpha = 0.7)
    # plt.scatter(new_datab[0,:],new_datab[1,:],color='red',s = 240, marker='s',label = "bad results for NiCoCr")
    # plt.legend(fontsize = 24)
#%%data processing 
def loadingdata(dataname):    
    ''' 
    Parameters
    ----------
    dataname : string 
        the path of loading  spydata file .
    ------
    df : datafram 
        the initial dataset without any processing 
    '''
    data = load_dictionary(dataname)
    datas = data[0]["data"]
    # these are raw data from experiments
    filename = "material_properties.csv"
    with open(filename, 'r') as csv_file:    
        reader = csv.DictReader(csv_file)     
        data = list(reader)
    # get material list
    material_list = {row.pop('Material'): {k: float(v) for k, v in row.items()} for row in data}
   #assign all list 
    materials = []
    stresses = []
    strains = []
    GS = []
    samplesizes = []
    strainrates = []
    processingmethods = []
    ShearModuluses = []
    PoissonRatios = []
    LatticeConsts = []
    AtomicVolumes = []
    StackingFaultEs = []
    Yieldpoint = []
    YE_observe = []
    k = 0
    while k<len(datas):        
        sscurve = datas[k]
        # -----------assign the first data point of  ss curve in training set -------------#
        strains.append(float((sscurve["YE_observe"]))) # assign observed yield  strain        
        if sscurve["YS_exp"] == 'nan':                
            stresses.append(float(sscurve["YS_observe"])) # if we cannot find the exeprimental data, use observed one
        else:
            stresses.append(float((sscurve["YS_exp"])))    # otherwise use reported one 
        #material type 
        materials.append(sscurve["material"])
        # grain size m 
        GS.append(sscurve["size"])
        # size sample 3 element array 
        samplesize = sscurve["samplesize"]
        while len(sscurve["samplesize"]) < 3:
            samplesize.append(np.nan) # if not find it, assign nan valyue 
        samplesizes.append(sscurve["samplesize"])
        # processing method
        processingmethods.append(sscurve["Processingmethod"])
        #strain rate
        strainrates.append(float(sscurve["StrainRate"] or 1e-4))
        if sscurve["YS_exp"] == 'nan':                
            Yieldpoint.append(float(sscurve["YS_observe"]))
        else:
            Yieldpoint.append(float((sscurve["YS_exp"])))
        ShearModuluses.append(material_list[sscurve["material"]]['ShearModulus'])
        PoissonRatios.append(material_list[sscurve["material"]]['PoissonRatio'])
        LatticeConsts.append(material_list[sscurve["material"]]['LatticeConst'])
        AtomicVolumes.append(material_list[sscurve["material"]]['AtomicVolume'])
        StackingFaultEs.append(material_list[sscurve["material"]]['StackingFaultE'])
        YE_observe.append(float((sscurve["YE_observe"])))
        # -----------assign the first data point of  ss curve in training set -------------#
       
        
       # ----------- add other data point into training set   -----------#
        for strain, stress in sscurve["sscurve"]:
            strains.append(strain)
            stresses.append(stress)
            materials.append(sscurve["material"])
            GS.append(sscurve["size"])
            samplesize = sscurve["samplesize"]
            while len(sscurve["samplesize"]) < 3:
                samplesize.append(np.nan)
            samplesizes.append(sscurve["samplesize"])
            
            processingmethods.append(sscurve["Processingmethod"])
            strainrates.append(float(sscurve["StrainRate"] or 1e-4))
            if sscurve["YS_exp"] == 'nan':                
                Yieldpoint.append(float(sscurve["YS_observe"]))
            else:
                Yieldpoint.append(float((sscurve["YS_exp"])))
            ShearModuluses.append(material_list[sscurve["material"]]['ShearModulus'])
            PoissonRatios.append(material_list[sscurve["material"]]['PoissonRatio'])
            LatticeConsts.append(material_list[sscurve["material"]]['LatticeConst'])
            AtomicVolumes.append(material_list[sscurve["material"]]['AtomicVolume'])
            StackingFaultEs.append(material_list[sscurve["material"]]['StackingFaultE'])
            YE_observe.append(float((sscurve["YE_observe"])))
        k += 1
            
    # how we deal with nan value in sample size
    samplesizes_np = np.array(samplesizes)
    samplesizes_np[np.isnan(samplesizes_np[:,0]),0] = 2 # mm 
    samplesizes_np[np.isnan(samplesizes_np[:,1]),1] = 2
    samplesizes_np[np.isnan(samplesizes_np[:,2]),2] = 5*samplesizes_np[np.isnan(samplesizes_np[:,2]),1]
    df = pd.DataFrame({
        
        'grainsize': GS,#unit um
        'strains': strains,
        'strainrates':strainrates,
        "Yieldpoint":Yieldpoint,#unit MPa
        'Material':materials,
        'ShearModuluses':ShearModuluses,#unit:GPa
        'PoissonRatios': PoissonRatios,
        'LatticeConsts': LatticeConsts,#unit:A
        'AtomicVolumes': AtomicVolumes,#unit:A^3
        'StackingFaultEs': StackingFaultEs,#unit: mJ
        'stresses':stresses,#unit MPa
        "YE_observe":YE_observe
    }) # assign the initial dataset without any processing 
    return df


def loadingStainlessSteeldata(dataname):    
    ''' 
    Parameters
    ----------
    dataname : string 
        the path of loading stainless steeel spydata file .
    ------
    df : datafram 
        the initial dataset without any processing 
    '''
    data = load_dictionary(dataname)
    datas = data[0]["data"]
    # these are raw data from experiments
    filename = "material_properties.csv"
    with open(filename, 'r') as csv_file:    
        reader = csv.DictReader(csv_file)     
        data = list(reader)
    # get material list
    material_list = {row.pop('Material'): {k: float(v) for k, v in row.items()} for row in data}
   #assign all list 
    material_name = material_list.keys()
    materials = []
    stresses = []
    strains = []
    GS = []
    samplesizes = []
    strainrates = []
    processingmethods = []
    ShearModuluses = []
    PoissonRatios = []
    LatticeConsts = []
    AtomicVolumes = []
    StackingFaultEs = []
    Yieldpoint = []
    YE_observe = []
    k = 0
    while k<len(datas):      
        print(k)
        sscurve = datas[k]
        if sscurve["material"] not in material_name:
            k = k+1
            continue
        ss_type = datas[k]['ss_type']
        strain_raw = sscurve["sscurve"][:,0]
        stress_raw = sscurve["sscurve"][:,1]
        
        if ss_type == 'E' :
            strain = np.log(1+strain_raw)
            stress = stress_raw*(1+strain_raw)
        else:
            strain = strain_raw
            stress = stress_raw
      
        
       # ----------- add other data point into training set   -----------#
        for strain, stress in sscurve["sscurve"]:
            
            strains.append(strain)
            stresses.append(stress)
            materials.append(sscurve["material"])
            GS.append(sscurve["size"])
            samplesize = []
            while len(samplesize) < 3:
                samplesize.append(np.nan)
            samplesizes.append(samplesize)     
            processingmethods.append('N/A')
            strainrates.append(float(sscurve["StrainRate"] or 1e-4))
            # if sscurve["YS_exp"] == 'nan':                
            #     Yieldpoint.append(float(sscurve["YS_observe"]))
            # else:
            Yieldpoint.append(float((sscurve["YS"])))
            ShearModuluses.append(material_list[sscurve["material"]]['ShearModulus'])
            PoissonRatios.append(material_list[sscurve["material"]]['PoissonRatio'])
            LatticeConsts.append(material_list[sscurve["material"]]['LatticeConst'])
            AtomicVolumes.append(material_list[sscurve["material"]]['AtomicVolume'])
            StackingFaultEs.append(material_list[sscurve["material"]]['StackingFaultE'])
            YE_observe.append(0.002)
        k += 1
    
    # how we deal with nan value in sample size
    samplesizes_np = np.array(samplesizes)
    print(samplesizes_np.shape)
    samplesizes_np[np.isnan(samplesizes_np[:,0]),0] = 2 # mm 
    samplesizes_np[np.isnan(samplesizes_np[:,1]),1] = 2
    samplesizes_np[np.isnan(samplesizes_np[:,2]),2] = 5*samplesizes_np[np.isnan(samplesizes_np[:,2]),1]
    df = pd.DataFrame({
        
        'grainsize': GS,#unit um
        'strains': strains,
        'strainrates':strainrates,
        "Yieldpoint":Yieldpoint,#unit MPa
        'Material':materials,
        'ShearModuluses':ShearModuluses,#unit:GPa
        'PoissonRatios': PoissonRatios,
        'LatticeConsts': LatticeConsts,#unit:A
        'AtomicVolumes': AtomicVolumes,#unit:A^3
        'StackingFaultEs': StackingFaultEs,#unit: mJ
        'stresses':stresses,#unit MPa
        "YE_observe":YE_observe
    }) # assign the initial dataset without any processing 
    return df

#%%--------------------Main function -------------------------------

# fig

from spyder_kernels.utils.iofuncs import load_dictionary
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import random
import math
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
# Simulate representative data to enable plotting
# These placeholders represent the values computed by your functions like initial_dislocation_density etc.
np.random.seed(0)
sim_density_y = np.random.lognormal(mean=14, sigma=0.4, size=1000)
sim_yield_stress = np.random.exponential(scale=1.0, size=1000)
sim_grainsize = np.random.exponential(scale=20.0, size=1000)
sim_density_train = np.random.lognormal(mean=14.5, sigma=0.3, size=1000)

# Define histogram styles
style_d = {'facecolor': '#DBE7C9', 'edgecolor': '#294B29', 'linewidth': 3}
style_e = {'facecolor': '#B9B4C7', 'edgecolor': '#352F44', 'linewidth': 3}
style_f = {'facecolor': '#A5C9CA', 'edgecolor': '#395B64', 'linewidth': 3}
style_g = {'facecolor': '#F2A07B', 'edgecolor': '#7D0633', 'linewidth': 3}
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# Reuse subplot layout

fig = plt.figure(figsize=(12,10), facecolor='white')
gs = gridspec.GridSpec(3, 8, width_ratios=[1,1,1,1,1,1,1,1],height_ratios=[1,1,1.2])

# Assign axes
ax_a = fig.add_subplot(gs[0:2, 0:4])
# ax_a_cb =  fig.add_subplot(gs[0:2, 3])
ax_b = fig.add_subplot(gs[0, 4:8])
# ax_b_cb =  fig.add_subplot(gs[0,8])
ax_c = fig.add_subplot(gs[1, 4:8])
# ax_c_cb =  fig.add_subplot(gs[1,8])
ax_d = fig.add_subplot(gs[2, 0:2])
ax_e = fig.add_subplot(gs[2, 2:4])
ax_f = fig.add_subplot(gs[2, 4:6])
ax_g = fig.add_subplot(gs[2, 6:8])
fig.align_ylabels([ax_b, ax_c])


# fig.rcParams['font.size'] = '24'
# path = 'samplesize97_withObserve_withExpyieldpoint3.spydata'
path = 'clean_data3.spydata'
# data = cleandata(path_name)
# path = 'samples_StainlessSteel_under10.spydata'
# data = cleandata(path)
path_StainlessSteel  = 'samples_StainlessSteel_under10.spydata'
data2 = loadingdata(path)
data1 = loadingStainlessSteeldata(path_StainlessSteel)
data = pd.concat([data1,data2],ignore_index=True)
#%%

VisualizationData_hist(data, [ax_d,ax_e,ax_f,ax_g])

# fig, ax = plt.subplots(1,1,figsize=(8,4), facecolor='white')
hexbin_data = VisualizationData_SS(data1,ax_c)
# ax_a.text(-0.2, 1.0,'(c)',transform=ax.transAxes,fontsize=18,color='black', ha='right', va='top',bbox=dict(facecolor='white', edgecolor='white',boxstyle="square,pad=0.8"))
cbar = fig.colorbar(hexbin_data, ax=ax_c, ticks=[0,50,100,150,200], extend='max', label='Count')
ax_c.text(0.95, .9,'Stainless steel',transform=ax_c.transAxes,fontsize=18,color='black', ha='right', va='top',bbox=dict(facecolor='white', edgecolor='black',boxstyle="round,pad=0.2") )
cbar.ax.tick_params(labelsize=22)
cbar.set_label("Count", fontsize=24)
# fig.tight_layout()
#%%
# fig, ax = plt.subplots(1,1,figsize=(8,4), facecolor='white')
hexbin_data = VisualizationData_pure(data2,ax_b)
# ax_b.text(-0.26, 1.0,'(b)',transform=ax.transAxes,fontsize=18,color='black', ha='right', va='top',bbox=dict(facecolor='white', edgecolor='white',boxstyle="square,pad=0.8"))
cbar = fig.colorbar(hexbin_data,ax= ax_b, ticks=[0,50,100,150,200],extend='max',label='Count')
cbar.ax.tick_params(labelsize=22)
cbar.set_label("Count", fontsize=24)
ax_b.text(0.95, .9,'Pure metals',transform=ax_b.transAxes,fontsize=18,color='black', ha='right', va='top',bbox=dict(facecolor='white', edgecolor='black',boxstyle="round,pad=0.2") )
# fig.tight_layout()

#%%

#%%

#%%

new_datagsGood = [5.93,
            2.47,
            34.82,
            14.56,
            11.23,        
            1.31,
            1.48,
            2.03
]
new_dataypGood = [348,
                    510,
                    175,
                    225,
                    286,
                    776,
                    699,
                    575,
                    ]
# new_datagsBad =[
# ]
# new_dataypBad = [
#                 ]
# plt.rcParams['font.size'] = '20'
# fig, ax = plt.subplots(1,1,figsize=(7,7), facecolor='white')
#
# for t in cbar.ax.get_yticklabels():
#      t.set_fontsize(20)
# rs=[10,50]

# plt.rcParams['font.size'] = '20'
# fig, ax = plt.subplots(1,1,figsize=(8,8), facecolor='white')
hexbin_data = VisualizationData(data,ax_a)

# ax.text(0.90, 0.95,"All data", transform=ax.transAxes,
         # fontsize=24, color='black', ha='right', va='top')


# VisualizationData_hist(data)
cbar = fig.colorbar(hexbin_data,ax= ax_a, ticks=[0,50,100,150,200],extend='max',label='Count')
cbar.ax.tick_params(labelsize=22)
cbar.set_label("Count", fontsize=24)
marker = 'o'
color = 'red'
label = "NiCoCr"
scatter_newdata(ax_a,np.array([new_datagsGood,new_dataypGood]),  color, marker, label)



new_dataypGood = [443,
554,
439,
536,
264,
169,
136,
94,
61,
94,
]

new_datagsGood = [2.95,
                2.43,
                3.42,
                3.07,
                6.87,
                16.02,
                21.8,
                46.5,
                70.9,
                46.5,
                ]
marker = 'x'
color = 'blue'
label = "CantorAlloy"
scatter_newdata(ax_a,np.array([new_datagsGood,new_dataypGood]),  color, marker, label)
# plt.text(-0.24, 1.0,'(a)',transform=ax.transAxes,fontsize=18,color='black', ha='right', va='top',bbox=dict(facecolor='white', edgecolor='white',boxstyle="square,pad=0.8"))
# fig.tight_layout()
ax_a.legend(prop={"size":24},loc='upper right')



# Add labels
labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g]
# label_coords = {
#     'a': ( -0.1, 1.05),
#     'b': ( -0.1235, 1.12),
#     'c': ( -0.1235, 1.12),
#     'd': ( -0.2, 0.9),
#     'e': ( -0.2, 0.9),
#     'f': ( -0.2, 0.9),
#     'g': ( -0.2, 0.9),
# }
for ax, label in zip(axes, labels):
    # Make each subplot tight within its own space
    ax.set_anchor('C')  # Center content in subplot box
    ax.tick_params(direction='in', length=6, width=1.2, labelsize=18)
    ax.set_xlabel(ax.get_xlabel(), fontsize=24, labelpad=2)
    ax.set_ylabel(ax.get_ylabel(), fontsize=24, labelpad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    # ax.set_xticks([0, 2, 4, 6, 8])
    # ax.set_yticks([0, 200, 400, 600])
    ax.tick_params(axis='both', labelsize=22)
    # ax.legend(prop={"size":24})

    # key = label.strip('()')
    # x, y = label_coords.get(key, (-0.1, 1.1))
    # ax.text(x, y, f'({key})',
    #     transform=ax.transAxes,
    #     fontsize=24, va='top', ha='right',
    #     bbox=dict(facecolor='white', edgecolor='white', boxstyle="square,pad=0.3"))

fig_labels = {
    '(a)': (0.03, 1),
    '(b)': (0.53, 1),
    '(c)': (0.53, 0.7),
    '(d)': (0.03, 0.36),
    '(e)': (0.295, 0.36),
    '(f)': (0.525, 0.36),
    '(g)': (0.785, 0.36),
}

for label, (x, y) in fig_labels.items():
    fig.text(x, y, label, fontsize=26, fontweight='normal', ha='right', va='top')
    
plt.subplots_adjust(wspace=0.5, hspace=2)
plt.tight_layout()
plt.savefig('Fig0 data for training.png')
