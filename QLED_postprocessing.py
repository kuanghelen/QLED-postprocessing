"""
Created on Mon Aug 1, 2022

@author: Helen Kuang

Base code by Gillian Shen
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from IPython.display import Image
import matplotlib as mpl
from pylab import cm
import time 
from datetime import date

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import streamlit as st


st.set_page_config(page_title='QLED')

def intro():
    st.title('QLED Testing Postprocessing')
    st.subheader('The Ginger Lab, University of Washington')
    st.caption('Gillian Shen')

#     with st.expander('Quick Guide'):
#         st.write("Insert information")
    
    st.markdown("___")
            

def pre(spectra_input, IV_photo_input, photo_data_input):
    today = date.today()
    date_string = date.isoformat(today)
    
    # TODO: what is the date for?
#     st.write(date_string)
    global Spectra
    global IV_EL
    global phototopic
    global numpoints
    global Sample_Name
    
#     r'%s' % variable
    
    Spectra = pd.read_csv(spectra_input, sep='\t',skipfooter=1)
#     st.write(Spectra)
    Spectra = Spectra.to_numpy()
#     st.write(Spectra)

    IV_EL = pd.read_csv(IV_photo_input, sep='\t')
#     st.write(IV_EL)
    IV_EL = IV_EL.to_numpy()
    
    #Phototopic curve
    phototopic = pd.read_csv(photo_data_input,header=None).to_numpy()
#     st.write(phototopic)
    
    # TODO: find these variables
    
    numpoints = len(IV_EL)
    Sample_Name = 'CommercialWhite1'
    
    plt.rc('font', family='Arial')
    plt.rcParams['axes.linewidth'] = 2
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    plt.rcParams['font.size'] = 12
    
    
    #https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
    #https://matplotlib.org/3.5.0/tutorials/colors/colormap-manipulation.html

    global colors
    colors = cm.get_cmap('PuBu', 8)
#     st.write(colors(0.56))
    
    global D_input, A_LED_input, A_phd_input
    
    st.sidebar.header("Adjust Settings")
#     v1, v2, v3 = st.columns(3, gap="medium")
#     with v1:
    D_input = st.sidebar.number_input("D (mm^2)", value=10)
#     with v2:
    A_LED_input = st.sidebar.number_input("A_LED (mm^2)", value=math.pi*4**2)
#     with v3:
    A_phd_input = st.sidebar.number_input("A_phd (mm^2)", value=100)
    
    ######################################################

def graph2():
    st.write("graph2")
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(IV_EL[:,0],IV_EL[:,2],linewidth=2)

    ax.set_xlabel('Bias Voltage(V)')
    ax.set_ylabel('Photourrent(mA)')
    ax.set_title('EL Characteristics of\n White Commercial LED')
    #ax.set_xlim(300,850)
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=10)
#     plt.show()
    st.pyplot(fig)
    
    
def graph3():
    st.write("graph3")
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])

    for k in range(numpoints):
        ax.plot(Spectra[:,0],Spectra[:,k+1],color = colors(k/numpoints), 
                 label=f'{"{:.1f}".format(IV_EL[k,0])}V', linewidth = 0.5)
    
    ax.set_xlabel('Wavelength(nm)')
    ax.set_ylabel('Counts')
    ax.set_title('Electroluminescence Spectra at Each\n Bias Voltage of White Commercial LED')
    #ax.set_xlim(300,850)
    
#     ax.legend(bbox_to_anchor=(2, 1), loc=1.02, frameon=False, fontsize=10, ncol=3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False, fontsize=10, ncol=3)
    
    #ax.set_xticks(ax.get_xticks()[::5])
#     plt.show()
    st.pyplot(fig)

#     plt.savefig(f'IV+Spectra/{date_string}{Sample_Name}_Spectra.png')
    
    ####################################################

def before4():
    global photodiode_data
    
#     photodiode_directory= "PD_responsivities/20220409/PhotodiodeE_000"
#     f = open(photodiode_directory+".qsdat", "r")

    photodiode_file = "PhotodiodeE_000"
    f = open(photodiode_file+'.qsdat', "r")
    
#     st.write(f.read())

    #Converting Data portion of .qsdat file to txt form for pandas
    stringA = "Wavelength("
    stringB = 'END DATA'
    file0 = open(photodiode_file+".qsdat", "r")
    file1 = open(photodiode_file+".txt", "w")
    i=0
    j=0
    k=0
    for line in file0:  
        i+=1
        if stringA in line:
#             print("Data starts line", i)
#             st.write("Data starts line", i)
            j+=1
        if stringB in line:
#             print("Data ends line", i)
#             st.write("Data ends line", i)
            k+=1   
        if j==1 and k==0:
            file1 = open(photodiode_file+".txt", "a")
            file1.write(line)
            file1.close()
    
    photodiode_data = pd.read_csv(photodiode_file+".txt", delimiter='\t')
#     st.write(photodiode_data)

    photodiode_data=photodiode_data.to_numpy()
    
    
    ##########################################################
    
def graph4():
    st.write("graph4")
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(photodiode_data[:,0],photodiode_data[:,3],linewidth=2)

    ax.set_xlabel('Wavelength(nm)')
    ax.set_ylabel('Responsivity (A/W)')
    ax.set_title('Responsivity Function of Photodiode E')
    #ax.set_xlim(300,850)
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=10)
#     plt.show()
    st.pyplot(fig)
    
    
    ##########################################################

def graph5():
    st.write("graph5")
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(photodiode_data[:,0],photodiode_data[:,2],linewidth=2)

    ax.set_xlabel('Wavelength($\lambda$)')
    ax.set_ylabel('EQE(%)')
    ax.set_title('Wavelength dependent EQE of device\n measured by Photodiode E?')
    #ax.set_xlim(300,850)
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=10)
#     plt.show()
    st.pyplot(fig)
    
#     ##########################################################

def before6():
    global D, A_LED, A_phd, Omega_phd, Selected_EL_Spectrum, normalized_EL_Spectrum, normalized_EL_Spectra
    
#     D = 10 #mm distance between sample and photodetector
#     #A_LED = 4.5 #mm^2
#     A_LED = math.pi*4**2 #mm^2
#     A_phd = 100 #mm^2
#     #treating the LED as a point source as it is much smaller than the photodetector active area
    D = D_input
    A_LED = A_LED_input
    A_phd = A_phd_input
    
    
    #angle subtended by the photodetector
    Omega_phd = 2*math.pi*(1-math.cos(math.sqrt(A_phd/math.pi)/D))
#     Omega_phd #units sr (The steradian or square radian is the SI unit of solid angle)
#     st.write(Omega_phd)
    
    #Selecting max EL spectrum to normalize and continue with calculation
    Selected_EL_Spectrum = 42
    normalized_EL_Spectrum = Spectra[:,Selected_EL_Spectrum+1]/np.amax(Spectra[:,Selected_EL_Spectrum+1])
    normalized_EL_Spectra=Spectra.copy()
    for i in range(numpoints-1): #This is because the 0V column is entirely zeros
        normalized_EL_Spectra[:,i+2] = Spectra[:,i+2]/np.amax(Spectra[:,i+2])
    
    
#     ##########################################################

def graph7():
    st.write("graph7")
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])

    for k in range(numpoints):
        ax.plot(normalized_EL_Spectra[:,0],normalized_EL_Spectra[:,k+1],color = colors(k/numpoints), 
                 label=f'{IV_EL[k,0]}V', linewidth = 1)

    ax.set_xlabel('Wavelength(nm)')
    ax.set_ylabel('Counts')
    ax.set_title('Normalized Electroluminescence Spectra at Each\n Bias Voltage of White Commercial LED')
    #ax.set_xlim(300,850)
#     ax.legend(bbox_to_anchor=(2, 1), loc=1, frameon=False, fontsize=10, ncol=3, )
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False, fontsize=10, ncol=3)
    #ax.set_xticks(ax.get_xticks()[::5])
#     plt.show()
    st.pyplot(fig)

    #plt.savefig(f'IV+Spectra/{date_string}{Sample_Name}_Spectra.png')


def before8():
    global e, h, c, calculated_QEs
    
    e=1.602176634e-19 #[C]
    h=6.62607015e-34 #[J.s]
    c=299792458 #[m.s-1]
    
    #Now attempting to convert from responsivity to quantum efficiency of the photodetector:
    calculated_QEs = photodiode_data[:,3]*h*c/(e*photodiode_data[:,0]*1e-9)
    
    
    ##########################################################
    
#BRAVO! QUANTUM EFFICIENCIES OF PHOTODETECTOR VERIFIED!!!! So now we know the EQE column in the 
#photodiode calibration file are the QEs of the photodetector itself:)

#Photodiode data and spectral data do not have consistent wavelength increments so I will use interpolation

#     st.write(np.amin(Spectra[:,0]),
#      np.amin(photodiode_data[:,0]),
#      np.amax(Spectra[:,0]),
#      np.amax(photodiode_data[:,0]))
    
#Infer QE of photodiode at a specific wavelength using interpolation
#X~wavelength, Y~Photodiode QE
def interpolate(Xmin,Xmax,Y1,Y2,currentXval):
    frac = (currentXval-Xmin)/(Xmax-Xmin)
    y = frac*(Y2-Y1)+Y1
    return y
    
#Finding lower and upper bounds for interpolation from photodiode array data. 
#Upper bound index will really just turn out to be lower bound +1
def upper_lower(wavelength,array):
    min_index=0
    for i in range(len(array)):
        if array[i,0]<wavelength:
            min_index+=1
            #max_index+=1
    result = min_index-1
    return result
    
@st.cache
def slow_computation(Cs):
    for a in range(numpoints):
        C_numerator = 0
        C_denominator = 0
        for b in range(2047): 
            Xcurrent = normalized_EL_Spectra[b,0]
            Xmin_index = upper_lower(Xcurrent,photodiode_data)
            Xmin = photodiode_data[Xmin_index,0]
            Xmax = photodiode_data[Xmin_index+1,0]
            Ymin = photodiode_data[Xmin_index,2]/100
            Ymax = photodiode_data[Xmin_index+1,2]/100
            QE = interpolate(Xmin,Xmax,Ymin,Ymax,Xcurrent)
            dlambda = normalized_EL_Spectra[b+1,0]-normalized_EL_Spectra[b,0]*1e-9
            C_numerator+=normalized_EL_Spectra[b,a+1]*QE*dlambda
            C_denominator+=normalized_EL_Spectra[b,a+1]*dlambda
        C=C_numerator/C_denominator
        Cs[a]=C

    return Cs
    
def before9():
    global Cs, IV_EL, Phi_phd_array
    
    #Doing out the integrals in to calculate C
    Cs = np.zeros((numpoints,))
    
    Cs = slow_computation(Cs)
    
    #Note there are some negative C values and this is most likely due to negative spectrum 
    #readings from dark-subtracted spectra?
    
    Phi_phd_array = np.zeros((numpoints,))
    for a in range(numpoints):
        Phi_phd = IV_EL[a,2]/(1000*(Omega_phd*Cs[a])*e)
        Phi_phd_array[a,]=Phi_phd #[photons.s-1.sr-1]
        
    #Adding a column to IV_EL for photon flux:
    IV_EL = np.append(IV_EL, np.zeros((numpoints,1)), axis=1)
    IV_EL[:,3]=Phi_phd_array #[photons.s-1.sr-1]
    

def graph9():
    st.write("graph9")
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(IV_EL[:,0],IV_EL[:,3],linewidth=2)

    ax.set_xlabel('Bias Voltage(V)')
    ax.set_ylabel('Photon flux ($photon.s^{-1}.sr^{-1}$)')
    ax.set_title('Incident Photon Flux on Photodiode\nfrom White Commercial LED')
    #ax.set_xlim(300,850)
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=10)
#     plt.show()
    st.pyplot(fig)
    
    
    ##########################################################
    
def before10():
    global IV_EL # TODO: possibly include R_prime and R and E_photon_array
    
    #Average energy per photon
    E_photon_array = np.zeros((numpoints,))
    for a in range(numpoints):
        E_numerator = 0
        E_denominator = 0
        for b in range(2047):
            dlambda = normalized_EL_Spectra[b+1,0]-normalized_EL_Spectra[b,0]*1e-9
            wavelength = normalized_EL_Spectra[b,0]*1e-9
            E_numerator+=normalized_EL_Spectra[b,a+1]*h*c*dlambda/wavelength
            E_denominator+=normalized_EL_Spectra[b,a+1]*dlambda
        E_photon=E_numerator/E_denominator   #[J/photon]
        E_photon_array[a]=E_photon
    
#     st.write(E_photon_array)

    R_prime = IV_EL[:,3]*E_photon_array  #Radiant Intensity [W.sr-1]
#     st.write(R_prime)

    R = R_prime/(A_LED*1e-6)
#     st.write(R) #Radiance [W.sr-1.m-2]

    
    #Adding a column to IV_EL for radiance:
    IV_EL = np.append(IV_EL, np.zeros((numpoints,1)), axis=1)
    IV_EL[:,4]=R #columns: V, I, Iphd, Photon Flux, Radiance
    
def graph10():
    st.write("graph10")
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(IV_EL[:,0],IV_EL[:,4],linewidth=2)

    ax.set_xlabel('Bias Voltage(V)')
    ax.set_ylabel('Radiance ($W.sr^{-1}.m^{-2}$)')
    ax.set_title('LED Radiance vs. Voltage \nfor White Commercial LED')
    #ax.set_xlim(300,850)
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=10)
#     plt.show()
    st.pyplot(fig)
    
    
    ##########################################################
    
    #External Quantum Efficiency
    
#     st.write(math.pi)

def before11():
    global IV_EL
    
    #Lambertian distribution leads to photon flux
    phi = math.pi*Phi_phd_array  #[photons.s-1]
    
    EQE_array = np.zeros((numpoints,))
    for a in range(numpoints):
        EQE = phi[a]/(IV_EL[a,1]/(1000*e))
        EQE_array[a] = EQE*100 #%
#     st.write(EQE_array)
    
    IV_EL = np.append(IV_EL, np.zeros((numpoints,1)), axis=1)
    IV_EL[:,5]=EQE_array #columns: V, I, Iphd, Photon Flux, Radiance, EQE
    
    #Adding a column for J [mA.cm-2]
    IV_EL = np.append(IV_EL, np.zeros((numpoints,1)), axis=1)
    IV_EL[:,6]=IV_EL[:,1]/(A_LED*1e-2) #columns: V, I, Iphd, Photon Flux, Radiance, EQE, J
    
    
    ##########################################################
    
def graph12(EQE):
    st.write("graph12")
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(IV_EL[:,6]/1000,IV_EL[:,5],linewidth=2)

    ax.set_xlabel('Current Density (A/$cm^{-2}$)')
    ax.set_ylabel('EQE(%)')
    ax.set_title('LED EQE vs. Current Density \nfor White Commercial LED')
#     ax.set_xlim(1e-8,1+5)
#     ax.set_ylim(0,100)
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=10)

    ax.set_yscale(EQE)
    
    # TODO: adjust range
    if EQE=='linear':
        ax.set_xlim(1e-8,1+5)
        ax.set_ylim(0,100)
    else:
        ax.set_xlim(1e-2,1+2)
        ax.set_ylim(0,80)
    
    
    ax.set_xscale('log') #as opposed to 'linear'
#     plt.show()
    st.pyplot(fig)
    
    
    ##########################################################
    
    # Luminance and current efficacy
    

def graph15():
    st.write("graph15")
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(phototopic[:,0],phototopic[:,1],linewidth=2)

    ax.set_xlabel('Wavelength(nm)')
    ax.set_ylabel('Phototopic factor')
    ax.set_title('Phototopic response of the human eye to light')
    #ax.set_xlim(1e-5,1e3+5000)
    #ax.set_ylim(0,100)
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=10)

    #ax.set_xscale('log') #as opposed to 'linear'
#     plt.show()
    st.pyplot(fig)
    
    
    ##########################################################
    
@st.cache
def slow_computation2(Ks, phototopic_scaling):
    for a in range(numpoints):
        K_numerator = 0
        K_denominator = 0
        for b in range(2047): 
            if normalized_EL_Spectra[b,0]>np.amin(phototopic[:,0]) and normalized_EL_Spectra[b,0]<np.amax(phototopic[:,0]):
                Xcurrent = normalized_EL_Spectra[b,0]
                Xmin_index = upper_lower(Xcurrent,phototopic)
                Xmin = phototopic[Xmin_index,0]
                Xmax = phototopic[Xmin_index+1,0]
                Ymin = phototopic[Xmin_index,1]
                Ymax = phototopic[Xmin_index+1,1]
                response = interpolate(Xmin,Xmax,Ymin,Ymax,Xcurrent)
                dlambda = normalized_EL_Spectra[b+1,0]-normalized_EL_Spectra[b,0]*1e-9
                K_numerator+=response*phototopic_scaling*normalized_EL_Spectra[b,a+1]*h*c*dlambda/(Xcurrent*1e-9)
                K_denominator+=normalized_EL_Spectra[b,a+1]*dlambda
        K=K_numerator/K_denominator
        Ks[a]=K
    return Ks
    
def before16():
    global IV_EL, Ks, L_prime
    
    #The above is normalized - it needs to be scaled by ~683.002 lm·W-1 before being used in further calcs 
    phototopic_scaling = 683.002 #lm·W-1
    
    #In order to take into account the perception of light by the human eye we weight the S(𝜆) by the
    #photopic function 𝑃(𝜆).
    Ks = np.zeros((numpoints,))
    Ks = slow_computation2(Ks, phototopic_scaling)
    
    
#     st.write(Ks) #lm.s.photon^-1
    
    #Luminous intensity 
    L_prime = np.zeros((numpoints,))
    for a in range(numpoints):
        L_prime[a] = Phi_phd_array[a]*Ks[a]
#     st.write(L_prime)

    ##Cross checking using second method of calculation
#     L_prime2 = np.zeros((numpoints,))
#     for a in range(numpoints):
#         L_prime2[a] = Ks[a]*IV_EL[a,2]*1e-3/(e*Cs[a]*Omega_phd)
    
    IV_EL = np.append(IV_EL, np.zeros((numpoints,1)), axis=1)
    IV_EL[:,7]=L_prime #columns: V, I, Iphd, Photon Flux, Radiance, EQE, J, Luminous intensity (cd)
    
    
    
    ##########################################################
    
def before17():
    global IV_EL
    L= L_prime/(A_LED*1e-6) #[cd.m-2]
    
    IV_EL = np.append(IV_EL, np.zeros((numpoints,1)), axis=1)
    IV_EL[:,8]=L #columns: V, I, Iphd, Photon Flux, Radiance, EQE, J, Luminous intensity (cd), Luminance (cd/m^2),
    
    
    ##########################################################
    
def graph17():
    st.write("graph17")
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(IV_EL[:,6]/1000,IV_EL[:,8],linewidth=2)

    ax.set_xlabel('Current Density (A/$cm^{-2}$)')
    ax.set_ylabel('Luminance (cd/$m^{-2}$)')
    ax.set_title('Luminance vs. Current Density \nfor White Commercial LED')
    #ax.set_xlim(0,2000)
    #ax.set_ylim(0,2e-16)
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=10)

    #ax.set_xscale('log') #as opposed to 'linear'
#     plt.show()
    st.pyplot(fig)
    
def before19():
    global IV_EL
    
    #columns: V, I, Iphd, Photon Flux, Radiance, EQE, J, Luminous intensity (cd), Luminance (cd/m^2),
    #luminous efficacy 
    eta_current = np.zeros((numpoints,))
    eta_lum = np.zeros((numpoints,))
    for a in range(numpoints):
        eta_c= Ks[a]*IV_EL[a,2]/(e*IV_EL[a,1]*Cs[a]*Omega_phd)
        eta_current[a]=eta_c
        eta_l= math.pi*Ks[a]*IV_EL[a,2]*1e-3/(e*Cs[a]*Omega_phd*IV_EL[a,0]*IV_EL[a,1]*1e-3)
        eta_lum[a]=eta_l
    
#     #Cross checking using second calculation method
#     eta_current2=L_prime/(IV_EL[:,1]*1e-3)
    
#     #Cross checking using second method
#     eta_lum2 = L_prime*math.pi/(IV_EL[:,0]*IV_EL[:,1]*1e-3)
    
    IV_EL = np.append(IV_EL, np.zeros((numpoints,1)), axis=1)
    IV_EL[:,9]=eta_current 
    #columns: V, I, Iphd, Photon Flux, Radiance, EQE, J, Luminous intensity (cd), Luminance (cd/m^2), 
    #eta_current (cd/A), eta_lum (lm/electricalW)
    
    IV_EL = np.append(IV_EL, np.zeros((numpoints,1)), axis=1)
    IV_EL[:,10]=eta_lum #columns: V, I, Iphd, Photon Flux, Radiance, EQE, J, Luminous intensity (cd), Luminance (cd/m^2),
    
    
    ##########################################################
    
def graph22():
    st.write("graph22")
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(IV_EL[:,6]/1000,IV_EL[:,10],linewidth=2)

    ax.set_xlabel('Current Density (A/$cm^{-2}$)')
    ax.set_ylabel('Luminous Efficacy (lm/W)')
    ax.set_title('Luminous Efficacy vs. Current Density \nfor White Commercial LED')
    ax.set_xlim(1e-2,2.3)
    ax.set_ylim(0,250)
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=10)

    ax.set_xscale('log') #as opposed to 'linear'
#     plt.show()
    st.pyplot(fig)
    
    
    ##########################################################
    
def graph26(current, luminance):
    st.write("graph26")
    #Now plotting a JVL curve
    fig, ax1 = plt.subplots(figsize=(4, 4))
    ax2 = ax1.twinx()

    line1, = ax1.plot(IV_EL[:,0],IV_EL[:,6],linewidth=2, color ='green', label = 'Current Density')
    line2, = ax2.plot(IV_EL[:,0],IV_EL[:,8],linewidth=2, label = 'Luminance')

    ax1.legend(handles=[line1, line2], fontsize = 10)

    ax1.set_xlabel(r'Voltage (V)', labelpad=10)
    ax1.set_ylabel('Current density (mA$.cm^{-2}$)', labelpad=10)
    ax1.set_title('JVL curve \nfor White Commercial LED', fontsize = 14)
    ax2.set_ylabel('Luminance (cd.$m^{-2}$)')
    
    ax1.set_yscale(current)
    ax2.set_yscale(luminance)
    st.pyplot(fig)

    
    ##########################################################
    

def do_stuff():
        
#     graph3()
    before4()
    
    before6()
#     graph3()
    before8()
    before9()
    before10()
    before11()
    before16()
    before17()
    before19()
    
    
    st.sidebar.header("Select the plots to show:")
    
    g26 = st.sidebar.checkbox("Current and Luminance vs. Voltage", value=True)
    if g26:
        col1, col2 = st.sidebar.columns(2, gap="medium")
        with col1:
            current26 = st.select_slider('Current', options=['log','linear'], value='log')
        with col2:
            luminance26 = st.select_slider('Luminance', options=['log','linear'], value='log')
        
        graph26(current=current26, luminance=luminance26)
    
    # TODO: graph14 (deleted) has a different x/y limit
    g12 = st.sidebar.checkbox("EQE% vs. Current Density", value=True)
    if g12:
        col1, col2 = st.sidebar.columns(2, gap="medium")
        with col1:
            EQE12 = st.select_slider('EQE%', options=['log','linear'], value='linear')
        graph12(EQE=EQE12)
    
    g17 = st.sidebar.checkbox("Luminous Efficacy vs Current Density", value=True)
    if g17:
        graph17()
    
    g22 = st.sidebar.checkbox("Luminance vs Current Density", value=True)
    if g22:
        graph22()
    
    g3 = st.sidebar.checkbox("Electroluminescence (EL) Spectra", value=True)
    if g3:
        graph3()
        
    g7 = st.sidebar.checkbox("Normalized EL Spectra", value=True)
    if g7:
        graph7()
    
    g2 = st.sidebar.checkbox("Photocurrent vs. Voltage")
    if g2:
        graph2()
        
    g9 = st.sidebar.checkbox("Photon Flux vs. Voltage")
    if g9:
        graph9()
        
    g10 = st.sidebar.checkbox("Radiance vs. Voltage")
    if g10:
        graph10()
        

    st.sidebar.write("")
    st.sidebar.header("Calibration Plots")
    
    g5 = st.sidebar.checkbox("Photodetector EQE vs. Wavelength")
    if g5:
        graph5()
        
    g4 = st.sidebar.checkbox("Responsivity vs. Wavelength")
    if g4:
        graph4()
        
    g15 = st.sidebar.checkbox("Phototopic Function of the Human Eye (Phototopic Factor vs. Wavelength)")
    if g15:
        graph15()
     
    st.sidebar.write("")
    st.sidebar.write("")

    
if __name__ == '__main__':
    intro()
    
    with st.expander('Uploads', expanded=True):
#         col1, col2, col3 = st.columns(3, gap="small")
#         with col1:
        spectra_input = st.file_uploader("Upload a spectra CSV")
#         with col2:
        IV_photo_input = st.file_uploader("Upload an IV+photocurrent CSV")
#         with col3:
        photo_data_input = st.file_uploader("Upload photodetector data CSV")
        
        placeholder = st.empty()
        test = placeholder.button("DEVELOPMENT: USE DEFAULT FILES")

    if "load_state" not in st.session_state:
        st.session_state.load_state = False
        
    if spectra_input and IV_photo_input and photo_data_input is not None:
        st.session_state.load_state = False
        placeholder.empty()
        try:
            st.write("Displaying plots based on your uploads:")
            pre(spectra_input, IV_photo_input, photo_data_input)
            do_stuff()
        except:
            st.error("Check your uploads for errors/formatting issues!")
    
    if (test or st.session_state.load_state):
        st.write("Displaying plots based on default data:")
        st.session_state.load_state = True
        a = '2022-05-24Commercial_White1_spectra.csv'
        b = '2022-05-24Commercial_White1IV+photocurrent.csv'
        c = 'StranksPhototopicLuminosityFunction.csv'
        
        pre(a, b, c)
#         set_vars()
        do_stuff()
        
