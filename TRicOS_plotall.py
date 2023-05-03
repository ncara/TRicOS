#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:36:40 2021

@author: Caramello
"""
import pandas as pd #This is the DataFrame package, DataFrames are like excell tables with neat properties
import matplotlib.pyplot as plt #this is the plot package
import numpy as np   #Numerical operations such as exp, log ect
import os as os      # Architecture package, to handle directories ect
import scipy as sp   #Some useful tools, especially for processing spectra in scipy.signal
from scipy.optimize import curve_fit # I don't know why but that one function refuses to come with the rest of the package, it gets its own import
import scipy.signal 
import seaborn as sns    #Visually distinct color palette 
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.sparse as sparse
import math as mth
#import scipy.spsolve as spsolve

os.chdir('./') 

def closest(lst, K): 
     lst = np.asarray(lst) 
     idx = (np.absor(lst - K)).argmin() 
     return(lst[idx])


def fct_baseline(x, a, b):
    return(a/np.power(x,4)+b)

def fct_relaxation_monoexp(x,a,b,tau):
    return(a-b*np.exp(-x/tau))

#def fct_relaxation_bi_exp(x,a,b1,tau1,b2,tau2):
#    return(a-b1*np.exp(-x/tau1)-b2*np.exp(-x/tau2))

def rescale_corrected(df, wlmin, wlmax): #scales with a factor corresponding to 
    a=df.copy() #df[df.wl.between(wlmin,wlmax)].copy()
    # offset=a.absor[a.wl.between(wlmax-10,wlmax-1)].min()
    # a.absor=a.absor.copy()-offset        
#    scale=df[df.wl.between(wlmin,wlmax)].mean()
    fact=1/a.absor[a.wl.between(wlmin,wlmax,inclusive="both")].max() #1/a.absor[a.wl.between(425,440)].max()
    a.absor=a.absor.copy()*fact      
    return(a.copy())






def rescale_raw(rawdata, df, wlmin, wlmax):
    a=df.copy()
    raw=rawdata.copy()
    # offset=a.absor[a.wl.between(wlmax-10,wlmax-1)].min()
    # a.absor=a.absor.copy()-offset        
    fact=1/a.absor[a.wl.between(wlmin,wlmax,inclusive="both")].max() #1/a.absor[a.wl.between(425,440)].max()
    raw.absor=raw.absor.copy()*fact      
    return(raw.copy())




def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def baselinefitcorr_3seg_smooth(df,  segment1, segment2, segmentend, sigmaby3segment):
    #segmentend=df.wl.between(600,800)
    segment=segment1+segment2+segmentend
    #min1 = closest(df.wl,minrange1)
    #min2 = closest(df.wl,minrange2)
    x=df.wl[segment].copy()
    y=df.absor[segment].copy()
    initialParameters = np.array([1e9, 0])
    # sigma=[1,0.01,1,0.01]
    n=len(df.absor[segment1])
    sigma=n*[sigmaby3segment[0]]
    n=len(df.absor[segment2])
    sigma=sigma + n*[sigmaby3segment[1]]
    n=len(df.absor[segmentend])
    sigma=sigma + n*[sigmaby3segment[2]]
    #print(sigma)
    para, pcov = sp.optimize.curve_fit(f=fct_baseline, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
    baseline=df.copy()
    baseline.absor=fct_baseline(baseline.wl.copy(), *para)
    #baseline.absor=baseline_als(np.array(y), 10^5, 0.01)
    #plt.plot(df.wl,df.absor)
    #plt.plot(baseline.wl, baseline.absor)
    #plt.show()
    corrected=df.copy()
    corrected.absor=df.absor.copy()-baseline.absor
    return(corrected)


def baselinefit_cst(df):
    base=df.absor[df.wl.between(700,800)].mean()
    # print(base)
    corrected=df.copy()
    corrected.absor=df.absor.copy()-base
    return(corrected)


def absorbance(tmp):
    ourdata=tmp.copy()
    ourdata['absor']=None
    for wl in ourdata.index:
        tmpdat=ourdata.I[wl].copy()
        tmpref=ourdata.I0[wl].copy()
        tmpabs=-np.log(tmpdat/tmpref)
        ourdata.absor[wl]=tmpabs
    # print(ourabs)
    # ourdata['absor']=ourabs
    return(ourdata)

floatize=np.vectorize(float)          
limit_plot=1000
limbas=8000

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True


# plt.rcParams["figure.figsize"] = (20/2.54,15/2.54)

####Parsing through the dir to find spectra and importing/treating them#### 
directory = './' #N:\Documents\spectra\TRicOS\20220523_KIRILL_test_data



listspec=[]
numspec=[]
for entry in os.scandir(directory):  
    if entry.path.endswith(".txt"):# and entry.is_file() and ("3U2" in entry.path) :
        if "ms" in entry.path :
            name_correct=entry.path.replace('ms', '000us')
            os.rename(entry.path, name_correct)
        else :
            name_correct=entry.path
        listspec.append(name_correct)
print(listspec)

raw_lamp={}

# spec=pd.DataFrame(data=listspec,index=numspec,columns=["paths"])
for nomfich in listspec:
    print(nomfich)
    
    raw_lamp[nomfich]=pd.read_csv(filepath_or_buffer= nomfich,   #pandas has a neat function to read char-separated tables
                        sep= ';',             #Columns delimited by semicolon
                        decimal=".",              #decimals delimited by colon
                        skip_blank_lines=True,        #There is a blank line at the end
                        skipfooter=0,             #2 lines of footer, not counting the blank line
                        skiprows=8,
                        index_col=0,
                        header=None,
                        # names=["I","bgd","I0"],
                        engine="python")          #The python engine is slower but supports header and footers
    # raw_lamp[nomfich].index=[round(x, 2) for x in raw_lamp[nomfich].index]
    raw_lamp[nomfich].index=raw_lamp[list(raw_lamp.keys())[0]].index
    if len(raw_lamp[nomfich].columns) == 4:
        raw_lamp[nomfich].columns=['I', 'bgd', 'I0', 'A']
    elif len(raw_lamp[nomfich].columns) == 3:
        raw_lamp[nomfich].columns=['I', 'bgd', 'I0']
# raw_lamp



# plt.plot(raw_lamp['./02_dark.txt'].index,raw_lamp['./02_dark.txt'].A)
# plt.plot(raw_lamp['./03_dark.txt'].index,raw_lamp['./03_dark.txt'].A)


average_signal=raw_lamp[list(raw_lamp.keys())[0]].copy()
average_signal.I=0
average_signal['wl']=floatize(average_signal.index)
average_ground=raw_lamp[list(raw_lamp.keys())[0]].copy()
average_ground.I=0
average_ground['wl']=floatize(average_ground.index)

raw_abs={}
onlysmoothed_spec={}
smoothed_spec={}
scaled_raw_abs={}
scaled_smoothed_spec={}
const_spec={}
for nomfich in raw_lamp:
    raw_abs[nomfich]=absorbance(raw_lamp[nomfich].copy())
    raw_abs[nomfich]['wl']=floatize(raw_abs[nomfich].index)
    raw_abs[nomfich]=raw_abs[nomfich].drop(columns=['I','bgd',"I0"]).copy()
    raw_abs[nomfich].dropna(axis=0, inplace=True)
    
    
    const_spec[nomfich]=baselinefit_cst(raw_abs[nomfich])
    
    # raw_abs[nomfich]=baselinefit_cst(raw_abs[nomfich].copy())

    # raw_ground=absorbance(average_ground.copy())
    # raw_ground=raw_ground.drop(columns=['I','bgd',"I0"]).copy()
    # raw_ground.dropna(axis=0, inplace=True)
    # avg_ground=baselinefit_cst(raw_ground.copy())


    onlysmoothed_spec[nomfich]=raw_abs[nomfich].copy()
    onlysmoothed_spec[nomfich].absor=sp.signal.savgol_filter(x=raw_abs[nomfich].absor.copy(),
                                                     window_length=21,
                                                     polyorder=3) 

    # onlysmoothed_ground=raw_ground.copy()
    # onlysmoothed_ground.absor=sp.signal.savgol_filter(x=raw_ground.absor.copy(),
    #                                                   window_length=21,
    #                                                   polyorder=3) 



    smoothed_spec[nomfich]=baselinefit_cst(onlysmoothed_spec[nomfich].copy())
    # smoothed_spec[nomfich].absor=sp.signal.savgol_filter(x=raw_abs[nomfich].absor.copy(),
    #                                                  window_length=21,
    #                                                  polyorder=3) 



    # smoothed_ground=avg_ground.copy()
    # smoothed_ground.absor=sp.signal.savgol_filter(x=avg_ground.absor.copy(),
    #                                                   window_length=21,
    #                                                   polyorder=3) 

    scaled_raw_abs[nomfich]=rescale_raw(const_spec[nomfich],smoothed_spec[nomfich], 520,580)
    # scaled_avg_ground=rescale_raw(avg_ground,smoothed_ground, 520,580)

    scaled_smoothed_spec[nomfich]=rescale_corrected(smoothed_spec[nomfich],520,580)



#%%% plot averaged non scaled, constant baseline correctedraw spectra

globmax=0
globmin=10
for i in raw_abs.keys():
    if globmax<raw_abs[i].absor[raw_abs[i].wl.between(260,700)].max():
        globmax=raw_abs[i].absor[raw_abs[i].wl.between(260,700)].max()
    if globmin>raw_abs[i].absor[raw_abs[i].wl.between(260,700)].min():
        globmin=raw_abs[i].absor[raw_abs[i].wl.between(260,700)].min()

fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [-]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.08, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='bright', n_colors=len(raw_abs.keys()))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
n=0      

                                      #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in raw_abs.keys():
    ax.plot(raw_abs[i].wl,                  #x-axis is wavelength
            raw_abs[i].absor ,                   #y-axis is absor, or emission, or else
            linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
            label=i[2:-4],
            color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n+=1

ax.set_title('raw in crystallo absorbance spectra (unscaled)', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([220,800]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
# ax.yaxis.set_ticks(np.arange(round(globmin,1), round(globmax+0.1,1), 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)
legend = plt.legend(loc='upper right', shadow=True, prop={'size':11})
    
    
# plt.show()
figfilename = "raw-unscaled_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "raw-unscaled_spec.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "raw-unscaled_spec.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
plt.close()

#%%% plot averaged non scaled, constant baseline correctedraw spectra

globmax=0
globmin=10
for i in const_spec.keys():
    if globmax<const_spec[i].absor[const_spec[i].wl.between(260,700)].max():
        globmax=const_spec[i].absor[const_spec[i].wl.between(260,700)].max()
    if globmin>const_spec[i].absor[const_spec[i].wl.between(260,700)].min():
        globmin=const_spec[i].absor[const_spec[i].wl.between(260,700)].min()

fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [-]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.08, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='bright', n_colors=len(const_spec.keys()))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
n=0      

                                      #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in const_spec.keys():
    ax.plot(const_spec[i].wl,                  #x-axis is wavelength
            const_spec[i].absor ,                   #y-axis is absor, or emission, or else
            linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
            label=i[2:-4],
            color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n+=1

ax.set_title('constant-corrected in crystallo absorbance spectra (unscaled)', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([220,800]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
# ax.yaxis.set_ticks(np.arange(round(globmin,1), round(globmax+0.1,1), 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)
legend = plt.legend(loc='upper right', shadow=True, prop={'size':11})
    
    
# plt.show()
figfilename = "constant-corrected-unscaled_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "constant-corrected-unscaled_spec.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "constant-corrected-unscaled_spec.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
plt.close()


#%%% plot averaged non scaled, constant baseline correctedraw spectra

globmax=0
globmin=10
for i in onlysmoothed_spec.keys():
    if globmax<onlysmoothed_spec[i].absor[onlysmoothed_spec[i].wl.between(260,700)].max():
        globmax=onlysmoothed_spec[i].absor[onlysmoothed_spec[i].wl.between(260,700)].max()
    if globmin>onlysmoothed_spec[i].absor[onlysmoothed_spec[i].wl.between(260,700)].min():
        globmin=onlysmoothed_spec[i].absor[onlysmoothed_spec[i].wl.between(260,700)].min()

fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [-]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.08, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='bright', n_colors=len(onlysmoothed_spec.keys()))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
n=0      

                                      #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in onlysmoothed_spec.keys():
    ax.plot(onlysmoothed_spec[i].wl,                  #x-axis is wavelength
            onlysmoothed_spec[i].absor ,                   #y-axis is absor, or emission, or else
            linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
            label=i[2:-4],
            color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n+=1

ax.set_title('smoothed in crystallo absorbance spectra (unscaled)', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([220,800]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
# ax.yaxis.set_ticks(np.arange(round(globmin,1), round(globmax+0.1,1), 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)
legend = plt.legend(loc='upper right', shadow=True, prop={'size':11})
    
    
# plt.show()
figfilename = "smoothed-unscaled_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "smoothed-unscaled_spec.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "smoothed-unscaled_spec.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
plt.close()


#%%% plot averaged non scaled, constant baseline correctedraw spectra

globmax=0
globmin=10
for i in smoothed_spec.keys():
    if globmax<smoothed_spec[i].absor[smoothed_spec[i].wl.between(260,700)].max():
        globmax=smoothed_spec[i].absor[smoothed_spec[i].wl.between(260,700)].max()
    if globmin>smoothed_spec[i].absor[smoothed_spec[i].wl.between(260,700)].min():
        globmin=smoothed_spec[i].absor[smoothed_spec[i].wl.between(260,700)].min()

fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [-]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.08, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='bright', n_colors=len(smoothed_spec.keys()))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
n=0      

                                      #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in smoothed_spec.keys():
    ax.plot(smoothed_spec[i].wl,                  #x-axis is wavelength
            smoothed_spec[i].absor ,                   #y-axis is absor, or emission, or else
            linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
            label=i[2:-4],
            color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n+=1

ax.set_title('smoothed, baseline corrected in crystallo absorbance spectra (unscaled)', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([220,800]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
# ax.yaxis.set_ticks(np.arange(round(globmin,1), round(globmax+0.1,1), 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)
legend = plt.legend(loc='upper right', shadow=True, prop={'size':11})
    
    
# plt.show()
figfilename = "smoothed_constant-corrected-unscaled_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "smoothed_constant-corrected-unscaled_spec.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "smoothed_constant-corrected-unscaled_spec.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
plt.close()






#%%% plot averaged non scaled, constant baseline correctedraw spectra

globmax=0
globmin=10
for i in scaled_raw_abs.keys():
    if globmax<scaled_raw_abs[i].absor[scaled_raw_abs[i].wl.between(260,700)].max():
        globmax=scaled_raw_abs[i].absor[scaled_raw_abs[i].wl.between(260,700)].max()
    if globmin>scaled_raw_abs[i].absor[scaled_raw_abs[i].wl.between(260,700)].min():
        globmin=scaled_raw_abs[i].absor[scaled_raw_abs[i].wl.between(260,700)].min()

fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [-]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.08, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='bright', n_colors=len(scaled_raw_abs.keys()))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
n=0      

                                      #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in scaled_raw_abs.keys():
    ax.plot(scaled_raw_abs[i].wl,                  #x-axis is wavelength
            scaled_raw_abs[i].absor ,                   #y-axis is absor, or emission, or else
            linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
            label=i[2:-4],
            color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n+=1

ax.set_title('raw in crystallo absorbance spectra (scaled)', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([220,800]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
# ax.yaxis.set_ticks(np.arange(round(globmin,1), round(globmax+0.1,1), 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)
legend = plt.legend(loc='upper right', shadow=True, prop={'size':11})
    
    
# plt.show()
figfilename = "raw-scaled_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "raw-scaled_spec.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "raw-scaled_spec.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
plt.close()





#%%% plot averaged non scaled, constant baseline correctedraw spectra

globmax=0
globmin=10
for i in scaled_smoothed_spec.keys():
    if globmax<scaled_smoothed_spec[i].absor[scaled_smoothed_spec[i].wl.between(260,700)].max():
        globmax=scaled_smoothed_spec[i].absor[scaled_smoothed_spec[i].wl.between(260,700)].max()
    if globmin>scaled_smoothed_spec[i].absor[scaled_smoothed_spec[i].wl.between(260,700)].min():
        globmin=scaled_smoothed_spec[i].absor[scaled_smoothed_spec[i].wl.between(260,700)].min()

fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [-]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.08, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='bright', n_colors=len(scaled_smoothed_spec.keys()))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
n=0      

                                      #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in scaled_smoothed_spec.keys():
    ax.plot(scaled_smoothed_spec[i].wl,                  #x-axis is wavelength
            scaled_smoothed_spec[i].absor ,                   #y-axis is absor, or emission, or else
            linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
            label=i[2:-4],
            color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n+=1

ax.set_title('smoothed in crystallo absorbance spectra (scaled)', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([220,800]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
# ax.yaxis.set_ticks(np.arange(round(globmin,1), round(globmax+0.1,1), 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)
legend = plt.legend(loc='upper right', shadow=True, prop={'size':11})
    
    
# plt.show()
figfilename = "smoothed-scaled_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "smoothed-scaled_spec.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "smoothed-scaled_spec.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
plt.close()


