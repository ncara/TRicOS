# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:42:42 2022

@author: NCARAMEL
"""

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
import sys
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
    base=df.absor[df.wl.between(650,700)].mean()
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




plt.rcParams["figure.figsize"] = (40/2.54,30/2.54)

####Parsing through the dir to find spectra and importing/treating them#### 
directory = './' #C:\Users\ncaramel\Work Folders\Documents\pyrep\TRicOS_test_data

print('Please input only processed spectra files, preferentially scaled averaged smoothed files')

path_to_spec1=sys.argv[1]
path_to_spec2=sys.argv[2]


# path_to_spec1='scaled_averaged_smoothed_300us.csv'
# path_to_spec2='scaled_averaged_smoothed_600us.csv'

# listspec=[]
# numspec=[]
# for entry in os.scandir(directory):  
#     if entry.path.endswith(".txt") and entry.is_file() and ("3U2" in entry.path) :
#         listspec.append(entry.path)
listspec=[path_to_spec1,path_to_spec2]
print(listspec)


str="Hello, World!"
print("World" in str)


spectras={}
timecode={}
# spec=pd.DataFrame(data=listspec,index=numspec,columns=["paths"])
for nomfich in listspec:
    print(nomfich)
    
    spectras[nomfich]=pd.read_csv(filepath_or_buffer= nomfich,   #pandas has a neat function to read char-separated tables
                        sep= ",",             #Columns delimited by semicolon
                        decimal=".",              #decimals delimited by colon
                        skip_blank_lines=True,        #There is a blank line at the end
                        skipfooter=0,             #2 lines of footer, not counting the blank line
                        header=0,
                        # skiprows=8,
                        # index_col=0,
                        # names=["wl","absor"],
                        engine="python")          #The python engine is slower but supports header and footers
    spectras[nomfich].index=spectras[nomfich].wl
    n=0
    timecode[nomfich]=''
    for char in nomfich:
        if char.isdigit() :
            print(n)
            timecode[nomfich]+=char
            lastnumber=n
        n+=1
    if nomfich[lastnumber+1]=='s' :
        timecode[nomfich]+='s'
    else :
        timecode[nomfich]+=nomfich[lastnumber+1:lastnumber+3]

    
#%%% plot dark-light spectra

diff_spec=spectras[path_to_spec2].copy()
for i in diff_spec.index :
    diff_spec.absor[i]=spectras[path_to_spec2].absor[i]-spectras[path_to_spec1].absor[i]
    
    

globmin=diff_spec.absor[diff_spec.wl.between(280,800)].min()

globmax=diff_spec.absor[diff_spec.wl.between(280,800)].max()


# globmax=diff_spec.absor[diff_spec.wl.between(280,700)].max()

fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [AU]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='bright', n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 

ax.plot(diff_spec.wl,                  #x-axis is wavelength
        diff_spec.absor ,                   #y-axis is absor, or emission, or else
        linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
        label=timecode[path_to_spec1] + ' - ' + timecode[path_to_spec2],
        color=palette[0])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect



ax.set_title(timecode[path_to_spec1] + ' - ' + timecode[path_to_spec2] + ' in crystallo absorbance difference spectrum', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([280,700]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
# ax.yaxis.set_ticks(np.arange(round(globmin,1), round(globmax+0.1,1), 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)
# legend = plt.legend(loc='upper right', shadow=True, prop={'size':7})
    
    
# plt.show()





figfilename = "difference_spectrum_" + timecode[path_to_spec1] + '-' + timecode[path_to_spec2] + ".pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "difference_spectrum_" + timecode[path_to_spec1] + '-' + timecode[path_to_spec2] + ".png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "difference_spectrum_" + timecode[path_to_spec1] + '-' + timecode[path_to_spec2] + ".svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
plt.close()

diff_spec.to_csv("difference_spectrum_" + timecode[path_to_spec1] + '-' + timecode[path_to_spec2] + ".csv", index=True)
   
