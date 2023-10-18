#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:36:40 2051

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
    if 'absor' in ourdata.columns :
        return(ourdata.drop(columns=['I','bgd',"I0"]).copy())
        print('avantes absorbance saved for spectrum')
    else :
        ourdata['absor']=None
        for wl in ourdata.index:
            tmpdat=ourdata.I[wl].copy() - ourdata.bgd[wl].copy()
            tmpref=ourdata.I0[wl].copy() - ourdata.bgd[wl].copy()
            if tmpdat/tmpref < 0:
                tmpabs=0
            else :
                tmpabs=-np.log(tmpdat/tmpref)
            ourdata.absor[wl]=tmpabs
        return(ourdata)










def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev)**2)

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
        raw_lamp[nomfich].columns=['I', 'bgd', 'I0', 'absor']
    elif len(raw_lamp[nomfich].columns) == 3:
        raw_lamp[nomfich].columns=['I', 'bgd', 'I0']
# raw_lamp



# plt.plot(raw_lamp['./02_dark.txt'].index,raw_lamp['./02_dark.txt'].A)
# plt.plot(raw_lamp['./03_dark.txt'].index,raw_lamp['./03_dark.txt'].A)

for nomfich in raw_lamp:
    if int(nomfich[2:4])%2==0 :
        average_signal=pd.DataFrame(index=raw_lamp[nomfich].index, columns = raw_lamp[nomfich].columns)
#        average_signal.I=0
        average_signal['wl']=floatize(average_signal.index)
#    break
average_ground=pd.DataFrame(index=raw_lamp[list(raw_lamp.keys())[0]].index, columns = raw_lamp['./01_dark.txt'].columns)
#average_ground.I=0
average_ground['wl']=floatize(average_ground.index)

for nomfich in raw_lamp:
    if int(nomfich[2:4])%2==0 :
        print(nomfich,'pair : signal')
        for wavelength in average_signal.wl:
            if np.isnan(average_signal.loc[wavelength,'I']):
                average_signal.loc[wavelength,'I']=raw_lamp[nomfich].loc[wavelength,'I']
            else:
                average_signal.loc[wavelength,'I']=(average_signal.loc[wavelength,'I']+raw_lamp[nomfich].loc[wavelength,'I'])/2
        for wavelength in average_signal.wl:
            if np.isnan(average_signal.loc[wavelength,'I0']):
                average_signal.loc[wavelength,'I0']=raw_lamp[nomfich].loc[wavelength,'I0']
            else:
                average_signal.loc[wavelength,'I0']=(average_signal.loc[wavelength,'I0']+raw_lamp[nomfich].loc[wavelength,'I0'])/2
        for wavelength in average_signal.wl:
            if np.isnan(average_signal.loc[wavelength,'bgd']):
                average_signal.loc[wavelength,'bgd']=raw_lamp[nomfich].loc[wavelength,'bgd']
            else:
                average_signal.loc[wavelength,'bgd']=(average_signal.loc[wavelength,'bgd']+raw_lamp[nomfich].loc[wavelength,'bgd'])/2
        if 'absor' in raw_lamp[nomfich] :    
            for wavelength in average_signal.wl:
                if np.isnan(average_signal.loc[wavelength,'absor']):
                    average_signal.loc[wavelength,'absor']=raw_lamp[nomfich].loc[wavelength,'absor']
                else:
                    average_signal.loc[wavelength,'absor']=(average_signal.loc[wavelength,'absor']+raw_lamp[nomfich].loc[wavelength,'absor'])/2
    elif int(nomfich[2:4])%2!=0 :
        print(nomfich,'impair : ground')
        for wavelength in average_ground.wl:
            if np.isnan(average_ground.loc[wavelength,'I']):
                average_ground.loc[wavelength,'I']=raw_lamp[nomfich].loc[wavelength,'I']
            else:
                average_ground.loc[wavelength,'I']=(average_ground.loc[wavelength,'I']+raw_lamp[nomfich].loc[wavelength,'I'])/2
        for wavelength in average_ground.wl:
            if np.isnan(average_ground.loc[wavelength,'I0']):
                average_ground.loc[wavelength,'I0']=raw_lamp[nomfich].loc[wavelength,'I0']
            else:
                average_ground.loc[wavelength,'I0']=(average_ground.loc[wavelength,'I0']+raw_lamp[nomfich].loc[wavelength,'I0'])/2
        for wavelength in average_ground.wl:
            if np.isnan(average_ground.loc[wavelength,'bgd']):
                average_ground.loc[wavelength,'bgd']=raw_lamp[nomfich].loc[wavelength,'bgd']
            else:
                average_ground.loc[wavelength,'bgd']=(average_ground.loc[wavelength,'bgd']+raw_lamp[nomfich].loc[wavelength,'bgd'])/2
        if 'absor' in raw_lamp[nomfich] :   
            for wavelength in average_ground.wl:
                if np.isnan(average_ground.loc[wavelength,'absor']):
                    average_ground.loc[wavelength,'absor']=raw_lamp[nomfich].loc[wavelength,'absor']
                else:
                    average_ground.loc[wavelength,'absor']=(average_ground.loc[wavelength,'absor']+raw_lamp[nomfich].loc[wavelength,'absor'])/2
    else:
        print('invalid file name')





raw_spec=absorbance(average_signal.copy())
#raw_spec=raw_spec.drop(columns=['I','bgd',"I0"]).copy()
raw_spec.dropna(axis=0, inplace=True)
avg_spec=baselinefit_cst(raw_spec.copy())

raw_ground=absorbance(average_ground.copy())
#raw_ground=raw_ground.drop(columns=['I','bgd',"I0"]).copy()
raw_ground.dropna(axis=0, inplace=True)
avg_ground=baselinefit_cst(raw_ground.copy())


onlysmoothed_spec=raw_spec.copy()
onlysmoothed_spec.absor=sp.signal.savgol_filter(x=raw_spec.absor.copy(),
                                                 window_length=51,
                                                 polyorder=3) 

onlysmoothed_ground=raw_ground.copy()
onlysmoothed_ground.absor=sp.signal.savgol_filter(x=raw_ground.absor.copy(),
                                                  window_length=51,
                                                  polyorder=3) 



smoothed_spec=avg_spec.copy()
smoothed_spec.absor=sp.signal.savgol_filter(x=avg_spec.absor.copy(),
                                                 window_length=51,
                                                 polyorder=3) 



smoothed_ground=avg_ground.copy()
smoothed_ground.absor=sp.signal.savgol_filter(x=avg_ground.absor.copy(),
                                                  window_length=51,
                                                  polyorder=3) 

scaled_avg_spec=rescale_raw(avg_spec,smoothed_spec, 278,282)
scaled_avg_ground=rescale_raw(avg_ground,smoothed_ground, 278,282)

scaled_smoothed_spec=rescale_corrected(smoothed_spec,278,282)
scaled_smoothed_ground=rescale_corrected(smoothed_ground,278,282)

for nomfich in raw_lamp:
    if int(nomfich[2:4])%2==0 :
        break

#%%% plot averaged non scaled, non corrected raw spectra


#
## time-point
#centroids={}
scaling_top=500
baseline_blue=700
baseline_red=800

#
#peakpos = float(raw_ground.absor[raw_ground.wl.between(scaling_top-25,scaling_top+25)].astype('float').idxmax())
#half_max = (raw_ground.absor[peakpos] - raw_ground.absor[baseline_blue : baseline_red].mean()) / 3
#seg = np.where(raw_ground.absor > half_max + raw_ground.absor[baseline_blue : baseline_red].mean())
#n=0
## to the blue
#area_ground=[]
#while list(raw_ground.index).index(peakpos)+n in seg[0] and raw_ground.absor.iloc[list(raw_ground.index).index(peakpos)+n] > raw_ground.absor.iloc[list(raw_ground.index).index(peakpos)+n:list(raw_ground.index).index(peakpos)+n+25].min():
#    area_ground.append(raw_ground.wl.iloc[list(raw_ground.index).index(peakpos)+n])
#    n+=1
##to the red
#n = 1
#while list(raw_ground.index).index(peakpos)-n in seg[0] and raw_ground.absor.iloc[list(raw_ground.index).index(peakpos)-n] > raw_ground.absor.iloc[list(raw_ground.index).index(peakpos)-n-25:list(raw_ground.index).index(peakpos)-n].min():
#    area_ground.insert(0,raw_ground.wl.iloc[list(raw_ground.index).index(peakpos)-n])
#    n+=1
#def fctfit(x,amp1,mean1,stdev1,amp2,mean2,stdev2):
#    return gaussian(x, amp1,mean1,stdev1)+gaussian(x, amp2,mean2,stdev2)
#    
#
#x=np.array(raw_ground.wl[area_ground])
#y=np.array(raw_ground.absor[area_ground])
#
#initialParameters=[half_max,len(y)/2,300,half_max,len(y)/2,300]
#sigma=[1]*len(y)
#para, pcov = sp.optimize.curve_fit(f=fctfit, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
#
#
#
#
#plt.plot(raw_ground.wl,raw_ground.absor)
#plt.plot(np.linspace(area_ground[0],area_ground[-1]), gaussian(np.linspace(area_ground[0],area_ground[-1]), para[0], para[1],para[2]))
#plt.plot(np.linspace(area_ground[0],area_ground[-1]), gaussian(np.linspace(area_ground[0],area_ground[-1]), para[3], para[4],para[5]))
#plt.plot(np.linspace(area_ground[0],area_ground[-1]),gaussian(np.linspace(area_ground[0],area_ground[-1]), para[0], para[1],para[2])+gaussian(np.linspace(area_ground[0],area_ground[-1]), para[3], para[4],para[5]))
#plt.show()
#
#
#
#x=np.array(raw_spec.wl[area_ground])
#y=np.array(raw_spec.absor[area_ground])
#
#initialParameters=[half_max,len(y)/2,300,half_max,len(y)/2,300]
#sigma=[1]*len(y)
#para, pcov = sp.optimize.curve_fit(f=fctfit, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
#
#plt.plot(raw_spec.wl,raw_spec.absor)
#plt.plot(np.linspace(area_ground[0],area_ground[-1]), gaussian(np.linspace(area_ground[0],area_ground[-1]), para[0], para[1],para[2]))
#plt.plot(np.linspace(area_ground[0],area_ground[-1]), gaussian(np.linspace(area_ground[0],area_ground[-1]), para[3], para[4],para[5]))
#plt.plot(np.linspace(area_ground[0],area_ground[-1]),gaussian(np.linspace(area_ground[0],area_ground[-1]), para[0], para[1],para[2])+gaussian(np.linspace(area_ground[0],area_ground[-1]), para[3], para[4],para[5]))
#plt.show()
#
## fit one then 2 
#x=np.array(raw_ground.wl[area_ground])
#y=np.array(raw_ground.absor[area_ground])
#
#initialParameters=[half_max,500,50]
#sigma=[1]*len(y)
#parag, pcov = sp.optimize.curve_fit(f=gaussian, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
#
#plt.plot(raw_ground.wl,raw_ground.absor)
#plt.plot(np.linspace(area_ground[0],area_ground[-1]), gaussian(np.linspace(area_ground[0],area_ground[-1]), parag[0], parag[1],parag[2]))
#plt.show()
#
#def fctfit(x, ampg, amp, mean, stdev):
#    return gaussian(x,ampg, parag[1],parag[2]) + gaussian(x,amp, mean, stdev)
#
#x=np.array(raw_spec.wl[area_ground])
#y=np.array(raw_spec.absor[area_ground])
#
#initialParameters=[parag[0],half_max,480,50]
#sigma=[1]*len(y)
#para, pcov = sp.optimize.curve_fit(f=fctfit, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
#
#plt.plot(raw_spec.wl,raw_spec.absor)
#plt.plot(np.linspace(area_ground[0],area_ground[-1]), gaussian(np.linspace(area_ground[0],area_ground[-1]), para[0], parag[1],parag[2]))
#plt.plot(np.linspace(area_ground[0],area_ground[-1]), gaussian(np.linspace(area_ground[0],area_ground[-1]), para[1], para[2],para[3]))
#plt.plot(np.linspace(area_ground[0],area_ground[-1]),fctfit(np.linspace(area_ground[0],area_ground[-1]),para[0],para[1],para[2],para[3]))
#plt.show()
#



peakpos = float(raw_ground.absor[raw_ground.wl.between(scaling_top-25,scaling_top+25)].astype('float').idxmax())
half_max = (raw_ground.absor[peakpos] - raw_ground.absor[baseline_blue : baseline_red].mean()) / 1.7
seg = np.where(raw_ground.absor > half_max + raw_ground.absor[baseline_blue : baseline_red].mean())
n=0
# to the blue
area_ground=[]
while list(raw_ground.index).index(peakpos)+n in seg[0] and raw_ground.absor.iloc[list(raw_ground.index).index(peakpos)+n] > raw_ground.absor.iloc[list(raw_ground.index).index(peakpos)+n:list(raw_ground.index).index(peakpos)+n+25].min():
    area_ground.append(raw_ground.wl.iloc[list(raw_ground.index).index(peakpos)+n])
    n+=1
#to the red
n = 1
while list(raw_ground.index).index(peakpos)-n in seg[0] and raw_ground.absor.iloc[list(raw_ground.index).index(peakpos)-n] > raw_ground.absor.iloc[list(raw_ground.index).index(peakpos)-n-25:list(raw_ground.index).index(peakpos)-n].min():
    area_ground.insert(0,raw_ground.wl.iloc[list(raw_ground.index).index(peakpos)-n])
    n+=1
a=np.sum(np.array(raw_ground.wl[area_ground])*np.array(raw_ground.absor[area_ground])/np.sum(np.array(raw_ground.absor[area_ground])))
centroid_ground = a





# time-point
#centroids={}
scaling_top=500
baseline_blue=700
baseline_red=800


peakpos = float(raw_spec.absor[raw_spec.wl.between(scaling_top-25,scaling_top+25)].astype('float').idxmax())
half_max = (raw_spec.absor[peakpos] - raw_spec.absor[baseline_blue : baseline_red].mean()) / 1.7
seg = np.where(raw_spec.absor > half_max + raw_spec.absor[baseline_blue : baseline_red].mean())
n=0
# to the blue
area_tp=[]
while list(raw_spec.index).index(peakpos)+n in seg[0] and raw_spec.absor.iloc[list(raw_spec.index).index(peakpos)+n] > raw_spec.absor.iloc[list(raw_spec.index).index(peakpos)+n:list(raw_spec.index).index(peakpos)+n+25].min():
    area_tp.append(raw_spec.wl.iloc[list(raw_spec.index).index(peakpos)+n])
    n+=1
#to the red
n = 1
while list(raw_spec.index).index(peakpos)-n in seg[0] and raw_spec.absor.iloc[list(raw_spec.index).index(peakpos)-n] > raw_spec.absor.iloc[list(raw_spec.index).index(peakpos)-n-25:list(raw_spec.index).index(peakpos)-n].min():
    area_tp.insert(0,raw_spec.wl.iloc[list(raw_spec.index).index(peakpos)-n])
    n+=1
a=np.sum(np.array(raw_spec.wl[area_tp])*np.array(raw_spec.absor[area_tp])/np.sum(np.array(raw_spec.absor[area_tp])))
centroid_tp = a

if raw_spec.absor[raw_spec.wl.between(280,800)].min()<raw_ground.absor[raw_ground.wl.between(280,800)].min():
    globmin=raw_spec.absor[raw_spec.wl.between(280,800)].min()
else:
    globmin=raw_ground.absor[raw_ground.wl.between(280,800)].min()

if raw_spec.absor[raw_spec.wl.between(280,800)].max()>raw_ground.absor[raw_ground.wl.between(280,800)].max():
    globmax=raw_spec.absor[raw_spec.wl.between(280,800)].max()
else:
    globmax=raw_ground.absor[raw_ground.wl.between(280,800)].max()


# globmax=raw_spec.absor[raw_spec.wl.between(260,700)].max()
# globmin=raw_spec.absor[raw_spec.wl.between(260,700)].min()
fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [-]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.08, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='bright', n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 

ax.plot(raw_spec.wl,                  #x-axis is wavelength
        raw_spec.absor ,                   #y-axis is absor, or emission, or else
        linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
        label=nomfich[2:-4]+' ' + str(round(centroid_tp,1)) + ' nm',
        color=palette[0])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
ax.plot(raw_spec.wl[area_tp],                  #x-axis is wavelength
        raw_spec.absor[area_tp] ,                   #y-axis is absor, or emission, or else
        linewidth=1,
        color='cyan')               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect


ax.axvline(centroid_tp, color = palette[0], ls = '-.')



ax.plot(raw_ground.wl,                  #x-axis is wavelength
        raw_ground.absor ,                   #y-axis is absor, or emission, or else
        linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
        label='ground ' + str(round(centroid_ground,1)) + ' nm',
        color=palette[1])              
ax.plot(raw_ground.wl[area_ground],                  #x-axis is wavelength
        raw_ground.absor[area_ground] ,                   #y-axis is absor, or emission, or else
        linewidth=1, 
        color='red')   
ax.axvline(centroid_ground, color = palette[1], ls = '-.')



ax.set_title('averaged in crystallo absorbance spectra (uncorrected,unscaled)', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([220,800]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
# ax.yaxis.set_ticks(np.arange(round(globmin,1), round(globmax+0.1,1), 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)
legend = plt.legend(loc='upper right', shadow=True, prop={'size':11})
    
    
# plt.show()
figfilename = "averaged-unscaled-uncorrected_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "averaged-unscaled-uncorrected_spec.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "averaged-unscaled-uncorrected_spec.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
plt.close()



#%%% plot averaged non scaled, corrected avg spectra


# time-point
#centroids={}
scaling_top=500
baseline_blue=700
baseline_red=800


peakpos = float(avg_ground.absor[avg_ground.wl.between(scaling_top-25,scaling_top+25)].astype('float').idxmax())
half_max = (avg_ground.absor[peakpos] - avg_ground.absor[baseline_blue : baseline_red].mean()) / 1.7
seg = np.where(avg_ground.absor > half_max + avg_ground.absor[baseline_blue : baseline_red].mean())
n=0
# to the blue
area_ground=[]
while list(avg_ground.index).index(peakpos)+n in seg[0] and avg_ground.absor.iloc[list(avg_ground.index).index(peakpos)+n] > avg_ground.absor.iloc[list(avg_ground.index).index(peakpos)+n:list(avg_ground.index).index(peakpos)+n+25].min():
    area_ground.append(avg_ground.wl.iloc[list(avg_ground.index).index(peakpos)+n])
    n+=1
#to the red
n = 1
while list(avg_ground.index).index(peakpos)-n in seg[0] and avg_ground.absor.iloc[list(avg_ground.index).index(peakpos)-n] > avg_ground.absor.iloc[list(avg_ground.index).index(peakpos)-n-25:list(avg_ground.index).index(peakpos)-n].min():
    area_ground.insert(0,avg_ground.wl.iloc[list(avg_ground.index).index(peakpos)-n])
    n+=1
a=np.sum(np.array(avg_ground.wl[area_ground])*np.array(avg_ground.absor[area_ground])/np.sum(np.array(avg_ground.absor[area_ground])))
centroid_ground = a


# time-point
#centroids={}
scaling_top=500
baseline_blue=700
baseline_red=800


peakpos = float(avg_spec.absor[avg_spec.wl.between(scaling_top-25,scaling_top+25)].astype('float').idxmax())
half_max = (avg_spec.absor[peakpos] - avg_spec.absor[baseline_blue : baseline_red].mean()) / 1.7
seg = np.where(avg_spec.absor > half_max + avg_spec.absor[baseline_blue : baseline_red].mean())
n=0
# to the blue
area_tp=[]
while list(avg_spec.index).index(peakpos)+n in seg[0] and avg_spec.absor.iloc[list(avg_spec.index).index(peakpos)+n] > avg_spec.absor.iloc[list(avg_spec.index).index(peakpos)+n:list(avg_spec.index).index(peakpos)+n+25].min():
    area_tp.append(avg_spec.wl.iloc[list(avg_spec.index).index(peakpos)+n])
    n+=1
#to the red
n = 1
while list(avg_spec.index).index(peakpos)-n in seg[0] and avg_spec.absor.iloc[list(avg_spec.index).index(peakpos)-n] > avg_spec.absor.iloc[list(avg_spec.index).index(peakpos)-n-25:list(avg_spec.index).index(peakpos)-n].min():
    area_tp.insert(0,avg_spec.wl.iloc[list(avg_spec.index).index(peakpos)-n])
    n+=1
a=np.sum(np.array(avg_spec.wl[area_tp])*np.array(avg_spec.absor[area_tp])/np.sum(np.array(avg_spec.absor[area_tp])))
centroid_tp = a

if avg_spec.absor[avg_spec.wl.between(280,800)].min()<avg_ground.absor[avg_ground.wl.between(280,800)].min():
    globmin=avg_spec.absor[avg_spec.wl.between(280,800)].min()
else:
    globmin=avg_ground.absor[avg_ground.wl.between(280,800)].min()

if avg_spec.absor[avg_spec.wl.between(280,800)].max()>avg_ground.absor[avg_ground.wl.between(280,800)].max():
    globmax=avg_spec.absor[avg_spec.wl.between(280,800)].max()
else:
    globmax=avg_ground.absor[avg_ground.wl.between(280,800)].max()


# globmax=avg_spec.absor[avg_spec.wl.between(260,700)].max()
# globmin=avg_spec.absor[avg_spec.wl.between(260,700)].min()
fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [-]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.08, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='bright', n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 

ax.plot(avg_spec.wl,                  #x-axis is wavelength
        avg_spec.absor ,                   #y-axis is absor, or emission, or else
        linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
        label=nomfich[2:-4]+' ' + str(round(centroid_tp,1)) + ' nm',
        color=palette[0])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
ax.plot(avg_spec.wl[area_tp],                  #x-axis is wavelength
        avg_spec.absor[area_tp] ,                   #y-axis is absor, or emission, or else
        linewidth=1,
        color='cyan')               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect


ax.axvline(centroid_tp, color = palette[0], ls = '-.')



ax.plot(avg_ground.wl,                  #x-axis is wavelength
        avg_ground.absor ,                   #y-axis is absor, or emission, or else
        linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
        label='ground ' + str(round(centroid_ground,1)) + ' nm',
        color=palette[1])              
ax.plot(avg_ground.wl[area_ground],                  #x-axis is wavelength
        avg_ground.absor[area_ground] ,                   #y-axis is absor, or emission, or else
        linewidth=1, 
        color='red')   
ax.axvline(centroid_ground, color = palette[1], ls = '-.')



ax.set_title('averaged in crystallo absorbance spectra (uncorrected,unscaled)', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([220,800]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
# ax.yaxis.set_ticks(np.arange(round(globmin,1), round(globmax+0.1,1), 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)
legend = plt.legend(loc='upper right', shadow=True, prop={'size':11})
    
    
# plt.show()
figfilename = "averaged-unscaled_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "averaged-unscaled_spec.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "averaged-unscaled_spec.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
plt.close()



#%%% plot averaged non scaled, non corrected onlysmoothed spectra


# time-point
#centroids={}
scaling_top=500
baseline_blue=700
baseline_red=800


peakpos = float(onlysmoothed_ground.absor[onlysmoothed_ground.wl.between(scaling_top-25,scaling_top+25)].astype('float').idxmax())
half_max = (onlysmoothed_ground.absor[peakpos] - onlysmoothed_ground.absor[baseline_blue : baseline_red].mean()) / 1.7
seg = np.where(onlysmoothed_ground.absor > half_max + onlysmoothed_ground.absor[baseline_blue : baseline_red].mean())
n=0
# to the blue
area_ground=[]
while list(onlysmoothed_ground.index).index(peakpos)+n in seg[0] and onlysmoothed_ground.absor.iloc[list(onlysmoothed_ground.index).index(peakpos)+n] > onlysmoothed_ground.absor.iloc[list(onlysmoothed_ground.index).index(peakpos)+n:list(onlysmoothed_ground.index).index(peakpos)+n+25].min():
    area_ground.append(onlysmoothed_ground.wl.iloc[list(onlysmoothed_ground.index).index(peakpos)+n])
    n+=1
#to the red
n = 1
while list(onlysmoothed_ground.index).index(peakpos)-n in seg[0] and onlysmoothed_ground.absor.iloc[list(onlysmoothed_ground.index).index(peakpos)-n] > onlysmoothed_ground.absor.iloc[list(onlysmoothed_ground.index).index(peakpos)-n-25:list(onlysmoothed_ground.index).index(peakpos)-n].min():
    area_ground.insert(0,onlysmoothed_ground.wl.iloc[list(onlysmoothed_ground.index).index(peakpos)-n])
    n+=1
a=np.sum(np.array(onlysmoothed_ground.wl[area_ground])*np.array(onlysmoothed_ground.absor[area_ground])/np.sum(np.array(onlysmoothed_ground.absor[area_ground])))
centroid_ground = a


# time-point
#centroids={}
scaling_top=500
baseline_blue=700
baseline_red=800


peakpos = float(onlysmoothed_spec.absor[onlysmoothed_spec.wl.between(scaling_top-25,scaling_top+25)].astype('float').idxmax())
half_max = (onlysmoothed_spec.absor[peakpos] - onlysmoothed_spec.absor[baseline_blue : baseline_red].mean()) / 1.7
seg = np.where(onlysmoothed_spec.absor > half_max + onlysmoothed_spec.absor[baseline_blue : baseline_red].mean())
n=0
# to the blue
area_tp=[]
while list(onlysmoothed_spec.index).index(peakpos)+n in seg[0] and onlysmoothed_spec.absor.iloc[list(onlysmoothed_spec.index).index(peakpos)+n] > onlysmoothed_spec.absor.iloc[list(onlysmoothed_spec.index).index(peakpos)+n:list(onlysmoothed_spec.index).index(peakpos)+n+25].min():
    area_tp.append(onlysmoothed_spec.wl.iloc[list(onlysmoothed_spec.index).index(peakpos)+n])
    n+=1
#to the red
n = 1
while list(onlysmoothed_spec.index).index(peakpos)-n in seg[0] and onlysmoothed_spec.absor.iloc[list(onlysmoothed_spec.index).index(peakpos)-n] > onlysmoothed_spec.absor.iloc[list(onlysmoothed_spec.index).index(peakpos)-n-25:list(onlysmoothed_spec.index).index(peakpos)-n].min():
    area_tp.insert(0,onlysmoothed_spec.wl.iloc[list(onlysmoothed_spec.index).index(peakpos)-n])
    n+=1
a=np.sum(np.array(onlysmoothed_spec.wl[area_tp])*np.array(onlysmoothed_spec.absor[area_tp])/np.sum(np.array(onlysmoothed_spec.absor[area_tp])))
centroid_tp = a

if onlysmoothed_spec.absor[onlysmoothed_spec.wl.between(280,800)].min()<onlysmoothed_ground.absor[onlysmoothed_ground.wl.between(280,800)].min():
    globmin=onlysmoothed_spec.absor[onlysmoothed_spec.wl.between(280,800)].min()
else:
    globmin=onlysmoothed_ground.absor[onlysmoothed_ground.wl.between(280,800)].min()

if onlysmoothed_spec.absor[onlysmoothed_spec.wl.between(280,800)].max()>onlysmoothed_ground.absor[onlysmoothed_ground.wl.between(280,800)].max():
    globmax=onlysmoothed_spec.absor[onlysmoothed_spec.wl.between(280,800)].max()
else:
    globmax=onlysmoothed_ground.absor[onlysmoothed_ground.wl.between(280,800)].max()


# globmax=onlysmoothed_spec.absor[onlysmoothed_spec.wl.between(260,700)].max()
# globmin=onlysmoothed_spec.absor[onlysmoothed_spec.wl.between(260,700)].min()
fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [-]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.08, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='bright', n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 

ax.plot(onlysmoothed_spec.wl,                  #x-axis is wavelength
        onlysmoothed_spec.absor ,                   #y-axis is absor, or emission, or else
        linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
        label=nomfich[2:-4]+' ' + str(round(centroid_tp,1)) + ' nm',
        color=palette[0])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
ax.plot(onlysmoothed_spec.wl[area_tp],                  #x-axis is wavelength
        onlysmoothed_spec.absor[area_tp] ,                   #y-axis is absor, or emission, or else
        linewidth=1,
        color='cyan')               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect


ax.axvline(centroid_tp, color = palette[0], ls = '-.')



ax.plot(onlysmoothed_ground.wl,                  #x-axis is wavelength
        onlysmoothed_ground.absor ,                   #y-axis is absor, or emission, or else
        linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
        label='ground ' + str(round(centroid_ground,1)) + ' nm',
        color=palette[1])              
ax.plot(onlysmoothed_ground.wl[area_ground],                  #x-axis is wavelength
        onlysmoothed_ground.absor[area_ground] ,                   #y-axis is absor, or emission, or else
        linewidth=1, 
        color='red')   
ax.axvline(centroid_ground, color = palette[1], ls = '-.')



ax.set_title('averaged in crystallo absorbance spectra (uncorrected,unscaled)', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([220,800]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
# ax.yaxis.set_ticks(np.arange(round(globmin,1), round(globmax+0.1,1), 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)
legend = plt.legend(loc='upper right', shadow=True, prop={'size':11})
    
    
# plt.show()
figfilename = "smoothed-unscaled-uncorrected_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "smoothed-unscaled-uncorrected_spec.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "smoothed-unscaled-uncorrected_spec.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
plt.close()




#%%% plot averaged non scaled, corrected smoothed spectra


# time-point
#centroids={}
scaling_top=500
baseline_blue=700
baseline_red=800


peakpos = float(smoothed_ground.absor[smoothed_ground.wl.between(scaling_top-25,scaling_top+25)].astype('float').idxmax())
half_max = (smoothed_ground.absor[peakpos] - smoothed_ground.absor[baseline_blue : baseline_red].mean()) / 1.7
seg = np.where(smoothed_ground.absor > half_max + smoothed_ground.absor[baseline_blue : baseline_red].mean())
n=0
# to the blue
area_ground=[]
while list(smoothed_ground.index).index(peakpos)+n in seg[0] and smoothed_ground.absor.iloc[list(smoothed_ground.index).index(peakpos)+n] > smoothed_ground.absor.iloc[list(smoothed_ground.index).index(peakpos)+n:list(smoothed_ground.index).index(peakpos)+n+25].min():
    area_ground.append(smoothed_ground.wl.iloc[list(smoothed_ground.index).index(peakpos)+n])
    n+=1
#to the red
n = 1
while list(smoothed_ground.index).index(peakpos)-n in seg[0] and smoothed_ground.absor.iloc[list(smoothed_ground.index).index(peakpos)-n] > smoothed_ground.absor.iloc[list(smoothed_ground.index).index(peakpos)-n-25:list(smoothed_ground.index).index(peakpos)-n].min():
    area_ground.insert(0,smoothed_ground.wl.iloc[list(smoothed_ground.index).index(peakpos)-n])
    n+=1
a=np.sum(np.array(smoothed_ground.wl[area_ground])*np.array(smoothed_ground.absor[area_ground])/np.sum(np.array(smoothed_ground.absor[area_ground])))
centroid_ground = a


# time-point
#centroids={}
scaling_top=500
baseline_blue=700
baseline_red=800


peakpos = float(smoothed_spec.absor[smoothed_spec.wl.between(scaling_top-25,scaling_top+25)].astype('float').idxmax())
half_max = (smoothed_spec.absor[peakpos] - smoothed_spec.absor[baseline_blue : baseline_red].mean()) / 1.7
seg = np.where(smoothed_spec.absor > half_max + smoothed_spec.absor[baseline_blue : baseline_red].mean())
n=0
# to the blue
area_tp=[]
while list(smoothed_spec.index).index(peakpos)+n in seg[0] and smoothed_spec.absor.iloc[list(smoothed_spec.index).index(peakpos)+n] > smoothed_spec.absor.iloc[list(smoothed_spec.index).index(peakpos)+n:list(smoothed_spec.index).index(peakpos)+n+25].min():
    area_tp.append(smoothed_spec.wl.iloc[list(smoothed_spec.index).index(peakpos)+n])
    n+=1
#to the red
n = 1
while list(smoothed_spec.index).index(peakpos)-n in seg[0] and smoothed_spec.absor.iloc[list(smoothed_spec.index).index(peakpos)-n] > smoothed_spec.absor.iloc[list(smoothed_spec.index).index(peakpos)-n-25:list(smoothed_spec.index).index(peakpos)-n].min():
    area_tp.insert(0,smoothed_spec.wl.iloc[list(smoothed_spec.index).index(peakpos)-n])
    n+=1
a=np.sum(np.array(smoothed_spec.wl[area_tp])*np.array(smoothed_spec.absor[area_tp])/np.sum(np.array(smoothed_spec.absor[area_tp])))
centroid_tp = a

if smoothed_spec.absor[smoothed_spec.wl.between(280,800)].min()<smoothed_ground.absor[smoothed_ground.wl.between(280,800)].min():
    globmin=smoothed_spec.absor[smoothed_spec.wl.between(280,800)].min()
else:
    globmin=smoothed_ground.absor[smoothed_ground.wl.between(280,800)].min()

if smoothed_spec.absor[smoothed_spec.wl.between(280,800)].max()>smoothed_ground.absor[smoothed_ground.wl.between(280,800)].max():
    globmax=smoothed_spec.absor[smoothed_spec.wl.between(280,800)].max()
else:
    globmax=smoothed_ground.absor[smoothed_ground.wl.between(280,800)].max()


# globmax=smoothed_spec.absor[smoothed_spec.wl.between(260,700)].max()
# globmin=smoothed_spec.absor[smoothed_spec.wl.between(260,700)].min()
fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [-]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.08, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='bright', n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 

ax.plot(smoothed_spec.wl,                  #x-axis is wavelength
        smoothed_spec.absor ,                   #y-axis is absor, or emission, or else
        linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
        label=nomfich[2:-4]+' ' + str(round(centroid_tp,1)) + ' nm',
        color=palette[0])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
ax.plot(smoothed_spec.wl[area_tp],                  #x-axis is wavelength
        smoothed_spec.absor[area_tp] ,                   #y-axis is absor, or emission, or else
        linewidth=1,
        color='cyan')               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect


ax.axvline(centroid_tp, color = palette[0], ls = '-.')



ax.plot(smoothed_ground.wl,                  #x-axis is wavelength
        smoothed_ground.absor ,                   #y-axis is absor, or emission, or else
        linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
        label='ground ' + str(round(centroid_ground,1)) + ' nm',
        color=palette[1])              
ax.plot(smoothed_ground.wl[area_ground],                  #x-axis is wavelength
        smoothed_ground.absor[area_ground] ,                   #y-axis is absor, or emission, or else
        linewidth=1, 
        color='red')   
ax.axvline(centroid_ground, color = palette[1], ls = '-.')



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




#%%% plot averaged non scaled, non corrected scaled_avg spectra


# time-point
#centroids={}
scaling_top=500
baseline_blue=700
baseline_red=800


peakpos = float(scaled_avg_ground.absor[scaled_avg_ground.wl.between(scaling_top-25,scaling_top+25)].astype('float').idxmax())
half_max = (scaled_avg_ground.absor[peakpos] - scaled_avg_ground.absor[baseline_blue : baseline_red].mean()) / 1.7
seg = np.where(scaled_avg_ground.absor > half_max + scaled_avg_ground.absor[baseline_blue : baseline_red].mean())
n=0
# to the blue
area_ground=[]
while list(scaled_avg_ground.index).index(peakpos)+n in seg[0] and scaled_avg_ground.absor.iloc[list(scaled_avg_ground.index).index(peakpos)+n] > scaled_avg_ground.absor.iloc[list(scaled_avg_ground.index).index(peakpos)+n:list(scaled_avg_ground.index).index(peakpos)+n+25].min():
    area_ground.append(scaled_avg_ground.wl.iloc[list(scaled_avg_ground.index).index(peakpos)+n])
    n+=1
#to the red
n = 1
while list(scaled_avg_ground.index).index(peakpos)-n in seg[0] and scaled_avg_ground.absor.iloc[list(scaled_avg_ground.index).index(peakpos)-n] > scaled_avg_ground.absor.iloc[list(scaled_avg_ground.index).index(peakpos)-n-25:list(scaled_avg_ground.index).index(peakpos)-n].min():
    area_ground.insert(0,scaled_avg_ground.wl.iloc[list(scaled_avg_ground.index).index(peakpos)-n])
    n+=1
a=np.sum(np.array(scaled_avg_ground.wl[area_ground])*np.array(scaled_avg_ground.absor[area_ground])/np.sum(np.array(scaled_avg_ground.absor[area_ground])))
centroid_ground = a


# time-point
#centroids={}
scaling_top=500
baseline_blue=700
baseline_red=800


peakpos = float(scaled_avg_spec.absor[scaled_avg_spec.wl.between(scaling_top-25,scaling_top+25)].astype('float').idxmax())
half_max = (scaled_avg_spec.absor[peakpos] - scaled_avg_spec.absor[baseline_blue : baseline_red].mean()) / 1.7
seg = np.where(scaled_avg_spec.absor > half_max + scaled_avg_spec.absor[baseline_blue : baseline_red].mean())
n=0
# to the blue
area_tp=[]
while list(scaled_avg_spec.index).index(peakpos)+n in seg[0] and scaled_avg_spec.absor.iloc[list(scaled_avg_spec.index).index(peakpos)+n] > scaled_avg_spec.absor.iloc[list(scaled_avg_spec.index).index(peakpos)+n:list(scaled_avg_spec.index).index(peakpos)+n+25].min():
    area_tp.append(scaled_avg_spec.wl.iloc[list(scaled_avg_spec.index).index(peakpos)+n])
    n+=1
#to the red
n = 1
while list(scaled_avg_spec.index).index(peakpos)-n in seg[0] and scaled_avg_spec.absor.iloc[list(scaled_avg_spec.index).index(peakpos)-n] > scaled_avg_spec.absor.iloc[list(scaled_avg_spec.index).index(peakpos)-n-25:list(scaled_avg_spec.index).index(peakpos)-n].min():
    area_tp.insert(0,scaled_avg_spec.wl.iloc[list(scaled_avg_spec.index).index(peakpos)-n])
    n+=1
a=np.sum(np.array(scaled_avg_spec.wl[area_tp])*np.array(scaled_avg_spec.absor[area_tp])/np.sum(np.array(scaled_avg_spec.absor[area_tp])))
centroid_tp = a

if scaled_avg_spec.absor[scaled_avg_spec.wl.between(280,800)].min()<scaled_avg_ground.absor[scaled_avg_ground.wl.between(280,800)].min():
    globmin=scaled_avg_spec.absor[scaled_avg_spec.wl.between(280,800)].min()
else:
    globmin=scaled_avg_ground.absor[scaled_avg_ground.wl.between(280,800)].min()

if scaled_avg_spec.absor[scaled_avg_spec.wl.between(280,800)].max()>scaled_avg_ground.absor[scaled_avg_ground.wl.between(280,800)].max():
    globmax=scaled_avg_spec.absor[scaled_avg_spec.wl.between(280,800)].max()
else:
    globmax=scaled_avg_ground.absor[scaled_avg_ground.wl.between(280,800)].max()


# globmax=scaled_avg_spec.absor[scaled_avg_spec.wl.between(260,700)].max()
# globmin=scaled_avg_spec.absor[scaled_avg_spec.wl.between(260,700)].min()
fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [-]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.08, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='bright', n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 

ax.plot(scaled_avg_spec.wl,                  #x-axis is wavelength
        scaled_avg_spec.absor ,                   #y-axis is absor, or emission, or else
        linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
        label=nomfich[2:-4]+' ' + str(round(centroid_tp,1)) + ' nm',
        color=palette[0])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
ax.plot(scaled_avg_spec.wl[area_tp],                  #x-axis is wavelength
        scaled_avg_spec.absor[area_tp] ,                   #y-axis is absor, or emission, or else
        linewidth=1,
        color='cyan')               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect


ax.axvline(centroid_tp, color = palette[0], ls = '-.')



ax.plot(scaled_avg_ground.wl,                  #x-axis is wavelength
        scaled_avg_ground.absor ,                   #y-axis is absor, or emission, or else
        linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
        label='ground ' + str(round(centroid_ground,1)) + ' nm',
        color=palette[1])              
ax.plot(scaled_avg_ground.wl[area_ground],                  #x-axis is wavelength
        scaled_avg_ground.absor[area_ground] ,                   #y-axis is absor, or emission, or else
        linewidth=1, 
        color='red')   
ax.axvline(centroid_ground, color = palette[1], ls = '-.')



ax.set_title('averaged in crystallo absorbance spectra (corrected,scaled)', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([220,800]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
# ax.yaxis.set_ticks(np.arange(round(globmin,1), round(globmax+0.1,1), 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)
legend = plt.legend(loc='upper right', shadow=True, prop={'size':11})
    
    
# plt.show()
figfilename = "averaged-scaled_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "averaged-scaled_spec.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "averaged-scaled_spec.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
plt.close()




#%%% plot averaged scaled, corrected smoothed spectra


# time-point
#centroids={}
scaling_top=500
baseline_blue=700
baseline_red=800


peakpos = float(scaled_smoothed_ground.absor[scaled_smoothed_ground.wl.between(scaling_top-25,scaling_top+25)].astype('float').idxmax())
half_max = (scaled_smoothed_ground.absor[peakpos] - scaled_smoothed_ground.absor[baseline_blue : baseline_red].mean()) / 1.7
seg = np.where(scaled_smoothed_ground.absor > half_max + scaled_smoothed_ground.absor[baseline_blue : baseline_red].mean())
n=0
# to the blue
area_ground=[]
while list(scaled_smoothed_ground.index).index(peakpos)+n in seg[0] and scaled_smoothed_ground.absor.iloc[list(scaled_smoothed_ground.index).index(peakpos)+n] > scaled_smoothed_ground.absor.iloc[list(scaled_smoothed_ground.index).index(peakpos)+n:list(scaled_smoothed_ground.index).index(peakpos)+n+25].min():
    area_ground.append(scaled_smoothed_ground.wl.iloc[list(scaled_smoothed_ground.index).index(peakpos)+n])
    n+=1
#to the red
n = 1
while list(scaled_smoothed_ground.index).index(peakpos)-n in seg[0] and scaled_smoothed_ground.absor.iloc[list(scaled_smoothed_ground.index).index(peakpos)-n] > scaled_smoothed_ground.absor.iloc[list(scaled_smoothed_ground.index).index(peakpos)-n-25:list(scaled_smoothed_ground.index).index(peakpos)-n].min():
    area_ground.insert(0,scaled_smoothed_ground.wl.iloc[list(scaled_smoothed_ground.index).index(peakpos)-n])
    n+=1
a=np.sum(np.array(scaled_smoothed_ground.wl[area_ground])*np.array(scaled_smoothed_ground.absor[area_ground])/np.sum(np.array(scaled_smoothed_ground.absor[area_ground])))
centroid_ground = a


# time-point
#centroids={}
scaling_top=500
baseline_blue=700
baseline_red=800


peakpos = float(scaled_smoothed_spec.absor[scaled_smoothed_spec.wl.between(scaling_top-25,scaling_top+25)].astype('float').idxmax())
half_max = (scaled_smoothed_spec.absor[peakpos] - scaled_smoothed_spec.absor[baseline_blue : baseline_red].mean()) / 1.7
seg = np.where(scaled_smoothed_spec.absor > half_max + scaled_smoothed_spec.absor[baseline_blue : baseline_red].mean())
n=0
# to the blue
area_tp=[]
while list(scaled_smoothed_spec.index).index(peakpos)+n in seg[0] and scaled_smoothed_spec.absor.iloc[list(scaled_smoothed_spec.index).index(peakpos)+n] > scaled_smoothed_spec.absor.iloc[list(scaled_smoothed_spec.index).index(peakpos)+n:list(scaled_smoothed_spec.index).index(peakpos)+n+25].min():
    area_tp.append(scaled_smoothed_spec.wl.iloc[list(scaled_smoothed_spec.index).index(peakpos)+n])
    n+=1
#to the red
n = 1
while list(scaled_smoothed_spec.index).index(peakpos)-n in seg[0] and scaled_smoothed_spec.absor.iloc[list(scaled_smoothed_spec.index).index(peakpos)-n] > scaled_smoothed_spec.absor.iloc[list(scaled_smoothed_spec.index).index(peakpos)-n-25:list(scaled_smoothed_spec.index).index(peakpos)-n].min():
    area_tp.insert(0,scaled_smoothed_spec.wl.iloc[list(scaled_smoothed_spec.index).index(peakpos)-n])
    n+=1
a=np.sum(np.array(scaled_smoothed_spec.wl[area_tp])*np.array(scaled_smoothed_spec.absor[area_tp])/np.sum(np.array(scaled_smoothed_spec.absor[area_tp])))
centroid_tp = a

if scaled_smoothed_spec.absor[scaled_smoothed_spec.wl.between(280,800)].min()<scaled_smoothed_ground.absor[scaled_smoothed_ground.wl.between(280,800)].min():
    globmin=scaled_smoothed_spec.absor[scaled_smoothed_spec.wl.between(280,800)].min()
else:
    globmin=scaled_smoothed_ground.absor[scaled_smoothed_ground.wl.between(280,800)].min()

if scaled_smoothed_spec.absor[scaled_smoothed_spec.wl.between(280,800)].max()>scaled_smoothed_ground.absor[scaled_smoothed_ground.wl.between(280,800)].max():
    globmax=scaled_smoothed_spec.absor[scaled_smoothed_spec.wl.between(280,800)].max()
else:
    globmax=scaled_smoothed_ground.absor[scaled_smoothed_ground.wl.between(280,800)].max()


# globmax=scaled_smoothed_spec.absor[scaled_smoothed_spec.wl.between(260,700)].max()
# globmin=scaled_smoothed_spec.absor[scaled_smoothed_spec.wl.between(260,700)].min()
fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [-]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.08, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='bright', n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 

ax.plot(scaled_smoothed_spec.wl,                  #x-axis is wavelength
        scaled_smoothed_spec.absor ,                   #y-axis is absor, or emission, or else
        linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
        label=nomfich[2:-4]+' ' + str(round(centroid_tp,1)) + ' nm',
        color=palette[0])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
ax.plot(scaled_smoothed_spec.wl[area_tp],                  #x-axis is wavelength
        scaled_smoothed_spec.absor[area_tp] ,                   #y-axis is absor, or emission, or else
        linewidth=1,
        color='cyan')               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect


ax.axvline(centroid_tp, color = palette[0], ls = '-.')



ax.plot(scaled_smoothed_ground.wl,                  #x-axis is wavelength
        scaled_smoothed_ground.absor ,                   #y-axis is absor, or emission, or else
        linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
        label='ground ' + str(round(centroid_ground,1)) + ' nm',
        color=palette[1])              
ax.plot(scaled_smoothed_ground.wl[area_ground],                  #x-axis is wavelength
        scaled_smoothed_ground.absor[area_ground] ,                   #y-axis is absor, or emission, or else
        linewidth=1, 
        color='red')   
ax.axvline(centroid_ground, color = palette[1], ls = '-.')



ax.set_title('smoothed in crystallo absorbance spectra (corrected,scaled)', fontsize=10, fontweight='bold')  #This sets the title of the plot
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


avg_spectras=pd.merge(left=avg_ground,right=avg_spec,on='wl', how='inner')
avg_spectras.rename( columns = {'wl':'wl', 'absor_x':'abs_ground', 'absor_y':'abs_light'}, inplace = True)
avg_spectras.to_csv('averaged_raw_' + nomfich[5:-4] + '.csv', index=False)

smoothed_spectras=pd.merge(left=smoothed_ground,right=smoothed_spec,on='wl', how='inner')
smoothed_spectras.rename( columns = {'wl':'wl', 'absor_x':'abs_ground', 'absor_y':'abs_light'}, inplace = True)
smoothed_spec.to_csv('averaged_smoothed_' + nomfich[5:-4] + '.csv',index=False)



scaled_avg_spectras=pd.merge(left=scaled_avg_ground,right=scaled_avg_spec,on='wl', how='inner')
scaled_avg_spectras.rename( columns = {'wl':'wl', 'absor_x':'abs_ground', 'absor_y':'abs_light'}, inplace = True)
scaled_avg_spectras.to_csv('scaled_averaged_raw_' + nomfich[5:-4] + '.csv', index=False)

scaled_smoothed_spectras=pd.merge(left=scaled_smoothed_ground,right=scaled_smoothed_spec,on='wl', how='inner')
scaled_smoothed_spectras.rename( columns = {'wl':'wl', 'absor_x':'abs_ground', 'absor_y':'abs_light'}, inplace = True)
scaled_smoothed_spectras.to_csv('scaled_averaged_smoothed_' + nomfich[5:-4] + '.csv', index=False)



