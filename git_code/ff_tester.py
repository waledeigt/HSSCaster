#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:41:23 2024

@author: weigtd1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:12:52 2023

@author: weigtd1
"""

import json

#from astropy.wcs import WCS,utils
import numpy as np
from astropy.time import Time                   #convert between different time coordinates
import sunpy
import ff_funcs as ptools
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt


#%%
#Reading in json files - AR11130 as a test. FFjson = json file used to create the FFs dircetory

#jfile = open('/Users/weigtd1/Desktop/Aalto_2023_2025/Fmodes_proj/predictor/code/light_curves/f_mode_analysisCR_test1.json')
jfile = open('/Users/weigtd1/Desktop/Aalto_2023_2025/Fmodes_proj/predictor/code/light_curves/f_mode_analysisAR11130.json')
indata = json.load(jfile)
jfile.close()

ffjson = open('/Users/weigtd1/Desktop/Aalto_2023_2025/Fmodes_proj/predictor/code/light_curves/f_mode_analysisCR_test1.json')
ffindata = json.load(ffjson)
ffjson.close()


#%%
#Sorting out data files needed
plot_path = indata['plot_path'] # date_0923new for res file =050
data_path = indata['data_path']
ff_path = '/Volumes/SP PHD U3/CR_FF/CR20142015_hp_v4/'

fits_files, mt_dates, lonvec, latvec, crs, k_range, nu_range = ptools.knu_reader(data_path)

obs_dates = mt_dates
_, ffdates = ptools.date_range(fits_files, obs_dates, ffindata['sdate'], ffindata['edate'])

print('DEBUGGING')
print(f'Number of data files: {len(fits_files)}')
sdate = indata['sdate']
edate = indata['edate']

#%%
#COMMENT OUT IF NOT NEEDED
#Reading in quiet k-nu diagram file
"""Reading in K-nu analysis"""

res_file = indata['resolution']
year = indata['year']
ik0 = 0
rk = []
rnu = []
rbkg = []
for indx in range(0,6):
    ridge_data =  ptools.read_peak_list("peaks_vs_k.csv")
    k = ridge_data[:,0]
    nu = (ridge_data[:, indx+2])
    
    nans, xx= ptools.nan_helper(nu)
    nu[nans]= np.interp(xx(nans), xx(~nans), nu[~nans])
    
    xnew = np.linspace(k[nans == False].min(), k[nans==False].max(), 600) 
    
    spl = make_interp_spline(k, nu, k=1)  # type: BSpline
    power_smooth = spl(xnew)
    
    rk.append(xnew)
    rnu.append(power_smooth)
#%%
#Selecting fits file for date range
carr_rot = indata['carr_rot']
emerge_rot = indata['emerge_rot']
first_em = indata['first_emerge']

ycar, tcar = carr_rot[:10], carr_rot[-5:]
yem, tem = emerge_rot[:10], emerge_rot[-5:]
yfem, tfem = first_em[:10], first_em[-5:]
car_rot_indx = np.where(mt_dates == Time(ycar + 'T' + tcar , format='isot'))[0] #for AR11124 (test)
emerge_indx = np.where(mt_dates == Time(yem[:10] + 'T' + tem[-5:] , format='isot'))[0] #for AR11124 (test)
fem_indx = np.where(mt_dates == Time(yfem + 'T' + tfem , format='isot'))[0] #for AR11124 (test)
#print('XXXXXXXXXX')
#print(f'{mt_dates}')
#print(f'{car_rot_indx} corresponds to {carr_rot}')
t0 = mt_dates[car_rot_indx[0]-12].datetime
lon0 = indata['lon0']
glat = indata['glat']
sig_lvl = indata['sigma_lvl']
frk1, frk2 = indata['kx_bound1'], indata['kx_bound2']
fmod_q1, fmod_q2 = indata['fmod_k1'], indata['fmod_k2']

#%%
# Solar rotation calulation for tracking regions, for specific latitude
"""Solar rotation model + longitude range"""
_,_,_,_,_,sol_rot = ptools.SU_sol_rot(slat=glat)
lon_range = np.arange(lon0,-1*lon0,sol_rot)
ffiles = fits_files[car_rot_indx[0]-12:(car_rot_indx[0] + len(lon_range))]
mt_dates = mt_dates[car_rot_indx[0]-12:(car_rot_indx[0] + len(lon_range))]
#%%
"""Calculating observer angles"""
# Comment out if not needed!
bdates = [ii.datetime for ii in mt_dates]
b0 = np.array([sunpy.coordinates.sun.B0(time=ii).deg for ii in bdates])
l0 = np.array([sunpy.coordinates.sun.L0(time=ii).deg for ii in bdates])
p0 = np.array([sunpy.coordinates.sun.P(time=ii).deg for ii in bdates])
#ang_r = np.array(sunpy.coordinates.sun.angular_radius(bdates))/3600

#%%
"""Basic f-mode model""" # omega**2 = g*k_h where g = 274 m/s**2, k_h = sqrt(l(l+1))/R_sun
# Creating f-mode mask = 'grid' variable'
kmin,kmax,numin,numax = min(k_range), max(k_range), min(nu_range), max(nu_range)
ext = [kmin,kmax,numin,numax]

ffig, grid, f_p = ptools.fmodel(rk, rnu, k_range, nu_range,frk1, frk2,q1=fmod_q1, q2=fmod_q2)


#%%
"""Knu-maps"""

ik0=0
#CRFF_av - interpolation of closest FFs to selected point
av_maps, dd_indx, whts, lon_pt, lat_p = ptools.CRFF_av(data_path, ff_path, lon_range, glat, lonvec, latvec, ffiles, bdates, ffdates)
#FF_deproj - generates normalised k-nu diagram from av_maps above
trackmap, bmap, mindx = ptools.FF_deproj(data_path, av_maps, dd_indx, whts, lon_range, glat, lonvec, latvec, ik0, ffiles, bdates, ffdates)

fpow = []
fav = []
fdev = []
fstd = []

ddates = mt_dates[car_rot_indx[0]-12:(car_rot_indx[0] + len(lon_range))]
fdates = [mt_dates[ix].datetime for ix in mindx]

#Works out median f-mode power from mask and associated devoation from median (MAD)
for ii in [trackmap[ix] for ix in mindx]:
    nuflat = np.ravel(ii)[grid]
    fpow.append(nuflat)
    fmed = np.nanmedian(nuflat)
    fav.append(np.nanmedian(nuflat))
    fstd.append(np.nanstd(nuflat))
    fmad = np.abs(nuflat - fmed)
    #rms =np.sum(np.abs(nuflat - fmu)**2)/len(nuflat)
    fdev.append(np.median(fmad))

#%%
# Plots line plot and associated text file used for data used
lpow_range = [lon_range[ix] for ix in mindx]
true_emerge = Time(yfem + 'T' + tfem , format='isot')
fmodefig, ax, tax = ptools.fmode_pow(fav, fdev, lpow_range, fdates, emerge_rot,true_emerge)
#fmodefig.savefig(plot_path + f'/av_fmode_power_ratio_MAD_{lon0}_{glat}_v6.png')


#np.savetxt(plot_path + f'/{res_file}res_AR11130_lat{glat}_fmode_params_v6.txt', np.c_[fav,fdev,lpow_range,fdates],
#           delimiter=',', header="Med_fpow,MAD,sol_lon,obsdate", fmt='%s')