#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:12:52 2023

@author: weigtd1
"""
import sys
import os
#from fmode_areas import read_peak_list
import matplotlib 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, Normalize, TwoSlopeNorm
from matplotlib.path import Path
import matplotlib.patches as patches

from astropy.io import fits
from astropy.time import Time                   #convert between different time coordinates
# from astropy.time import TimeDelta              #add/subtract time intervals 
# from scipy.signal import find_peaks
from scipy.interpolate import make_interp_spline, BSpline
from scipy import special

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map
from sunpy.coordinates import frames, get_horizons_coord, sun

sys.path.append("./utils")

from sklearn.neighbors import NearestNeighbors

#%%
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]



def read_peak_list(file='peaks_vs_k.csv'):
	line = []
	pmax = 0
	pf = open(file,'r')
	lines = pf.readlines()
	for tline in lines:
		if len(tline.strip())==0:
			continue
		if tline.strip()[0]=='#':
			continue
		tsplit = np.float32(tline.split(','))
		pmax = np.max([pmax,len(tsplit)])
		line.append(tsplit)
	peak_array = np.zeros([len(line), pmax]) + np.nan
	pf.close()
	for il,l in enumerate(line):
		peak_array[il,:len(l)]=l
	return(peak_array)


#%%
def date_range(fits_files,mt_dates, date_start, date_end):
    start = Time(date_start[0:10] + 'T' + date_start[-5:], format='isot')
    end = Time(date_end[0:10] + 'T' + date_end[-5:], format='isot')

    return([x for x in fits_files if x<=date_end and x>=date_start], 
           [xx for xx in mt_dates if xx < end and xx>=start])
#%%

def knu_reader(data_path):

    fits_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    fits_files = [j for j in fits_files if j.endswith('FD_k-nu.fits')]#[0:100]
    #fits_files = [ff for ff in fits_files if os.path.getsize(data_path +'/'+ ff) >= 536270400]
    #fits_files = [x for x in fits_files if x<=date_start and x>=date_end]
    
    
    mt_dates = []
    count = 0
    crs = []
    for ii in fits_files:
        f=fits.open(data_path + f'/{ii}')
        f[0].header
        
        date = f[0].header['MT_DATE']
        time = f[0].header['MT_TIME']
        cr_code = f[0].header['MT_CODE']
        crs.append(cr_code)
        
        obs_date = Time(date + 'T' + time, format='isot')
        mt_dates.append(obs_date)
        
        
        count = count+1
        if count == 1:
            # compute lon/lat vector from header data
            lon0,lon1 = f[0].header['LON0'],f[0].header[('LON1')]
            lat0,lat1 = f[0].header['LAT0'],f[0].header[('LAT1')]
            gs=f[0].header['GRIDSTEP']
            ps=f[0].header['PATCH_SZ']
            lonvec=np.arange(f[0].header['NAXIS4']) *gs +lon0+ps/2
            latvec=np.arange(f[0].header['NAXIS5']) *gs +lat0+ps/2
            cp = f[0].header["CALPAR*"]
            dmap = f[0].data[0,0,0,:,:]
            kmin,kmax = f[0].header["K_MIN"],f[0].header["K_MAX"]
            numin,numax = f[0].header["NU_MIN"],f[0].header["NU_MAX"]
            # print(len(dmap))
            # print(len(dmap[0]))
            k_range = np.linspace(kmin, kmax, num=len(dmap))
            nu_range = np.linspace(numin, numax, num=len(dmap[0]))
            f.close()
            
        else:
            f.close()
    
    return(sorted(fits_files), sorted(mt_dates),lonvec, latvec, crs, k_range, nu_range)

    #return(sorted(fits_files), sorted(mt_dates),lonvec[0], latvec[0], crs, k_range, nu_range)
        
#%%
from scipy.ndimage import uniform_filter1d

def rolling_mean_along_axis(a, W, axis=-1):
    # a : Input ndarray
    # W : Window size
    # axis : Axis along which we will apply rolling/sliding mean
    hW = W//2
    L = a.shape[axis]-W+1   
    indexer = [slice(None) for _ in range(a.ndim)]
    indexer[axis] = slice(hW,hW+L)
    return uniform_filter1d(a,W,axis=axis)[tuple(indexer)]

#%%
def blockwise_average_ND(a, factors):    
    """
    `a` is the N-dim input array
    `factors` is the blocksize on which averaging is to be performed
    """

    factors = np.asanyarray(factors)
    sh = np.column_stack([a.shape//factors, factors]).ravel()
    b = a.reshape(sh).mean(tuple(range(1, 2*a.ndim, 2)))

    return b

#%%
"""Solar rotation - Snodgrass and Ulrich 1990 A&A"""

def SU_sol_rot(slat=np.linspace(-90,90,91), A=14.713, B=-2.396, C=-1.787):
    import numpy as np
    slat=np.deg2rad(slat)
    dA = 0.0491
    dB = 0.188
    dC = 0.253
    
    sup = (A+dA) + (B+dB)*(np.sin(slat))**2 + (C+dC)*(np.sin(slat))**4
    sm = A + B*(np.sin(slat))**2 + C*(np.sin(slat))**4
    slow = (A-dA) + (B-dB)*(np.sin(slat))**2 + (C-dC)*(np.sin(slat))**4
    
    # fig = plt.figure(figsize=(12,8))
    # plt.scatter(np.rad2deg(slat), sup, color='orange', label='S&U 1990 upper limit')
    # plt.scatter(np.rad2deg(slat), slow, color='k', label='S&U 1990 low limit')
    # plt.scatter(np.rad2deg(slat), sm, color='r', linestyle='--', label='S&U 1990 mean')
    # plt.legend()
    # plt.xlabel(r'Solar latitude ($\psi:^{\circ}$)')
    # plt.ylabel(r'Solar rotation ($^{\circ}$ day$^{-1}$)')
    # # plt.title(f"All HMI {ii} {crs[0]}/{crs[-1]} observations \n Average {cp[0]} plot: [lon = {grid_pnt[0][0]:.1f}, lat ={grid_pnt[0][1]:.1f}]")
    
    # fig = plt.figure(figsize=(12,8))
    # plt.scatter(np.rad2deg(slat), (sup/24)*4, color='orange', label='S&U 1990 upper limit')
    # plt.scatter(np.rad2deg(slat), (slow/24)*4, color='k', label='S&U 1990 low limit')
    # plt.scatter(np.rad2deg(slat), (sm/24)*4, color='r', linestyle='--', label='S&U 1990 mean')
    # plt.legend()
    # plt.xlabel(r'Solar latitude ($\psi:^{\circ}$)')
    # plt.ylabel(r'Solar rotation ($^{\circ}$ 4 hr$^{-1}$)')
    
    sup4 = (sup/24)*4
    slow4 = (slow/24)*4
    sm4 = (sm/24)*4
    return(sup, sup4, slow, slow4, sm, sm4)

#%%
def fmodel(rk, rnu, k_range, nu_range, frk1, frk2, q1=1.1, q2=0.75):
    
    
    R_sun = 6.96E8
    k_h = np.sqrt(rk[0]*(rk[0]+1))/R_sun

    nu_mod0 = np.sqrt(k_h*274)/(2*np.pi)*1000 #q=0
    nu_modq1 = np.sqrt(k_h*274*q1)/(2*np.pi)*1000
    nu_modq2 = np.sqrt(k_h*274*q2)/(2*np.pi)*1000

    frk = np.array(rk[0])

    fm_idx = np.where((frk>=frk1) & (frk<=frk2))[0]
    f_c1 = [i for i in zip(frk[fm_idx], nu_modq1[fm_idx])]
    f_c2 = sorted([j for j in zip(frk[fm_idx], nu_modq2[fm_idx])], reverse=True)
    f_c = f_c1 + f_c2

    f_c.insert(-1, (f_c1[0][0],f_c1[0][1]))


    f_p = Path(f_c)

    X, Y = np.meshgrid(k_range, nu_range)
    X, Y = blockwise_average_ND(X, [2,2]), blockwise_average_ND(Y, [2,2])
    #grid_pnts = np.column_stack((np.array(X),np.array(Y))).T
    X , Y = X.flatten(), Y.flatten()
    pts = np.column_stack((X,Y))
    grid = f_p.contains_points(pts)


    fig, ax = plt.subplots(figsize=(10,7.5))
    patch = patches.PathPatch(f_p, facecolor='yellow', lw=2, edgecolor='k',
                              alpha=0.25,label='f-mode region', hatch='xxx')


    ax.plot(rk[0],nu_mod0, label=r'k$_{x}$', color='k')
    ax.plot(rk[0],nu_modq1, label=f'{q1}'+r'k$_{x}$', color='k', linestyle='--')
    ax.plot(rk[0],nu_modq2, label=f'{q2}'+r'k$_{x}$', color='k', linestyle='-.')
    ax.plot(rk[0], rnu[0], color='red', label='Data')
    #ax.set_xlim(500,2500)
    ax.add_artist(patch)
    ax.set_ylim(1,7)
    ax.set_xlabel('k')
    ax.set_ylabel(r'$\nu$')
    ax.legend(ncols=2, fontsize=12)
    ax.set_title('Basic f-mode model')
    
    return(fig, grid, f_p)
#%%
def fmode_pow(fav, ferr, llon, fin_dates, emerge_rot, true_emerge):
    from matplotlib.dates import DateFormatter
    
    # f25 = np.array(fpow25)
    # f75 = np.array(fpow75)
    fav = np.array(fav)
    fstd = np.array(ferr)
    # fmid = np.array(fmid)
    # fmin = np.array(fmin)
    # fmax = np.array(fmax)
    
    fdates = fin_dates
    
    ar_emerge =  Time(emerge_rot[:10] + 'T' + emerge_rot[-5:] , format='isot')
    #ar_indx = np.where(fdates == ar_emerge.datetime)
    fig = plt.figure(figsize=(15,7))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    
    ax1.plot(llon,fav, color='pink', linestyle='None')
    ax2.plot(fdates, fav,color='k', label='Mean Ratio') # Create a dummy plot
    ax2.axvline(ar_emerge.datetime, color='k', alpha=0.2)
    ax2.set_xlabel('Time')
    ax1.set_xlabel('Solar Longitude')
    ax1.set_ylabel('Median F-mode Ratio')
    
    ax2.plot(fdates,fav+fstd, color='k', linestyle='--',label=r'$\pm$ FF MAD', alpha=0.5)
    ax2.plot(fdates,fav-fstd, color='k', linestyle='--', alpha=0.5)
    ax1.axhline(1.0, color='red', linewidth = 0.5, linestyle = '--')
    ax2.axvline(true_emerge.datetime, color='k', linestyle='--', alpha=0.2)

    
    myFmt = DateFormatter(r"%d/%m")
    ax2.xaxis.set_major_formatter(myFmt)
    #fig.suptitle(f'Full averaged kx direction [45,75]', y=1.05)
    return(fig, ax1, ax2)
#%%
def CRFF_av(data_path,ff_path, lon_range, glat, lonvec, latvec, ffiles, mt_dates, ffdates):
    
    
    lng, ltg = np.meshgrid(lonvec, latvec)
    lng , ltg = lng.flatten(), ltg.flatten()
    points = np.column_stack((lng,ltg))
        
    av_maps = []
    dd_indx = []
    whts = []
    lon_pt = []
    lat_pt = []
    for findx, glon in enumerate(lon_range):
        data_files = data_path + f'{ffiles[findx]}'
        
        proj_points = SkyCoord(points[:,0]*u.deg, points[:,1]*u.deg, 
                                frame=frames.HeliographicStonyhurst, obstime=ffdates[len(ffdates)//2],
                                observer = "earth")
        proj_points = proj_points.transform_to(frames.Helioprojective)
        
        # dproj_points = SkyCoord(points[:,0]*u.deg, points[:,1]*u.deg, 
        #                         frame=frames.HeliographicStonyhurst, obstime=ffdates[len(ffdates)//2],
        #                         observer = "earth")
        # dproj_points = proj_points.transform_to(frames.Helioprojective)
        
        #sunr = dproj_points.distance.km/proj_points.distance.km
        # glon, glat = 24,  -12
        
        g_pnt = SkyCoord(glon*u.deg, glat*u.deg, 
                                frame=frames.HeliographicStonyhurst, obstime=mt_dates[findx],
                                observer = "earth")
        g_pnt = g_pnt.transform_to(frames.Helioprojective) # before obstime=mt_dates[findx]
        
        grid_pnt = [[g_pnt.Tx.arcsec, g_pnt.Ty.arcsec]]
        
        
        knn = NearestNeighbors(n_neighbors=4)
        knn.fit(np.column_stack((proj_points.Tx, proj_points.Ty)))
        dist, indx = knn.kneighbors(X=grid_pnt, return_distance=True)
        print(dist,indx)
        
        weights = []
        for ii in dist[0]:
            w = ii/np.nansum(dist[0])
            weights.append(w)
        
        weights = np.flip(weights)
        whts.append(weights)
        
        hr = data_files[-18:-13]
        # for ii in hrs:    
        #     alltt = [at[-18:-13] for at in data_files]
        dd = ff_path + '/' + hr + '/' 
            
    
        ff_files = [f for f in os.listdir(dd) if os.path.isfile(os.path.join(dd, f))]
        ffs = []
     
        for ix in indx[0]:
            print(f'{(proj_points.Tx.arcsec)[ix]:.3f}_{(proj_points.Ty.arcsec)[ix]:.3f} at {hr} maps')
            tx, ty = round(proj_points.Tx.arcsec[ix],3), round(proj_points.Ty.arcsec[ix],3)
            
            pf = [j[:-16].split("_") for j in ff_files]
            ppf = np.array(pf).astype(float)
            dtx = np.abs(tx - ppf[:,0])
            dty = np.abs(ty - ppf[:,1])
            inx = np.where((dtx <= 1))[0]
            iny = np.where((dty <= 1))[0]
            inxy = np.intersect1d(inx, iny)
            print(inx, iny, inxy)
            #final = [j for j in ff_files if j.startswith(f'{(proj_points.Tx.arcsec)[ix]:.3f}_{(proj_points.Ty.arcsec)[ix]:.3f}_')]
            final = ff_files[inxy[0]]
            ffs.append(np.loadtxt(dd + final))
            print(f'Loading {dd+final})')
        
        print(f'Finished for ({glon} lon, {glat} lat: {ffiles[findx]}')
        print('______________________________________________________')
        data = np.average(np.array(ffs), weights=weights, axis=0)
        av_maps.append(data)
        dd_indx.append(indx[0])
        lon_pt.append(glon)
        lat_pt.append(glat)

    return(av_maps, dd_indx, whts, lon_pt, lat_pt)#,rms)


#%%
def FF_deproj(data_path, av_maps, dd_indx, weights, lon_range, glat, lonvec, latvec,ik0, ffiles, mt_dates, ffdates):

    # av_maps = []
    # dd_indx = []
    # whts = []
    # lon_pt = []
    # lat_pt = []
    
    lng, ltg = np.meshgrid(lonvec, latvec)
    lng , ltg = lng.flatten(), ltg.flatten()
    points = np.column_stack((lng,ltg))
    lngr = np.deg2rad(lng)
    ltgr = np.deg2rad(ltg)
    

    
    
    mindx = []
    btrack = []
    trackmap = []
    for findx, glon in enumerate(lon_range):
    
        data_files = data_path + f'{ffiles[findx]}'
        # glon, glat = 24,  -12
        proj_points = SkyCoord(points[:,0]*u.deg, points[:,1]*u.deg, 
                                frame=frames.HeliographicStonyhurst, obstime=mt_dates[findx],
                                observer = "earth") #obstime=ffdates[len(ffdates)//2]
        proj_points = proj_points.transform_to(frames.Helioprojective)
        # glon, glat = 24,  -12
        
        g_pnt = SkyCoord(glon*u.deg, glat*u.deg, 
                                frame=frames.HeliographicStonyhurst, obstime=mt_dates[findx],
                                observer = "earth")
        g_pnt = g_pnt.transform_to(frames.Helioprojective)
        
        grid_pnt = [[g_pnt.Tx.arcsec, g_pnt.Ty.arcsec]]
        
        
        knn = NearestNeighbors(n_neighbors=4)
        knn.fit(np.column_stack((proj_points.Tx, proj_points.Ty)))
        dist, indx = knn.kneighbors(X=grid_pnt, return_distance=True)
        print(dist,indx)
        #print(dist,indx)
        
        #lat_ind = np.concatenate([np.where(ii.lat.degree == latvec)[0] for ii in points[indx[0]]\.transform_to(frames.HeliographicStonyhurst)])
        latdx = np.concatenate([np.where(round(ii.lat.degree, 3) == latvec)[0] for ii in proj_points.transform_to(frames.HeliographicStonyhurst)[indx[0]]])
        lindx = np.concatenate([np.where(round(ii.lon.degree, 3) == lonvec)[0] for ii in proj_points.transform_to(frames.HeliographicStonyhurst)[indx[0]]])
        #lat_ind = np.concatenate([np.where(ii == platvec)[0] for ii in proj_points[indx[0]][:,1]])
        #lat_ind = np.concatenate([np.where(ii == latvec)[0] for ii in points[indx[0]][:,1]])

        #lon_ind = []
        # for ii in lat_ind:
        #     ind = np.concatenate([np.where(jj == plonvec[ii])[0] for jj in proj_points[indx[0]][:,0]])
        #     lon_ind.append(np.round(np.mean(ind).astype(int)))
        #lon_ind = np.concatenate([np.where(ii == lonvec)[0] for ii in points[indx[0]][:,0]])

      
        """Generating maps"""
        #ik0 = 0        
        f=fits.open(data_path + f'/{ffiles[findx]}')
        fmaps = []
        bmaps = []
        for lat_ind,lon_ind in zip(latdx,lindx): 
            data_map = f[0].data[lat_ind ,lon_ind,ik0,:,:]
            bmap = f[1].data[latdx, lindx]
            fmaps.append(data_map)
            bmaps.append(bmap)
        f.close()
            
        print(len(weights[findx]))
        """Creating point map"""
        # for ii in np.array(fmaps):
        mm = np.array(fmaps)
        print(len(mm))
        av4 = np.average(mm, weights=weights[findx]
                        ,axis=0)
        av_bw = blockwise_average_ND(av4, [2,2])
        
        bb = np.array(bmaps)
        av_bb = np.average(bb, weights=weights[findx],
                            axis = 0)
        
        btrack.append(av_bb)
        #av_ind = (av_bw)*((av_bw > 0.8) & (av_bw < 1.3))
        #av_ind = (av_bw)*(av_bw > 0.1) # pixe; thresholding

        #av = np.where(av_ind==0, np.nan, av_ind)
        #av = av_ind
        av=av_bw
        hrs_indx = []
        hrs = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00']
        
        for ii in hrs:
            idx = [i for i, j in enumerate(ffiles[findx:findx+1]) if j.endswith(f'{ii}_FD_k-nu.fits')]
            hrs_indx.append(idx)
            
        #print(hrs_indx)
        
        # av_indx = np.array([ii for ii, jj in enumerate(hrs_indx) if len(jj) != 0])[0]
        # av_map = np.array(av_maps)
        #brms_arr = np.array(brms)
        
        #img = np.abs(av/av_map[findx][av_indx])
        img_bw = np.abs(av/blockwise_average_ND(av_maps[findx],[2,2]))
        #brms_tt = brms[findx][av_indx]

        # uo_maps.append(img)
        
        # hrs_indx_arr = np.concatenate(hrs_indx)
        # map_indx = np.argsort(hrs_indx_arr)
        # o_maps = [np.array(uo_maps)[jk,:,:] for jk in map_indx]
        #img_bw = blockwise_average_ND(trackmap[0], [5,5]
        print(np.nanmean(img_bw))
        if np.nanmean(img_bw) != np.nan and np.nanmean(img_bw) > 0.0 and np.nanmean(img_bw) < 1e10:
            trackmap.append(img_bw)
            # btrack.append(brms_tt)
            mindx.append(findx)
            print(f'finished for {glon}, file: {ffiles[findx]}')
        else:
            trackmap.append(img_bw)
            #btrack.append(brms_tt)
            print(f'No valid map for: {glon}, file: {ffiles[findx]}')
        print('____________________________________')
    return(trackmap, btrack, mindx)

