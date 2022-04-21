# -*- coding: utf-8 -*-
#!/usr/bin/env python
import datetime as dt
#from Scientific.IO.NetCDF import *
import numpy as np
import os, sys
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata
import time
import glob
#import pygrib
import math
#from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
#from matplotlib.mlab import griddata
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl


def main():


    ilist = ['peligro.csv']
    tlist = ['PELIGRO POR SEQUIA METEOROLOGICA']

    try:
        fdate  = open('Fecha.txt','r')
        mess = (fdate.readline()).split()[0]
        fdate.seek(0)
        anno = (fdate.readline()).split()[-1]
        mes = mess+" del "+anno
        
    except:
#        continue
        mes="Marzo del 2017"

    for i in range(len(ilist)):

        spifile = np.loadtxt(ilist[i],delimiter=',',dtype='float32')

        var   =  spifile[:,2]
        elat2 =  spifile[:,1]
        elon2 =  spifile[:,0]

#---

        plotspi(var,elat2,elon2,tlist[i],mes)


#    spifile = np.loadtxt(ilist[0],delimiter=',')

#    var   =  spifile[:,2]
#    elat2 =  spifile[:,1]
#    elon2 =  spifile[:,0]

##---

#    plotspi(var,elat2,elon2,tlist[0],mes)


def plotspi(outspi,ilats,ilons,tlist,mes):

    out,lats,lons = vinterp(outspi,ilats,ilons)
    newout = checkmask(out)

    fig = plt.figure(1,figsize=(16.00, 6.00), dpi=300)
    ax  = fig.add_subplot(111)
    #fig.suptitle("Mapa de peligro", fontsize=14, fontweight='bold')
#    plt.title("Mapa de "+tlist+" correspondiente al mes de \n"+mes, fontsize=14, fontweight='bold')

    m = Basemap(resolution='h',llcrnrlon=-85.2, llcrnrlat=19.6, urcrnrlon=-73.9, urcrnrlat=23.4, projection='lcc', lat_0=22., lon_0=-79., area_thresh=50.)
    xi, yi = m(lons, lats)

    m.drawmeridians(range(0, 360, 1),labels=[1,0,0,1],fontsize=10, color=(1,1,1,1), linewidth=0)
    m.drawparallels(range(-180, 180, 1),labels=[1,0,0,1],fontsize=10, color=(1,1,1,1), linewidth=0)
    m.drawmapscale(-82.8, 20.0, -79, 22., 200, barstyle='fancy')

    m.drawmapboundary(fill_color='#ffffff')
    #m.fillcontinents(color='#dfdcd8',lake_color='#98acc0')

    colors = ('#ffffff','#ffff73','#ffaa00','#e64c00','#732600')
    clevs= [-1.,0.,1.,2,3.,4.]
    cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors, N=9, gamma=1.0)

    cs = m.contourf(xi,yi,newout,clevs,cmap=cmap1,interpolation='gaussian', extend='both')

    cs.cmap.set_under('#ffffff')
    cs.cmap.set_over('#732600')
    #cb = m.colorbar(cs, location='right', label="",pad="10%")

    m.readshapefile('shp/reproj_muni', 'MUNICIPIO', drawbounds = False, linewidth=0.1)

    for info, shape in zip(m.MUNICIPIO_info, m.MUNICIPIO):
        x, y = zip(*shape) 
        m.plot(x, y, marker=None, color=(80/255,80/255,80/255), linewidth=0.1)

    m.readshapefile('shp/reproj_prov', 'PROVINCIA', drawbounds = False, linewidth=0.2)

    for info, shape in zip(m.PROVINCIA_info, m.PROVINCIA):
        x, y = zip(*shape) 
        m.plot(x, y, marker=None, color='k', linewidth=0.4)

#    ax2 = fig.add_axes([0.925, 0.15, 0.02, 0.69])
#    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, spacing='uniform', ticks=clevs, boundaries=clevs, format='%2.1f')

    ax2 = fig.add_axes([0.17, 0.32, 0.25, 0.06])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, spacing='uniform', ticks=clevs, boundaries=clevs, format='%2.1f', orientation="horizontal")

    cb.set_ticklabels(['','','','',''], update_ticks=True)
#    cb.set_label('Valores del indice', rotation=270, labelpad=20)

    plt.text(0.80, 0.88,"SISTEMA NACIONAL DE VIGILANCIA DE LA SEQUIA\nCENTRO NACIONAL DEL CLIMA\nINSTITUTO DE METEOROLOGIA DE CUBA\n", ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

    plt.text(0.80, 0.77,tlist[:]+"\n"+mes+"\nNorma: 1971-2000", ha='center', va='center', fontsize=11, transform=ax.transAxes)

    plt.text(0.215, 0.296,"S/P              D              M              S              E", ha='center', va='center', fontsize=11, transform=ax.transAxes)

    plt.text(0.22, 0.18,"S/P: SIN PELIGRO  D: DEBIL  M: MODERADO  S: SEVERO  E: EXTREMO", ha='center', va='center', fontsize=10, transform=ax.transAxes)


    ###plt.savefig(tlist.split()[0]+"_CUBA.png")
    plt.savefig(tlist.split()[0]+"_CUBA.png", pad_inches=0)
    plt.clf()
    plt.cla()



def vinterp(var,lat,lon):

    lats = lat
    lons = lon

#    y = np.arange(np.min(lats),np.max(lats),0.0360036)
#    x = np.arange(np.min(lons),np.max(lons),0.0360036)
    y = np.arange(np.min(lats),np.max(lats),0.0060006)
    x = np.arange(np.min(lons),np.max(lons),0.0060006)

#    XX, YY = np.meshgrid(x, y)

    XX = np.loadtxt("XX.txt",delimiter=',')
    YY = np.loadtxt("YY.txt",delimiter=',')

    points = np.empty(shape=(len(lats),2))
    points[:,0] = lons
    points[:,1] = lats

    out = griddata(points, var, (XX, YY), method = 'linear')

#    np.savetxt("XX.txt",XX,fmt='%3.2f',delimiter=',')
#    np.savetxt("YY.txt",YY,fmt='%3.2f',delimiter=',')
#    np.savetxt("rain.txt",out,fmt='%3.2f',delimiter=',')


    return out,YY,XX



def checkmask(out):

    mask = np.loadtxt("mask.txt")[-1::-1,:]
    newout = np.zeros(shape=out.shape)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):

            if int(mask[i,j]) == 0:
                newout[i,j] = np.nan
            else:
                newout[i,j] = out[i,j]

    return newout


if __name__=='__main__':
    main()
