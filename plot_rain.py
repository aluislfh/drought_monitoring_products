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
from matplotlib.colors import ListedColormap
#import shapefile
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
import matplotlib as mpl


def main():


    ilist = ['Lluvia.csv']
    tlist = ['PRECIPITACION MENSUAL ACUMULADA']

    try:
        fdate  = open('Fecha.txt','r')
        mess = (fdate.readline()).split()[0]
        fdate.seek(0)
        anno = (fdate.readline()).split()[-1]
        mes = mess+" del "+anno
        
    except:
#        continue
        mes="Mayo del 2017"


    for i in range(len(ilist)):

        spifile = np.loadtxt(ilist[i],delimiter=',')

        var   =  spifile[:,2]
        elat2 =  spifile[:,1]
        elon2 =  spifile[:,0]

#---

        plotspi(var,elat2,elon2,tlist[i],mes)



#   Mapas provinciales

#    for i in range(len(ilist)):


#        prov = open("provincias_ok.txt",'r')

#        
#        idprov=0
#        for j in range(19):
#            idprov=int(idprov)
#            idprov+=1
#            name, lat1, lat2, lon1, lon2, idprov = (prov.readline()).split()
#            plotspi_prov(var,elat2,elon2,tlist[i],mes, lon1, lon2, lat1, lat2, name, idprov)

#        prov.close()


def splits(name):

    full=""

    for i in range(len(name.split())):
        full=full+name.split()[i]+"_"

    return full



def limits(mat):

    return np.min(mat)-0.2, np.max(mat)+0.2


def plotspi(outspi,ilats,ilons,tlist,mes):

    out,lats,lons = vinterp(outspi,ilats,ilons)
    newout = checkmask(out)


    fig = plt.figure(1,figsize=(16.00, 6.00), dpi=300)
    ax  = fig.add_subplot(111)
    #fig.suptitle("Mapa de peligro", fontsize=14, fontweight='bold')
#    plt.title("Mapa de "+tlist+" en el mes de \n"+mes, fontsize=14, fontweight='bold')

    m = Basemap(resolution='h',llcrnrlon=-85.2, llcrnrlat=19.6, urcrnrlon=-73.9, urcrnrlat=23.4, projection='lcc', lat_0=22., lon_0=-79., area_thresh=50.)
    xi, yi = m(lons, lats)

    m.drawmeridians(range(0, 360, 1),labels=[1,0,0,1],fontsize=10, color=(1,1,1,1), linewidth=0)
    m.drawparallels(range(-180, 180, 1),labels=[1,0,0,1],fontsize=10, color=(1,1,1,1), linewidth=0)
    m.drawmapscale(-82.8, 20.0, -79, 22., 200, barstyle='fancy')

    m.drawmapboundary(fill_color='#ffffff')
    #m.fillcontinents(color='#dfdcd8',lake_color='#98acc0')

#    colors = ('#732600','#ff6900','#ffaa00','#ffff73','#ffffff','#d3ffbe','#a3ff73','#38a800','#267300')
#    clevs= [-2.0,-1.5,-1.0,-0.5,0.5,1.0,1.5,2.0]
#    cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors, N=9, gamma=1.0)

    clevs, cmap1 = rainpalettle()

    cs = m.contourf(xi,yi,newout,clevs,cmap=cmap1,interpolation='gaussian', extend='both')

    cs.cmap.set_under('#ffaa00')
    cs.cmap.set_over('#267300')
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

    ax2 = fig.add_axes([0.15, 0.32, 0.32, 0.04])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, spacing='uniform', ticks=clevs, boundaries=clevs, format='%2.0f', orientation="horizontal")

    cb.set_label('VALORES DE PRECIPITACION ACUMULADA [mm/mes]', labelpad=-46)

    plt.text(0.80, 0.88,"SISTEMA NACIONAL DE VIGILANCIA DE LA SEQUIA\nCENTRO NACIONAL DEL CLIMA\nINSTITUTO DE METEOROLOGIA DE CUBA\n", ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

    plt.text(0.80, 0.77,tlist[:]+"\n"+mes+"\nNorma: 1971-2000", ha='center', va='center', fontsize=11, transform=ax.transAxes)

#    plt.text(0.22, 0.294,"S         M         D                 D         M         S", ha='center', va='center', fontsize=11, transform=ax.transAxes)

#    plt.text(0.22, 0.18,"S-E: SEVERO - EXTREMO  M: MODERADO  D: DEBIL  S/N: SIN DEFICIT", ha='center', va='center', fontsize=10, transform=ax.transAxes)


    plt.savefig(tlist.split()[0]+"_CUBA.png")
    plt.clf()
    plt.cla()



def plotspi_prov(outspi,ilats,ilons,tlist,mes,lon1,lon2,lat1,lat2,provname,idprov):

    out,lats,lons = vinterp(outspi,ilats,ilons)
    newout = checkmask(out)

    fig = plt.figure(1,figsize=(14.00, 8.5), dpi=300)
    ax  = fig.add_subplot(111)
    #fig.suptitle("Mapa de peligro", fontsize=14, fontweight='bold')
#    plt.title("Mapa de "+tlist+" en el mes de \n"+mes, fontsize=14, fontweight='bold')

    m = Basemap(resolution='h',llcrnrlon=float(lon1), llcrnrlat=float(lat1), urcrnrlon=float(lon2),
             urcrnrlat=float(lat2), projection='lcc', lat_0=float(lat1)+(np.abs(float(lat1)-float(lat2))/2), 
                lon_0=float(lon1)+(np.abs(float(lon1)-float(lon2))/2), area_thresh=50.)
    xi, yi = m(lons, lats)

    m.drawmeridians(range(0, 360, 1),labels=[1,0,0,1],fontsize=10, color=(1,1,1,1), linewidth=0)
    m.drawparallels(range(-180, 180, 1),labels=[1,0,0,1],fontsize=10, color=(1,1,1,1), linewidth=0)
#    m.drawmapscale(-82.8, 20.0, -79, 22., 200, barstyle='fancy')

    m.drawmapboundary(fill_color='#ffffff')
    #m.fillcontinents(color='#dfdcd8',lake_color='#98acc0')

#    colors = ('#732600','#ff6900','#ffaa00','#ffff73','#ffffff','#d3ffbe','#a3ff73','#38a800','#267300')
#    clevs= [-2.0,-1.5,-1.0,-0.5,0.5,1.0,1.5,2.0]
#    cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors, N=9, gamma=1.0)

    clevs, cmap1 = rainpalettle()

    cs = m.contourf(xi,yi,newout,clevs,cmap=cmap1,interpolation='gaussian', extend='both')

    cs.cmap.set_under('#ffaa00')
    cs.cmap.set_over('#267300')
    #cb = m.colorbar(cs, location='right', label="",pad="10%")


    m.readshapefile('shp/reproj_muni', 'MUNICIPIO', drawbounds = False, linewidth=0.6)

    for info, shape in zip(m.MUNICIPIO_info, m.MUNICIPIO):
        x, y = zip(*shape) 
        m.plot(x, y, marker=None, color=(60/255,60/255,60/255), linewidth=0.6)


    m.readshapefile('shp/reproj_prov', 'PROVINCIA', drawbounds = False, linewidth=1.5)

    for info, shape in zip(m.PROVINCIA_info, m.PROVINCIA):
        x, y = zip(*shape) 
        m.plot(x, y, marker=None, color='k', linewidth=1.5)


    if provname != "Occidente" or provname != "Centro" or provname != "Oriente":

        m.readshapefile('shp/gis_osm_roads_free_1', 'roads', drawbounds = False, linewidth=0.2)

        for info, shape in zip(m.roads_info, m.roads):
            x, y = zip(*shape) 
            m.plot(x, y, marker=None, color=(200/255,200/255,200/255), linewidth=0.2)


        m.readshapefile('shp/gis_osm_waterways_free_1', 'waterways', drawbounds = False, linewidth=0.6)

        for info, shape in zip(m.waterways_info, m.waterways):
            x, y = zip(*shape) 
            m.plot(x, y, marker=None, color='b', linewidth=0.6)

    else:

        centros = np.loadtxt('Municipios/centroides.txt')
        m.scatter(centros[:,1],centros[:,2], marker='o',color='r', s=10, zorder=2)



    r = shapefile.Reader(r"shp/reproj_prov")
    shapes = r.shapes()
    records = r.records()

    idprov=int(idprov)
    if idprov <= 16:

        c=0
        for record, shape in zip(records,shapes):
            c+=1
            lons,lats = zip(*shape.points)
            data = np.array(m(lons, lats)).T
#            print record
            
            if len(shape.parts) == 1:
                segs = [data,]
            else:
                segs = []
                for i in range(1,len(shape.parts)):
                    index = shape.parts[i-1]
                    index2 = shape.parts[i]
                    segs.append(data[index:index2])
                segs.append(data[index2:])


            if c != idprov:

                lines = LineCollection(segs,antialiaseds=(1,))
                lines.set_facecolors('white')
                lines.set_edgecolors('w')
                lines.set_linewidth(0.1)
                ax.add_collection(lines)


#    ax2 = fig.add_axes([0.925, 0.15, 0.02, 0.69])
#    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, spacing='uniform', ticks=clevs, boundaries=clevs, format='%2.1f')

    ax2 = fig.add_axes([0.92, 0.30, 0.02, 0.48])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, spacing='uniform', ticks=clevs, boundaries=clevs, format='%2.0f', orientation="vertical")

#    cb.set_label('VALORES DE PRECIPITACION ACUMULADA\n[mm/mes]', labelpad=-86, fontsize=10)

    fig.suptitle("\n\n\nSISTEMA NACIONAL DE VIGILANCIA DE LA SEQUIA\nCENTRO NACIONAL DEL CLIMA\nINSTITUTO DE METEOROLOGIA DE CUBA\n", ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

    plt.title(tlist[:]+" ( "+mes+"  Norma: 1971-2000  Provincia: "+provname+" ) \n\n", ha='center', va='center', fontsize=11, transform=ax.transAxes)

#    plt.text(0.22, 0.294,"S         M         D                 D         M         S", ha='center', va='center', fontsize=11, transform=ax.transAxes)

#    plt.text(0.22, 0.18,"S-E: SEVERO - EXTREMO  M: MODERADO  D: DEBIL  S/N: SIN DEFICIT", ha='center', va='center', fontsize=10, transform=ax.transAxes)


    ##plt.savefig(tlist.split()[0]+"_"+provname+".png")
    plt.savefig(tlist.split()[0]+"_"+provname+".png", pad_inches=0)
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

#    plt.figure(1)
#    plt.imshow(out)
#    mask = np.loadtxt("mask.txt")[-1::-1,:]
#    newout = np.zeros(shape=out.shape)

#    for i in range(mask.shape[0]):
#        for j in range(mask.shape[1]):

#            if int(mask[i,j]) == 0:
#                newout[i,j] = np.nan
#            else:
#                newout[i,j] = out[i,j]

#    plt.figure(2)
#    plt.imshow(newout)

#    plt.figure(3)
#    plt.imshow(mask)

#    print mask.shape, out.shape
#    plt.show()
#    sys.exit()

    mask = np.loadtxt("mask.txt")[-1::-1,:]
    newout = np.zeros(shape=out.shape)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):

            if int(mask[i,j]) == 0:
                newout[i,j] = np.nan
            else:
                newout[i,j] = out[i,j]

    return newout

#    return newout


def rainpalettle():

    colors = [(255,255,255),(210,255,254),(136,254,253),(0,198,255),(25,150,255),(60,65,255),(60,188,61),(165,215,31),(255,230,0),(255,195,0),(255,125,0),(255,0,0),(200,0,0),(212,100,195),(181,86,71),(132,0,148),(220,220,220),(180,180,180),(140,140,140),(90,90,90),(50,50,50)]
    clevs = np.array([0,0.2,1,3,5,7,10,15,20,25,30,40,50,60,70,80,100,125,150,175,200])*2

    colors = ['#ffaa00','#ffd37f','#ffebaf','#ffffbe','#d3ffbe','#a3ff73','#55ff00','#38a800','#267300']
    clevs = [0,10,20,50,100,150,200,300,400,600]

    nclevs = []
    for i in np.arange(0,600,5):
        nclevs.append(float(i))

    ncmaps = []
    for value in range(len(nclevs)):
        for i in range(len(colors)):
            if float(nclevs[value]) >= float(clevs[i]) and float(nclevs[value]) < float(clevs[i+1]):
                #ncmaps.append((float(colors[i][0])/255,float(colors[i][1])/255,float(colors[i][2])/255))
                ncmaps.append(colors[i])

    cmap1 = ListedColormap(ncmaps, name='from_list', N=None)

#    colors = ['#ffaa00','#ffd37f','#ffebaf','#ffffbe','#d3ffbe','#a3ff73','#55ff00','#38a800','#267300']
#    clevs = [0,10,20,50,100,150,200,300,400]
#    cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors)


    return clevs, cmap1




if __name__=='__main__':
    main()
