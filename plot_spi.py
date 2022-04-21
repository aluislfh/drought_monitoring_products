# -*- coding: utf-8 -*-
#!/usr/bin/env python
import datetime as dt
import datetime
#import datetime.datetime
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
#import shapefile
from matplotlib.collections import LineCollection
import matplotlib as mpl


def main():

    ilist = ['SPI12.csv','SPI15.csv','SPI18.csv','SPI3.csv','SPI6.csv','SPI9.csv','SPIMes.csv']
    tlist = ['SPI12','SPI15','SPI18','SPI3','SPI6','SPI9','SPI1']

#    ilist = ['SPIMes.csv']
#    tlist = ['SPI1']

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

        if tlist[i] == 'SPI1':
            newmes      = mes
        else:
            anterior = before(mes,tlist[i])
            newmes      = anterior+"-"+mes

#---
        
        plotspi(var,elat2,elon2,tlist[i],newmes)


#   Mapas provinciales

#    for i in range(len(ilist)):


#        prov = open("provincias_ok.txt",'r')

#        
#        idprov=0
#        for j in range(19):
#            idprov=int(idprov)
#            idprov+=1
#            name, lat1, lat2, lon1, lon2, idprov = (prov.readline()).split()
#            plotspi_prov(var,elat2,elon2,tlist[i],newmes, lon1, lon2, lat1, lat2, name, idprov)

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

    fig = plt.figure(1,figsize=(16.00, 7.20), dpi=300)
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

#    colors = ('#ff6900','#ffaa00','#ffff73','#ffffff','#d3ffbe','#a3ff73','#38a800')
    colors = ('#732600','#ff6900','#ffaa00','#ffff73','#ffffff','#d3ffbe','#a3ff73','#38a800','#267300')
    clevs= [-3.0,-2.0,-1.5,-1.0,-0.5,0.5,1.0,1.5,2.0,3.0]
    cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors, N=9, gamma=1.0)

    cs = m.contourf(xi,yi,newout,clevs,cmap=cmap1,interpolation='gaussian', extend='both')

    cs.cmap.set_under('#732600')
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

#    cs.cmap.set_under('#732600')
#    cs.cmap.set_over('#267300')

    #[*left*, *bottom*, *width*, *height*]
    ax2 = fig.add_axes([0.17, 0.36, 0.25, 0.03])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, spacing='uniform', ticks=clevs, boundaries=clevs, format='%2.1f', orientation="horizontal", extend='both')


    cb.set_ticklabels(['<-2.0',-1.5,-1.0,-0.5,0.5,1.0,1.5,'>2.0'])
#    cb.set_label('[Valores de SPI]', rotation=270, labelpad=30, y=0.45)
#    cb.set_label('[Valores de SPI]', rotation=0, labelpad=-50, y=0.45)


#    lon = -76.0
#    lat = 23.1
#    x, y = m(lon, lat)
#    plt.text(x, y, "SISTEMA NACIONAL DE VIGILANCIA DE LA SEQUIA\nCENTRO NACIONAL DEL CLIMA\nINSTITUTO DE METEOROLOGIA DE CUBA\n",fontsize=14,fontweight='bold', ha='left',va='bottom',color='k')

#    lon = -76.0
#    lat = 22.6
#    x, y = m(lon, lat)
#    plt.text(x, y, "INDICE DE PRECIPITACION ESTANDARIZADA ("+tlist+")\n"+mes+". Norma: 1971-2000",fontsize=12, ha='left',va='bottom',color='k')

    plt.text(0.80, 0.88,"SISTEMA NACIONAL DE VIGILANCIA DE LA SEQUIA\nCENTRO NACIONAL DEL CLIMA\nINSTITUTO DE METEOROLOGIA DE CUBA\n", ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

    plt.text(0.80, 0.77,"INDICE DE PRECIPITACION ESTANDARIZADA ("+tlist[:3]+" "+tlist[3:]+")\n"+mes+".\nNorma: 1971-2000", ha='center', va='center', fontsize=11, transform=ax.transAxes)

    plt.text(0.30, 0.35,"EXCESO", ha='center', va='center', fontsize=10,color='g', transform=ax.transAxes)
    plt.text(0.15, 0.35,"DEFICIT", ha='center', va='center', fontsize=10,color='r', transform=ax.transAxes)
    plt.text(0.22, 0.294,"S         M         D                 D         M         S", ha='center', va='center', fontsize=11, transform=ax.transAxes)
    plt.text(0.218, 0.2954, "E                                                                                                                      E", ha='center', va='center',color='w', fontsize=7, transform=ax.transAxes)
    plt.text(0.22, 0.18,"S-E: SEVERO - EXTREMO  M: MODERADO  D: DEBIL", ha='center', va='center', fontsize=10, transform=ax.transAxes)


#    x2, y2 = m(-76.2, 21.9)
#    x3, y3 = m(-75.8, 21.6)
#    plt.imshow(plt.imread('logo.png'), extent=(x2, x3, y2, y3))


    ##plt.savefig(tlist+"_CUBA.png")
    plt.savefig(tlist+"_CUBA.png", pad_inches=0)
    plt.clf()
    plt.cla()





def plotspi_prov(outspi,ilats,ilons,tlist,mes,lon1,lon2,lat1,lat2,provname,idprov):

    out,lats,lons = vinterp(outspi,ilats,ilons)
    newout = checkmask(out)

    fig = plt.figure(1,figsize=(16.00, 7.20), dpi=300)
    ax  = fig.add_subplot(111)
    #fig.suptitle("Mapa de peligro", fontsize=14, fontweight='bold')
#    plt.title("Mapa de "+tlist+" correspondiente al mes de \n"+mes, fontsize=14, fontweight='bold')

    m = Basemap(resolution='h',llcrnrlon=float(lon1), llcrnrlat=float(lat1), urcrnrlon=float(lon2),
             urcrnrlat=float(lat2), projection='lcc', lat_0=float(lat1)+(np.abs(float(lat1)-float(lat2))/2), 
                lon_0=float(lon1)+(np.abs(float(lon1)-float(lon2))/2), area_thresh=50.)
    xi, yi = m(lons, lats)

    m.drawmeridians(range(0, 360, 1),labels=[1,0,0,1],fontsize=10, color=(1,1,1,1), linewidth=0)
    m.drawparallels(range(-180, 180, 1),labels=[1,0,0,1],fontsize=10, color=(1,1,1,1), linewidth=0)
#    m.drawmapscale(-82.8, 20.0, -79, 22., 200, barstyle='fancy')

    m.drawmapboundary(fill_color='#ffffff')
    #m.fillcontinents(color='#dfdcd8',lake_color='#98acc0')

#    colors = ('#ff6900','#ffaa00','#ffff73','#ffffff','#d3ffbe','#a3ff73','#38a800')
    colors = ('#732600','#ff6900','#ffaa00','#ffff73','#ffffff','#d3ffbe','#a3ff73','#38a800','#267300')
    clevs= [-3.0,-2.0,-1.5,-1.0,-0.5,0.5,1.0,1.5,2.0,3.0]
    cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors, N=9, gamma=1.0)

    cs = m.contourf(xi,yi,newout,clevs,cmap=cmap1,interpolation='gaussian', extend='both')

    cs.cmap.set_under('#732600')
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

#    cs.cmap.set_under('#732600')
#    cs.cmap.set_over('#267300')

    #[*left*, *bottom*, *width*, *height*]
#    ax2 = fig.add_axes([0.17, 0.36, 0.25, 0.03])
#    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, spacing='uniform', ticks=clevs, boundaries=clevs, format='%2.1f', orientation="horizontal", extend='both')

    ax2 = fig.add_axes([0.92, 0.30, 0.02, 0.48])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, spacing='uniform', ticks=clevs, boundaries=clevs, format='%2.0f', orientation="vertical")

    cb.set_ticklabels(['<-2.0',-1.5,-1.0,-0.5,0.5,1.0,1.5,'>2.0'])
#    cb.set_label('[Valores de SPI]', rotation=270, labelpad=30, y=0.45)
#    cb.set_label('[Valores de SPI]', rotation=0, labelpad=-50, y=0.45)


#    lon = -76.0
#    lat = 23.1
#    x, y = m(lon, lat)
#    plt.text(x, y, "SISTEMA NACIONAL DE VIGILANCIA DE LA SEQUIA\nCENTRO NACIONAL DEL CLIMA\nINSTITUTO DE METEOROLOGIA DE CUBA\n",fontsize=14,fontweight='bold', ha='left',va='bottom',color='k')

#    lon = -76.0
#    lat = 22.6
#    x, y = m(lon, lat)
#    plt.text(x, y, "INDICE DE PRECIPITACION ESTANDARIZADA ("+tlist+")\n"+mes+". Norma: 1971-2000",fontsize=12, ha='left',va='bottom',color='k')

#    plt.text(0.80, 0.88,"SISTEMA NACIONAL DE VIGILANCIA DE LA SEQUIA\nCENTRO NACIONAL DEL CLIMA\nINSTITUTO DE METEOROLOGIA DE CUBA\n", ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

#    plt.text(0.80, 0.77,"INDICE DE PRECIPITACION ESTANDARIZADA ("+tlist[:3]+" "+tlist[3:]+")\n"+mes+".\nNorma: 1971-2000", ha='center', va='center', fontsize=11, transform=ax.transAxes)

#    plt.text(0.30, 0.35,"EXCESO", ha='center', va='center', fontsize=10,color='g', transform=ax.transAxes)
#    plt.text(0.15, 0.35,"DEFICIT", ha='center', va='center', fontsize=10,color='r', transform=ax.transAxes)
#    plt.text(0.22, 0.294,"S         M         D                 D         M         S", ha='center', va='center', fontsize=11, transform=ax.transAxes)
#    plt.text(0.218, 0.2954, "E                                                                                                                      E", ha='center', va='center',color='w', fontsize=7, transform=ax.transAxes)
#    plt.text(0.22, 0.18,"S-E: SEVERO - EXTREMO  M: MODERADO  D: DEBIL", ha='center', va='center', fontsize=10, transform=ax.transAxes)


    cb.set_label('VALORES INDICE DE PRECIPITACION ESTANDARIZADA (' +tlist[:]+')', labelpad=-86, fontsize=10)

    fig.suptitle("\n\n\nSISTEMA NACIONAL DE VIGILANCIA DE LA SEQUIA\nCENTRO NACIONAL DEL CLIMA\nINSTITUTO DE METEOROLOGIA DE CUBA\n", ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

    plt.title("INDICE DE PRECIPITACION ESTANDARIZADA ("+tlist[:]+")  "+mes+"  Norma: 1971-2000  Provincia: "+provname+" ) \n\n", ha='center', va='center', fontsize=11, transform=ax.transAxes)


    plt.savefig(tlist+"_"+provname+".png")
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




def before(mes,tlist):
#    print mes,tlist
    Meses = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
    meses = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"]
    MESES = ["ENERO","FEBRERO","MARZO","ABRIL","MAYO","JUNIO","JULIO","AGOSTO","SEPTIEMBRE","OCTUBRE","NOVIEMBRE","DICIEMBRE"]

    MM = mes.split()[0]
    YYYY = mes.split()[-1]

    c = 0
    march = {}
    for i in np.arange(1,len(meses)+1,1):
        march[Meses[i-1]] = i
        if meses[i-1] == MM or Meses[i-1] == MM or MESES[i-1] == MM:
            c=i

    spi = int(tlist[3:])

#    print int(c), int(YYYY)
    ndate = datetime.datetime(int(YYYY),int(c),15,0)

    odates    = str(ndate).split()
    odfolders = odates[0]

    oyear   = odfolders[0:4]
    omonth  = int(odfolders[5:7])


    adate = ndate - datetime.timedelta(days=spi*31)

    adate = adate + datetime.timedelta(days=31)

#    if c == 1:
#        adate = adate + datetime.timedelta(days=31)
#    else:
#        adate = adate + datetime.timedelta(days=2*31)

    idates    = str(adate).split()
    dfolders = idates[0]

    year   = dfolders[0:4]
    month  = int(dfolders[5:7])-1

#    print month, Meses
    if oyear != year:
        anterior = Meses[month]+" del "+year
    else:
        anterior = Meses[month]


    return anterior


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
