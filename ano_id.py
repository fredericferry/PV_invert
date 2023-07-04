# -*- coding: utf-8 -*-
"""
Projet modélisation - Outil d'inversion du tourbillon potentiel
Quentin ALGISI, Thomas BURGOT, Marie CASSAS, Benoît TOUZE
Adapté d'un programme Matlab écrit par G. J. Hakim (2013)
Janvier - Février 2016
-------------------------------------------------------------------------------
Définition d'anomalies idéalisées de TPQG :

    - bulle_tpqg : crée une anomalie en forme de "bulle" à la position indiquée
    
    - double_ano_tpqg : crée deux anomalies, une positive, une négative de part
            et d'autre d'une position donnée par rapport à l'axe zonal pour
            faire un rapide de jet
    
    - ano_theta : crée une anomalie de theta sur l'horizontale à la position
            indiquée
"""

from constantes_id import AMP, NLAT, NLON, AMP_TH
from init_id import NZ, LATS, LONS, NIV_P
from numpy import exp, meshgrid, sqrt, zeros
  
def bulle_tpqg(p_max, lat_max, lon_max, signe = "+", forme = "sphere") :
    """
    Définit une anomalie 3D gaussienne dans un champ à la position donnée.
    
    Paramètres :
    ------------
    p_max : float
        Niveau pression du centre de l'anomalie.
    lat_max : float
        Latitude du centre de l'anomalie.
    lon_max : float
        Longitude du centre de l'anomalie.
    signe : str
        Signe de l'anomalie de TPQG : entre "+" et "-" (par défaut "+").
    forme : str
        Forme de l'anomalie de TPQG : entre "sphere", "large" et "fine" (par 
        défaut "sphere").
        
    Sortie :
    ------------
    ano : ndarray, float
        Tableau 3D du champ d'anomalie de TPQG.    
    """
    if (forme == "sphere" or forme == "bulle" or forme == "boule") :
        AZ = 9000
        AX = 1
        AY = AX
    elif (forme == "large" or forme == "plate") :
        AZ = 8000
        AX = 3
        AY = AX
    elif (forme == "fine" or forme == "etiree" or forme == "étirée") :
        AZ = 20000
        AX = 1
        AY = AX
        
    if (signe == "+" or signe == "positif" or signe == "positive" or 
    signe == "plus") :
        ampli = AMP
    elif (signe == "-" or signe == "negatif" or signe == "negative" or 
    signe == "négatif" or signe == "négative" or signe == "moins") :
        ampli = -AMP
    
    lon, lat = meshgrid(LONS, LATS)
    rr = sqrt(((lon - lon_max) / AX) ** 2 + ((lat - lat_max) / AY) ** 2)
    
    ano = zeros((NZ, NLAT, NLON))
    
    for niv in range(NZ) :
        ano[niv, :, :] = (ampli * exp(-((NIV_P[niv] - p_max) / AZ) ** 2) * 
        exp(-(rr ** 2)))
            
    return ano
    
def double_ano_tpqg(p_m, lat_m, lon_m) :
    """
    Définit une double anomalie 3D gaussienne dans un champ à la position 
    donnée. L'anomalie positive est au Nord, négative au Sud.
    
    Paramètres :
    ------------
    p_max : float
        Niveau pression du milieu entre les deux anomalies.
    lat_max : float
        Latitude du milieu entre les deux anomalies.
    lon_max : float
        Longitude du milieu entre les deux anomalies.
        
    Sortie :
    ------------
    ano : ndarray, float
        Tableau 3D du champ d'anomalie de TPQG.
    """
    AZ = 9000
    #AXn = 5
    #AYn = 1
    AXn = 3
    AYn = 3
    AXp = 3
    AYp = 3
    
    lon, lat = meshgrid(LONS, LATS)
    rr1 = sqrt(((lon - lon_m) / AXp) ** 2 + ((lat - lat_m - 1.5) / AYp) ** 2)
    rr2 = sqrt(((lon - lon_m) / AXn) ** 2 + ((lat - lat_m + 1.5) / AYn) ** 2)
    
    ano = zeros((NZ, NLAT, NLON))
    
    for niv in range(NZ) :
        ano[niv, :, :] = (AMP * exp(-((NIV_P[niv] - p_m) / AZ) ** 2) * 
        exp(-(rr1 ** 2))) + (-AMP * exp(-((NIV_P[niv] - p_m) / AZ) ** 2) * 
        exp(-(rr2 ** 2)))
    
    return ano
    
def ano_theta(lat_max, lon_max, signe = "+") :
    """
    Définit une anomalie de theta en basses couches dans un champ à la position
    donnée.
    
    Paramètres :
    ------------
    lat_max : float
        Latitude du centre de l'anomalie.
    lon_max : float
        Longitude du centre de l'anomalie.
    signe : str
        Signe de l'anomalie de theta : entre "+" et "-" (par défaut "+").
    
    Sortie :
    ------------
    ano : ndarray, float
        Tableau 2D du champ d'anomalie de theta.
    """
    AX = 3
    AY = AX
    
    lon, lat = meshgrid(LONS, LATS)
    rr = sqrt(((lon - lon_max) / AX) ** 2 + ((lat - lat_max) / AY) ** 2)
    
    if (signe == "+" or signe == "positif" or signe == "positive" or 
    signe == "plus") :
        ano = AMP_TH * exp(-(rr ** 2))
    elif (signe == "-" or signe == "negatif" or signe == "negative" or 
    signe == "négatif" or signe == "négative" or signe == "moins") :
        ano = -AMP_TH * exp(-(rr ** 2))
    
    return ano
