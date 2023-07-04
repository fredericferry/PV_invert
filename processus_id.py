# -*- coding: utf-8 -*-
"""
Projet modélisation - Outil d'inversion du tourbillon potentiel
Quentin ALGISI, Thomas BURGOT, Marie CASSAS, Benoît TOUZE
Adapté d'un programme Scilab écrit par Philippe ARBOGAST
Janvier - Février 2016
-------------------------------------------------------------------------------
Processus effectuant toutes les opérations pour les cas idéalisés avec 
inversion du TPQG.
"""

from init_id import (ZSTAR, COEFF_T_THETA, NZ, LATS, LONS, TH_INI, T_INI, 
                     U_INI, V_INI)
from ano_id import bulle_tpqg, double_ano_tpqg
from constantes_id import NLAT, NLON, W_TPQG, W_OMEGA, LON_MIN, LON_MAX
from diagnostics_id import theta, inv_tpqg, vor_geop, wind, inv_omega
import numpy as np

def processus_tpqg_id(alt, type_ano = "simple", signe = "+", 
                      forme = "sphere", bc = False, signebc = "+", tilt = 0) :
    """
    Réalise toutes les opérations nécessaires pour obtenir les champs pour les
    sorties graphiques dans le cas d'une anomalie artificielle de TPQG dans une 
    zone barocline idéalisée.
    
    Paramètres :
    ------------
    alt : float
        Niveau vertical du centre de l'anomalie en hPa.
    type_ano : str
        Type d'anomalie à créer : entre "simple" et "double" (par défaut 
        "simple").
    signe : str
        Signe de l'anomalie dans le cas d'une anomalie simple : entre "+" et 
        "-" (par défaut "+").
    forme : str
        Forme de l'anomalie dans le cas d'une anomalie simple : entre "sphere",
        "fine" et "large" (par défaut "sphere").
    bc : boolean
        Variable pour choisir si on veut ajouter une anomalie de basses couches
        entre True et False (par défaut False).
    signebc : str
        Signe de l'anomalie de basses couches à ajouter : entre "+" et "-" 
        (par défaut "+").
    tilt : float
        Valeur de décalage entre l'anomalie d'altitude et l'anomalie de basses
        couches en degrés (par défaut 0). Le tilt doit être un multiple de 0.5.
    
    Sorties :
    ------------
    ZSTAR : ndarray, float
        Liste des niveaux verticaux en coordonnée z*.
    LATS : ndarray, float
        Liste des latitudes du domaine (ordre Nord --> Sud).
    LONS : ndarray, float
        Liste des longitudes du domaine (ordre Ouest --> Est).
    ilat0 : float
        Indice de latitude du centre de l'anomalie.
    ilon0 : float
        Indice de longitude du centre de l'anomalie.
    U_INI : ndarray, float
        Tableau 3D du champ de vent zonal initial.
    V_INI : ndarray, float
        Tableau 3D du champ de vent méridien initial.
    FF_INI : ndarray, float
        Tableau 3D du champ de norme du vent initial.
    ANO : ndarray, float
        Tableau 3D du champ d'anomalie de TPQG.
    DPHI : ndarray, float
        Tableau 3D du champ d'anomalie de géopotentiel.
    DU : ndarray, float
        Tableau 3D du champ d'anomalie de vent zonal.
    DV : ndarray, float
        Tableau 3D du champ d'anomalie de vent méridien.
    U_ANO : ndarray, float
        Tableau 3D du champ de vent zonal avec anomalie.
    V_ANO : ndarray, float
        Tableau 3D du champ de vent méridien avec anomalie.
    FF_ANO : ndarray, float
        Tableau 3D du champ de norme du vent avec anomalie.
    TH_ANO : ndarray,float
        Tableau 3D du champ de température potentielle avec anomalie.
    DTR : ndarray, float
        Tableau 3D du champ d'anomalie de tourbillon.
    OMEGA_INI : ndarray, float
        Tableau 3D du champ de vitesse verticale omega initiale.
    OMEGA_ANO : ndarray, float
        Tableau 3D du champ de vitesse verticale avec anomalie.
    DOMEGA : ndarray, float
        Tableau 3D du champ d'anomalie de vitesse verticale.
    """
    lat0 = 39
    lon0 = (LON_MAX + LON_MIN) / 2

    ilat0 = np.where(LATS == lat0)[0][0]
    ilon0 = np.where(LONS == lon0)[0][0]
    
    if type_ano == "double" :
        ANO = double_ano_tpqg(alt * 100, lat0, lon0)
    elif type_ano == "simple" :
        ANO = bulle_tpqg(alt * 100, lat0, lon0, signe, forme)

    DPHI = inv_tpqg(ANO, 800, W_TPQG, bc, signebc, tilt)

    DTH = theta(DPHI)
    TH_ANO = TH_INI + DTH

    T_ANO = np.zeros((NZ, NLAT, NLON))
    for niv in range(NZ) :
        T_ANO[niv, :, :] = TH_ANO[niv, : ,:] / COEFF_T_THETA[niv]

    DTR = vor_geop(DPHI)
    DU, DV = wind(DPHI)
    U_ANO = U_INI + DU
    V_ANO = V_INI + DV
    
    FF_INI = np.sqrt(U_INI ** 2 + V_INI ** 2)
    FF_ANO = np.sqrt(U_ANO ** 2 + V_ANO ** 2)

    OMEGA_INI = inv_omega(U_INI, V_INI, T_INI, 500, W_OMEGA)
    OMEGA_ANO = inv_omega(U_ANO, V_ANO, T_ANO, 500, W_OMEGA)
    
    DOMEGA = OMEGA_ANO - OMEGA_INI
    
    return ZSTAR, LATS, LONS, ilat0, ilon0, U_INI, V_INI, FF_INI, ANO, DPHI,\
    DU, DV, U_ANO, V_ANO, FF_ANO, TH_INI, TH_ANO, DTR, OMEGA_INI, OMEGA_ANO,\
    DOMEGA
