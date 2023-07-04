# -*- coding: utf-8 -*-
"""
Projet modélisation - Outil d'inversion du tourbillon potentiel
Quentin ALGISI, Thomas BURGOT, Marie CASSAS, Benoît TOUZE
Janvier - Février 2016
-------------------------------------------------------------------------------
Etat initial idéalisé type zone barocline idéalisée.
"""

from donnees import proce
from constantes_id import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, NLAT, NLON, RA, \
CP, THETA0, G, P0
from numpy import zeros, vstack, asarray, delete, mean, where, log

#######################################
# Extraction et redimensionnage des champs du fichier du 1er janvier 2016
# Fichier de la situation réelle utilisée pour créer l'état initial
FICHIER = "1er_janvier_2016.nc"
 
# Lecture des champs de départ
U_TOUS = proce(FICHIER, 'u')
V_TOUS = proce(FICHIER, 'v')
T_TOUS = proce(FICHIER, 't_3')
PHI_TOUS = proce(FICHIER, 'z_2')

# Lecture des latitudes, longitudes, niveaux verticaux
LATS_TOUS = proce(FICHIER, 'lat')
LONS_TOUS = proce(FICHIER, 'lon')
NIV_P_TOUS = proce(FICHIER, 'lev')

# Nombres de latitudes, longitudes et niveaux verticaux disponibles
NLEVEL = NIV_P_TOUS.shape[0]
    
# Position des bornes pour le dimensionnage horizontal
ilatmin = where(LATS_TOUS == LAT_MIN)[0][0]
ilatmax = where(LATS_TOUS == LAT_MAX)[0][0]
    
ilonmin = where(LONS_TOUS == LON_MIN)[0][0]
ilonmax = where(LONS_TOUS == LON_MAX)[0][0]
    
# Nouvelles dimensions horizontales
LATS = LATS_TOUS[ilatmax:ilatmin+1]
LONS = LONS_TOUS[ilonmin:ilonmax+1]

# Coupe des champs de paramètres pour garder seulement entre 50 et 900 hPa
# et entre 25 et 45°N, et 20 et 40°W
T = zeros((1, NLAT, NLON))
U = zeros((1, NLAT, NLON))
V = zeros((1, NLAT, NLON))
PHI = zeros((1, NLAT, NLON))

NIV_P = [] # Niveaux verticaux conservés entre 50 et 900 hPa
for niv in range(NLEVEL) :
    if (NIV_P_TOUS[niv] >= 5000 and NIV_P_TOUS[niv] <= 90000) :
        NIV_P.append(NIV_P_TOUS[niv])
        T = vstack((T, T_TOUS[niv, ilatmax:ilatmin+1, ilonmin:ilonmax+1]
        .reshape(1, NLAT, NLON)))
        U = vstack((U, U_TOUS[niv, ilatmax:ilatmin+1, ilonmin:ilonmax+1]
        .reshape(1, NLAT, NLON)))
        V = vstack((V, V_TOUS[niv, ilatmax:ilatmin+1, ilonmin:ilonmax+1]
        .reshape(1, NLAT, NLON)))
        PHI = vstack((PHI, PHI_TOUS[niv, ilatmax:ilatmin+1, ilonmin:ilonmax+1]
        .reshape(1, NLAT, NLON)))
        
NIV_P = asarray(NIV_P)
NZ = NIV_P.size # Nombre de niveaux verticaux conservés
U = delete(U, asarray(range(NLON * NLAT))).reshape(NZ, NLAT, NLON)
V = delete(V, asarray(range(NLON * NLAT))).reshape(NZ, NLAT, NLON)
T = delete(T, asarray(range(NLON * NLAT))).reshape(NZ, NLAT, NLON)
PHI = delete(PHI, asarray(range(NLON * NLAT))).reshape(NZ, NLAT, NLON)

#######################################
# Définition de l'état moyenné selon l'axe zonal pour chaque niveau
U_INI = zeros(U.shape)
V_INI = zeros(V.shape)
T_INI = zeros(T.shape)
PHI_INI = zeros(PHI.shape)

for niv in range(NZ) :
    for lati in range(NLAT) :
        U_INI[niv, lati, :] = mean(U[niv, lati, :])
        V_INI[niv, lati, :] = mean(V[niv, lati, :])
        T_INI[niv, lati, :] = mean(T[niv, lati, :])
        PHI_INI[niv, lati, :] = mean(PHI[niv, lati, :])
        
#######################################
# Repères des niveaux verticaux utiles        
N300 = where(NIV_P == 30000.)[0][0]   # Niveau 300 hPa
N500 = where(NIV_P == 50000.)[0][0]   # Niveau 500 hPa
N600 = where(NIV_P == 60000.)[0][0]   # Niveau 500 hPa
N850 = where(NIV_P == 85000.)[0][0]   # Niveau 500 hPa
N900 = where(NIV_P == 90000.)[0][0]   # Niveau 500 hPa
# La liste des niveaux pression en logarithme
LN_P = log(NIV_P)
# Coordonnée Z* de l'annexe du Coronel (version corrigée)
ZSTAR = CP * THETA0 / G * (1 - (NIV_P / P0) ** (RA / CP))

# Le coefficient dans l'équation reliant theta à T : (P0/P)^(R/CP)
COEFF_T_THETA = (P0 / NIV_P) ** (RA / CP)

#######################################
# Calcul du champ de température potentielle et des températures moyennes
TH_INI = zeros((NZ, NLAT, NLON))
TMOY = zeros(NZ)
THMOY = zeros(NZ)

for niv in range(NZ) :
    TH_INI[niv, :, :] = T_INI[niv, :, :] * COEFF_T_THETA[niv]
    TMOY[niv] = mean(T[niv, :, :])
    THMOY[niv] = mean(TH_INI[niv, :, :])
    
#######################################
# Inverse de RHO (densité)
INVRHO = RA * TMOY / NIV_P
# Coefficient sigma pour l'équation en oméga : -1/(rho*theta)*dtheta/dp
SIGMA = zeros(NZ)

for niv in range(1, NZ-1) :
    SIGMA[niv] = (- INVRHO[niv] * (THMOY[niv+1] - THMOY[niv-1]) / (THMOY[niv] *
    NIV_P[niv+1] - NIV_P[niv-1]))
