# -*- coding: utf-8 -*-
"""
Projet modélisation - Outil d'inversion du tourbillon potentiel
Quentin ALGISI, Thomas BURGOT, Marie CASSAS, Benoît TOUZE
Adapté d'un programme Scilab écrit par Philippe ARBOGAST
Janvier - Février 2016
-------------------------------------------------------------------------------
Définition des constantes et paramètres utilisés dans le programme.
"""

from numpy import pi, sin, cos

#######################################
# Constantes physiques générales :
# Latitude de référence
LAT = 45. * pi / 180
# Pour la projection des distances sur l'axe zonal
C = cos(LAT)
# Ecart entre les points de grille selon l'axe méridien (équivaut à 0.5°)
DY = 6370000 * 0.5 * pi / 180
# Ecart entre les points de grille selon l'axe zonal
DX = DY * C
# Paramètre de Coriolis
F = 2 * 2 * pi / (3600 * 24) * sin(LAT)
# Température potentielle de référence
THETA0 = 300.
# Constante d'accélération gravitationnelle
G = 9.81
# Constante des gaz parfaits pour l'air sec
RA = 287.05
# Chaleur massique de l'air
CP = 1005.
# Pression de référence
P0 = 1000. * 100
# Pour la coordonnée en ln(P)
DP = 10000.

#######################################
# Coefficients de relaxation pour les inversions :
W_TPQG = 0.3
W_OMEGA = 0.3

#######################################
# Configuration des anomalies :
# Intensité de l'anomalie de TPQG
AMP = 5 * 10 ** (-4)
# Intensité de l'anomalie de température potentielle
AMP_TH = 10

#######################################
# Domaine spatial :
# Latitudes et longitudes minimales et maximales :
LAT_MIN = 27
LAT_MAX = 47
LON_MIN = -40
LON_MAX = -20

# Nombre de points en latitude et longitude :
NLAT = (LAT_MAX - LAT_MIN) * 2 + 1
NLON = (LON_MAX - LON_MIN) * 2 + 1
