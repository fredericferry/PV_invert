# -*- coding: utf-8 -*-
"""
Projet modélisation - Outil d'inversion du tourbillon potentiel
Quentin ALGISI, Thomas BURGOT, Marie CASSAS, Benoît TOUZE
Adapté d'un programme Scilab écrit par Philippe ARBOGAST
Janvier - Février 2016
-------------------------------------------------------------------------------
Outils de lissage de champs:

    - liss2d : lissage d'un champ 3D par moyennage sur l'horizontale à chaque
            niveau vertical
            
    - liss : lissage d'un champ 3D par moyennage dans les 3 dimensions
"""

import numpy as np

def liss2d(champ, niter) :
    """
    Permet de lisser un champ 3D en le considérant niveau vertical par niveau
    vertical (lissage 2D à chaque niveau).
    
    Paramètres :
    ------------
    champ : ndarray, float
        Tableau 3D du champ à lisser.
    niter : int
        Nombre d'itérations de lissage à faire.
        
    Sortie :
    ------------
    champ_lis : ndarray, float
        Tableau 3D du champ lissé.
    """
    # Récupération des dimensions du champ de points de grille
    dim = champ.shape
    nlat = dim[1]  # nombre de niveaux en latitude
    nlon = dim[2]  # nombre de niveaux en longitude
    
    # Création/initialisation du champ lissé
    champ_lis = np.zeros(dim)
    champ_temp = champ  # champ intermédiaire utilisé pour le calcul
    
    # Calcul du champ lissé par niter itérations (barycentre entre le point et
    # ses 4 voisins)
    for i in range(niter) :
        champ_lis[:, 1:nlat-1, 1:nlon-1] = (
        0.5 * champ_temp[:, 1:nlat-1, 1:nlon-1] + 0.5 / 4 * (
        champ_temp[:, 0:nlat-2, 1:nlon-1] + champ_temp[:, 2:nlat, 1:nlon-1] +
        champ_temp[:, 1:nlat-1, 0:nlon-2] + champ_temp[:, 1:nlat-1, 2:nlon]))
        
        champ_temp = champ_lis
        
    return champ_lis
        
def liss(champ, niter) :
    """
    Permet de lisser un champ 3D.
    
    Paramètres :
    ------------
    champ : ndarray, float
        Tableau 3D du champ à lisser.
    niter : int
        Nombre d'itérations de lissage à faire.
        
    Sortie :
    ------------
    champ_lis : ndarray, float
        Tableau 3D du champ lissé.
    """
    # Récupération des dimensions du champ de points de grille
    dim = champ.shape
    nz = dim[0]    # nombre de niveaux verticaux
    nlat = dim[1]  # nombre de niveaux en latitude
    nlon = dim[2]  # nombre de niveaux en longitude
    
    # Création/initialisation du champ lissé
    champ_lis = np.zeros(dim)
    champ_temp = champ  # champ intermédiaire utilisé pour le calcul 
    
    # Calcul du champ lissé par niter itérations (barycentre entre le point et
    # ses 6 voisins)
    for i in range(niter) :
        champ_lis[1:nz-1, 1:nlat-1, 1:nlon-1] = (
        0.5 * champ_temp[1:nz-1, 1:nlat-1, 1:nlon-1] + 0.5 / 6 * (
        champ_temp[1:nz-1, 0:nlat-2, 1:nlon-1] + 
        champ_temp[1:nz-1, 2:nlat, 1:nlon-1] +
        champ_temp[1:nz-1, 1:nlat-1, 0:nlon-2] + 
        champ_temp[1:nz-1, 1:nlat-1, 2:nlon] + 
        champ_temp[0:nz-2, 1:nlat-1, 1:nlon-1] +
        champ_temp[2:nz, 1:nlat-1, 1:nlon-1]))
        
        champ_temp = champ_lis
    
    return champ_lis
