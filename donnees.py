# -*- coding: utf-8 -*-
"""
Projet modélisation - Outil d'inversion du tourbillon potentiel
Quentin ALGISI, Thomas BURGOT, Marie CASSAS, Benoît TOUZE
Janvier - Février 2016
-------------------------------------------------------------------------------
Gestion des fichiers de données netCDF.
"""

from netCDF4 import Dataset
import numpy as np

def proce(chemin_du_fichier, variable):
    """
    Procédure permettant la lecture des fichiers de données Netcdf 
    ainsi que la conversion des données en tableau ndarray.
    
    Paramètre : 
    -------------	
    Chemin_du_fichier : str
        Nom du fichier netcdf de données.
    
    variable : str
        Variable de sortie désirée.
    
    Sortie :
    -------------
    var : ndarray, float
        Tableau 3D de la variable voulue.        
    """

    file = chemin_du_fichier
    fh = Dataset(file, mode='r')
    
    # Conversion en ndarray (squeeze pour eliminer la dimension temporelle)
    var = np.squeeze(fh.variables[variable][:])
    
    fh.close()
    
    return var
