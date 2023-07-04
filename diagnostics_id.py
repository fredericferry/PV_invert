# -*- coding: utf-8 -*-
"""
Projet modélisation - Outil d'inversion du tourbillon potentiel
Quentin ALGISI, Thomas BURGOT, Marie CASSAS, Benoît TOUZE
Adapté d'un programme Scilab écrit par Philippe ARBOGAST
Janvier - Février 2016
-------------------------------------------------------------------------------
Diagnostics utilisés dans le cas idéalisé :

    - vor_geop : calcul du tourbillon relatif à partir du géopotentiel
    
    - wind : calcul du vent géostrophique à partir du géopotentiel
    
    - divq : calcul de la divergence du vecteur Q (forçage géostrophique) à
            partir du vent et de la température
            
    - theta : calcul de la température potentielle à partir du géopotentiel

    - tpqg : calcul du TPQG à partir de l'anomalie de géopotentiel
    
    - inv_omega : calcul de la vitesse verticale omega en inversant l'équation
            en omega
    
    - inv_tpqg : calcul de l'anomalie de géopotentiel en inversant l'anomalie
            de TPQG
"""

import numpy as np
from constantes_id import F, DX, DY, C, RA, LON_MIN, LON_MAX
from init_id import NIV_P, LN_P, SIGMA, COEFF_T_THETA
from lissage import liss2d
from ano_id import ano_theta

def vor_geop(phi) :
    """
    Retourne le champ de tourbillon relatif calculé à partir du champ de
    géopotentiel.
    
    Paramètre :
    ------------
    phi : ndarray, float
        Tableau 3D du champ de géopotentiel.
    
    Sortie :
    ------------
    tr : ndarray, float
        Tableau 3D du champ de tourbillon relatif.
    """
    # Récupération des dimensions du champ de points de grille
    dim = phi.shape
    nlat = dim[1]  # nombre de niveaux en latitude
    nlon = dim[2]  # nombre de niveaux en longitude

    # Création/initialisation du champ de tourbillon relatif
    tr = np.zeros(dim)
    
    # Calcul du tourbillon avec la formule tr = 1/f*lap(phi)
    tr[:, 1:nlat-1, 1:nlon-1] = (((phi[:, 0:nlat-2, 1:nlon-1] + 
    phi[:, 2:nlat, 1:nlon-1] - 2 * phi[:, 1:nlat-1, 1:nlon-1]) / DY ** 2 + (
    phi[:, 1:nlat-1, 0:nlon-2] + phi[:, 1:nlat-1, 2:nlon] 
    - 2 * phi[:, 1:nlat-1, 1:nlon-1]) / DX ** 2) / F)
    
    return tr
    
def wind(phi) :
    """
    Retourne les champs de vents zonal et méridien calculés à partir du
    géopotentiel.
    
    Paramètre :
    ------------
    phi : ndarray, float
        Tableau 3D du champ de géopotentiel.
        
    Sorties :
    ------------
    u : ndarray, float
        Tableau 3D du champ de vent zonal.
    v : ndarray, float
        Tableau 3D du champ de vent méridien.
    """
    # Récupération des dimensions du champ de points de grille
    dim = phi.shape
    nlat = dim[1]  # nombre de niveaux en latitude
    nlon = dim[2]  # nombre de niveaux en longitude
    
    # Création/initialisation des champs de vents zonal et méridien
    u = np.zeros(dim)
    v = np.zeros(dim)
    
    # Calcul des champs de vents avec les formules :
    # u = -1/f*dphi/dy et v = 1/f*dphi/dx
    u[:, 1:nlat-1, 1:nlon-1] = (-(phi[:, 0:nlat-2, 1:nlon-1] - 
    phi[:, 2:nlat, 1:nlon-1]) / (2 * DY * F))
    v[:, 1:nlat-1, 1:nlon-1] = ((phi[:, 1:nlat-1, 2:nlon] - 
    phi[:, 1:nlat-1, 0:nlon-2]) / (2 * DX * F))
    
    return u, v
    
def divq(u, v, t) :
    """
    Retourne le champ de divergence du vecteur Q calculé à partir des champs de
    vent et de température.
    
    Paramètres :
    ------------
    u : ndarray, float
        Tableau 3D du champ de vent zonal.
    v : ndarray, float
        Tableau 3D du champ de vent méridien.
    t : ndarray, float
        Tableau 3D du champ de température.
    
    Sortie :
    ------------
    div : ndarray, float
        Tableau 3D du champ de divergence du vecteur Q.
    """
    # Récupération des dimensions du champ de points de grille
    dim = u.shape
    nlat = dim[1]  # nombre de niveaux en latitude
    nlon = dim[2]  # nombre de niveaux en longitude
    
    # Création/initialisation des champs de Q1, Q2 et divergence de Q
    q1 = np.zeros(dim)
    q2 = np.zeros(dim)
    div = np.zeros(dim)
    
    # Calcul des champs de Q1 et Q2 à partir des formules :
    # Q1 = -g/theta0*(dt/dy*dv/dx + dt/dx*du/dx)
    # Q2 = -g/theta0*(dt/dx*du/dy + dt/dy*dv/dy)
    q1[:, 1:nlat-1, 1:nlon-1] = ((t[:, 0:nlat-2, 1:nlon-1] - 
    t[:, 2:nlat, 1:nlon-1]) / (2 * DY) * (v[:, 1:nlat-1, 2:nlon] - 
    v[:, 1:nlat-1, 0:nlon-2]) / (2 * DX) + (t[:, 1:nlat-1, 2:nlon] - 
    t[:, 1:nlat-1, 0:nlon-2]) / (2 * DX) * (u[:, 1:nlat-1, 2:nlon] - 
    u[:, 1:nlat-1, 0:nlon-2]) / (2 * DX))
    q2[:, 1:nlat-1, 1:nlon-1] = ((t[:, 1:nlat-1, 2:nlon] - 
    t[:, 1:nlat-1, 0:nlon-2]) / (2 * DX) * (u[:, 0:nlat-2, 1:nlon-1] - 
    u[:, 2:nlat, 1:nlon-1]) / (2 * DY) + (t[:, 0:nlat-2, 1:nlon-1] - 
    t[:, 2:nlat, 1:nlon-1]) / (2 * DY) * (v[:, 0:nlat-2, 1:nlon-1] - 
    v[:, 2:nlat, 1:nlon-1]) / (2 * DY))
    
    # Calcul de la divergence de Q : divQ = dq1/dx + dq2/dy
    div[:, 1:nlat-1, 1:nlon-1] = ((q1[:, 1:nlat-1, 2:nlon] - 
    q1[:, 1:nlat-1, 0:nlon-2]) / (2 * DX) + (q2[:, 0:nlat-2, 1:nlon-1] - 
    q2[:, 2:nlat, 1:nlon-1]) / (2 * DY))
    
    return div
    
def theta(phi) :
    """
    Retourne le champ de température potentielle calculé à partir du champ de
    géopotentiel.
    
    Paramètre :
    ------------
    phi : ndarray, float
        Tableau 3D du champ de géopotentiel.
    
    Sortie :
    ------------
    th : ndarray, float
        Tableau 3D du champ de température potentielle.
    """
    # Récupération des dimensions du champ de points de grille
    dim = phi.shape
    nz = dim[0]    # nombre de niveaux verticaux
    
    # Création/initialisation du champ de température potentielle
    th = np.zeros(dim)
    
    # Calcul du champ de température potentielle à partir de la formule :
    # theta = -1/R*dphi/d(ln(P))*(P0/P)^(R/Cp)
    for niv in range(1, nz-1) :
        th[niv, :, :] = (-COEFF_T_THETA[niv] / RA * (phi[niv+1, :, :] - 
        phi[niv-1, :, :]) / (LN_P[niv+1] - LN_P[niv-1]))
    
    return th
    
def tpqg(phi) :
    """
    Retourne le champ de tourbillon potentiel quasi-géostrophique calculé à
    partir du champ de géopotentiel.
    
    Paramètre :
    ------------
    phi : ndarray, float
        Tableau 3D du champ de géopotentiel.
    
    Sortie :
    ------------
    q : ndarray, float
        Tableau 3D du champ de tourbillon potentiel quasi-géostrophique.
    """
    # Récupération des dimensions du champ de points de grille
    dim = phi.shape
    nz = dim[0]    # nombre de niveaux verticaux
    nlat = dim[1]  # nombre de niveaux en latitude
    nlon = dim[2]  # nombre de niveaux en longitude

    # Création/initialisation du champ de tpqg
    q = np.zeros(dim)
    
    # Calcul du tpqg avec la formule :
    # q = 1/f*lap(phi) + f*d/dp(1/sigma*dphi/dp)
    for niv in range(2, nz-2) :
        coef_pmoins = (F / (SIGMA[niv-1] * 
        (NIV_P[niv+1] - NIV_P[niv-1]) * (NIV_P[niv] - NIV_P[niv-1])))
        coef_pplus = (F / (SIGMA[niv+1] * 
        (NIV_P[niv+1] - NIV_P[niv-1]) * (NIV_P[niv+1] - NIV_P[niv])))
        
        q[niv, 1:nlat-1, 1:nlon-1] = ((((phi[niv, 0:nlat-2, 1:nlon-1] +
        phi[niv, 2:nlat, 1:nlon-1] - 2 * phi[niv, 1:nlat-1, 1:nlon-1]) / (
        DY ** 2) + (phi[niv, 1:nlat-1, 0:nlon-2] + phi[niv, 1:nlat-1, 2:nlon]
        - 2 * phi[niv, 1:nlat-1, 1:nlon-1]) / (DX ** 2)) / F) + coef_pplus * (
        phi[niv+1, 1:nlat-1, 1:nlon-1] - phi[niv, 1:nlat-1, 1:nlon-1]) +
        coef_pmoins * (phi[niv, 1:nlat-1, 1:nlon-1] -
        phi[niv-1, 1:nlat-1, 1:nlon-1]))
    
    return q
    
def inv_omega(u, v, t, niter, w) :
    """
    Retourne le champ de vitesse verticale omega calculé à partir des champs de
    vent et de température.
    On inverse le laplacien par la méthode de surrelaxation.
    
    Paramètres :
    ------------
    u : ndarray, float
        Tableau 3D du champ de vent zonal.
    v : ndarray, float
        Tableau 3D du champ de vent méridien.
    t : ndarray, float
        Tableau 3D du champ de température.
    niter : int
        Nombre d'itérations de surrelaxation à faire.
    w : float
        Coefficient de relaxation.
        
    Sortie :
    ------------
    omega : ndarray, float
        Tableau 3D du champ de géopotentiel.
    """
    # Récupération des dimensions du champ de points de grille
    dim = u.shape
    nz = dim[0]    # nombre de niveaux verticaux
    nlat = dim[1]  # nombre de niveaux en latitude
    nlon = dim[2]  # nombre de niveaux en longitude
    
    # Niveau pour le test de la bonne convergence de la méthode
    #niv_test = 9
    
    # Création/initialisation des champs et vecteurs
    omega = np.zeros(dim)
    omega_temp = np.zeros(dim)  # champ intermédiaire utilisé pour le calcul
    mb_droite = np.zeros(dim)
    
    # Formation du membre de droite dans l'équation : 2R/(sigma*P)divQ
    div = divq(u, v, t)
    div = liss2d(div, 10)
    for niv in range(1,nz-1) :
        mb_droite[niv, :, :] = (2 * div[niv, :, :] * RA * (DY ** 2) /
                                (SIGMA[niv] * NIV_P[niv]))    
	
    # Calcul du champ de omega par niter itérations de surrelaxation à
    # partir de la formule : omega(x,y,z,n+1) = (1-w)*omega(x,y,z,n) +
    # w/(2/C²+2+coef_pplus+coef_pmoins)*((omega(x-dx,y,z,n)+omega(x+dx,y,z,n))
    # /C² + omega(x,y-dy,z,n) + omega(x,y+dy,z,n) + coef_pmoins*omega(x,y,z-dz)
    # + coef_pplus*omega(x,y,z+dz) - mb_droite(x,y,z))
    for i in range(niter) :
        for niv in range(1, nz-2) :
            # Ecriture des coefficients pour la coordonnée verticale
            coef_pmoins = ((DY ** 2) * (F ** 2) * 2 / (SIGMA[niv] * (NIV_P[niv]
            - NIV_P[niv-1]) * (NIV_P[niv+1] - NIV_P[niv-1])))
            coef_pplus = ((DY ** 2) * (F ** 2) * 2 / (SIGMA[niv+1] *
            (NIV_P[niv+1] - NIV_P[niv]) * (NIV_P[niv+1] - NIV_P[niv-1])))
            
            omega[niv, 1:nlat-1, 1:nlon-1] = (
            (1 - w) * omega_temp[niv, 1:nlat-1, 1:nlon-1] + w /(2 / (C ** 2) +
            2 + coef_pplus + coef_pmoins) * ((
            omega_temp[niv, 1:nlat-1, 0:nlon-2] +
            omega_temp[niv, 1:nlat-1, 2:nlon]) / (C ** 2) + 
            omega_temp[niv, 0:nlat-2, 1:nlon-1] + 
            omega_temp[niv, 2:nlat, 1:nlon-1] + 
            coef_pmoins * omega_temp[niv-1, 1:nlat-1, 1:nlon-1] + 
            coef_pplus * omega_temp[niv+1, 1:nlat-1, 1:nlon-1] - 
            mb_droite[niv, 1:nlat-1, 1:nlon-1]))
		
        # Pour vérifier que la méthode converge bien, on calcule le laplacien
        # moins le membre de droite au niveau 600 hPa à chaque étape et on
        # affiche le max de cette valeur qui doit tendre vers 0
        #eq_omega = ((omega[niv_test, 1:nlat-1, 0:nlon-2] + 
        #omega[niv_test, 1:nlat-1, 2:nlon] - 
        #2*omega[niv_test, 1:nlat-1, 1:nlon-1])/(C**2) + 
        #omega[niv_test, 0:nlat-2, 1:nlon-1] + 
        #omega[niv_test, 2:nlat, 1:nlon-1]
        #- 2*omega[niv_test, 1:nlat-1, 1:nlon-1]	+ coef_pplus*(
        #omega[niv_test+1, 1:nlat-1, 1:nlon-1] -
        #omega[niv_test, 1:nlat-1, 1:nlon-1]) - coef_pmoins*(
        #omega[niv_test, 1:nlat-1, 1:nlon-1] - 
        #omega[niv_test-1, 1:nlat-1, 1:nlon-1])
        #- mb_droite[niv_test, 1:nlat-1, 1:nlon-1])
        #print(np.max(abs(eq_omega)))
		
        omega_temp = omega
    
    return omega
    
def inv_tpqg(q, niter, w, bc, signebc, tilt) :
    """
    Retourne le champ de géopotentiel calculé à partir du champ de tourbillon
    potentiel quasi-géostrophique.
    On inverse le laplacien par la méthode de surrelaxation.
    
    Paramètres :
    ------------
    q : ndarray, float
        Tableau 3D du champ de tourbillon potentiel quasi-géostrophique.
    niter : int
        Nombre d'itérations de surrelaxation à faire.
    w : float
        Coefficient de relaxation.
    bc : boolean
        Variable pour choisir si on veut ajouter une anomalie de basses couches
        entre True et False (par défaut False).
    signebc : str
        Signe de l'anomalie de basses couches à ajouter : entre "+" et "-" 
        (par défaut "+").
    tilt : float
        Valeur de décalage entre l'anomalie d'altitude et l'anomalie de basses
        couches en degrés (par défaut 0). Le tilt est un multiple de 0.5.
        
    Sortie :
    ------------
    phi : ndarray, float
        Tableau 3D du champ de géopotentiel.
    """
    # Récupération des dimensions du champ de points de grille
    dim = q.shape
    nz = dim[0]    # nombre de niveaux verticaux
    nlat = dim[1]  # nombre de niveaux en latitude
    nlon = dim[2]  # nombre de niveaux en longitude
    lon0 = (LON_MAX + LON_MIN) / 2
    
    # Création/initialisation du champ de géopotentiel
    phi = np.zeros(dim)
    phi_temp = np.zeros(dim)  # champ intermédiaire utilisé pour le calcul
    
    th_ano = ano_theta(40, lon0 + tilt, "+")
    
    # Calcul du champ de géopotentiel par niter itérations de surrelaxation à
    # partir de la formule : phi(x,y,P,n+1) = (1-w)*phi(x,y,P,n) +
    #  w/(2/C²+2+coef_pplus+coef_pmoins)*((phi(x-dx,y,P,n)+phi(x+dx,y,P,n))/C²
    # + phi(x,y-dy,P,n) + phi(x,y+dy,P,n) + coef_pmoins*phi(x,y,P-dP) +
    # coef_pplus*phi(x,y,P+dP) - f*q(x,y,z))
    for i in range(niter) :
        #  Ecriture des coefficients pour la dérivée verticale dans l'inversion
        for niv in range(1, nz-2) :
            coef_pmoins = ((F ** 2) * (DY ** 2) / (SIGMA[niv] * 
            (NIV_P[niv+1] - NIV_P[niv]) * (NIV_P[niv] - NIV_P[niv-1])))
            coef_pplus = ((F ** 2) * (DY ** 2) / (SIGMA[niv+1] * 
            (NIV_P[niv+1] - NIV_P[niv]) * (NIV_P[niv+1] - NIV_P[niv])))
            
            phi[niv, 1:nlat-1, 1:nlon-1] = (
            (1 - w) * phi_temp[niv, 1:nlat-1, 1:nlon-1] + w / (2 / (C ** 2) + 2
            + coef_pplus + coef_pmoins) * ((phi_temp[niv, 1:nlat-1, 0:nlon-2] + 
            phi_temp[niv, 1:nlat-1, 2:nlon]) / (C ** 2) + 
            phi_temp[niv, 0:nlat-2, 1:nlon-1] + 
            phi_temp[niv, 2:nlat, 1:nlon-1] + 
            coef_pmoins * phi_temp[niv-1, 1:nlat-1, 1:nlon-1] + 
            coef_pplus * phi_temp[niv+1, 1:nlat-1, 1:nlon-1] - 
            F * (DY ** 2) * q[niv, 1:nlat-1, 1:nlon-1]))
            
        # Conditions aux limites :
        if bc and (signebc == "+" or signebc == "positif" or \
        signebc == "positive") :
            # Neumann avec une anomalie de theta positive en basses couches
            phi[0, :, :] = phi[1, :, :]
            phi[nz-1, :, :] = phi[nz-2, :, :] - th_ano * RA * (LN_P[nz-1] - 
            LN_P[nz-2]) / (COEFF_T_THETA[nz-2])
        elif bc and (signebc == "-" or signebc == "negatif" or \
        signebc == "negative" or signebc == "négative" \
        or signebc == "négatif") :
            # Neumann avec une anomalie de theta négative en basses couches
            phi[0, :, :] = phi[1, :, :]
            phi[nz-1, :, :] = phi[nz-2, :, :] + th_ano * RA * (LN_P[nz-1] - 
            LN_P[nz-2]) / (COEFF_T_THETA[nz-2])
        else :
        # Neumann homogène 
            phi[0, :, :] = phi[1, :, :]
            phi[nz-1, :, :] = phi[nz-2, :, :]
                
        # Dirichlet nul
        #phi[0, :, :] = 0
        #phi[nz-1, :, :] = 0
        
        phi_temp = phi
    
    return phi
