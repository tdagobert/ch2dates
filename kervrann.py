#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""
…
"""

import os
from os.path import exists, join
import argparse
import numpy as np
import iio
from scipy.ndimage import gaussian_filter

def calculer_phi(u, v, l, cfg, est_uu=False):
    """
    D'après la formule (2.2).
    Paramètres
    ----------
    u : np.array ndim=(nlig, ncol)
        Image de référence.
    v : np.array ndim=(nlig, ncol)
        Image de comparaison.
    l : int
        Demi-côté de la vignette carrée.
    cfg : Namespace
    """

    nlig, ncol = u.shape

    # résultat
    phi_uvl = np.nan * np.ones((nlig, ncol, cfg.b**2))

    # images filtrées
    u_rho = gaussian_filter(u, cfg.sigma)
    v_rho = gaussian_filter(v, cfg.sigma)

    # calcul pixellien
    for xi in np.arange(nlig):
#        print(xi)
        for xj in np.arange(ncol):
            try:
                if cfg.metrique == "l2":
                    uu = u[xi-l:xi+l+1, xj-l:xj+l+1] - u_rho[xi, xj]
                elif cfg.metrique == "ratio":
                    uu = u[xi-l:xi+l+1, xj-l:xj+l+1]
                elif cfg.metrique == "correlation":
                    uu = u[xi-l:xi+l+1, xj-l:xj+l+1]                    
                k = 0
                for m in np.arange(cfg.b):
                    for n in np.arange(cfg.b):
                        yi = xi + m - cfg.b // 2
                        yj = xj + n - cfg.b // 2
                        if not est_uu or (est_uu and not (yi == xi and yj == xj)):
                            if cfg.metrique == "l2":
                                vv = v[yi-l:yi+l+1, yj-l:yj+l+1] - v_rho[yi, yj]
                                phi_uvl[xi, xj, k] = np.sum((uu - vv)**2)
                            elif cfg.metrique == "ratio":
                                vv = v[yi-l:yi+l+1, yj-l:yj+l+1] * (u_rho[xi, xj] / v_rho[yi, yj])
                                phi_uvl[xi, xj, k] = np.sum((uu - vv)**2)
                            elif cfg.metrique == "correlation":
                                vv = v[yi-l:yi+l+1, yj-l:yj+l+1]
                                phi_uvl[xi, xj, k] = np.sum(uu * vv) / (np.sqrt(np.sum(uu*uu)) * np.sqrt(np.sum(vv*vv)))
                        k += 1
#                        if xi == 10 and xj == 10:
#                            print(f"{xi} {xj} {yi} {yj}")
 #               exit()
            except ValueError:
                pass
    print("calculer phi")
    return phi_uvl


def calculer_pfas(cfg, im1, im2, ican):
    """
    Paramètres
    ----------
    cfg: Namespace
    im1: np.array(nlig, ncol)
    im2: np.array(nlig, ncol)
    ican: int
    Retour
    ------
    decisions: np.array(L, nlig, ncol)
    pfas: np.array(L, nlig, ncol)
    """

    nlig, ncol = im1.shape
    pfas = []
    decisions = []
    for l in np.arange(1, cfg.scale+1):
        print(f"Échelle {l}")
        # calcul de φ(u, u, l)
        phi_uul = calculer_phi(im1, im1, l, cfg, est_uu=True)
        print(phi_uul.shape)
        nlig, ncol, ncan = phi_uul.shape
#        for n in np.arange(ncan):
#            iio.write(f"phi_uul_{n:03}.tif", phi_uul[:, :, n])

        # calcul de φ(u, v, l)
        phi_uvl = calculer_phi(im1, im2, l, cfg)

        # calcul de τ_mean(l) d'après (5.1)
        tau_l_mean = []
        for i in np.arange(nlig):
            for j in np.arange(ncol):
                try:
                    # recherche du minimum sur le voisinage b(x)
                    tau_l_mean += [np.nanmin(phi_uul[i, j, :])]
                except ValueError:
                    pass
        tau_l_mean = np.nanmean(np.array(tau_l_mean))
        print(f"# calcul de τ_mean(l) d'après (5.1) {tau_l_mean:3.5e}")

        # calcul de τ(u, l) d'après (5.1)
        tau_ul = np.zeros((nlig, ncol))
        for i in np.arange(nlig):
            for j in np.arange(ncol):
                try:
                    tau_ul[i, j] = np.nanmax(
                        (np.nanmax(phi_uul[i, j, :]), tau_l_mean)
                    )
                    
                except ValueError:
                    pass
#        iio.write(join(cfg.repout, f"tau_ul_s{l}_c{ican}.tif"), tau_ul)
        print("# calcul de τ(u, l) d'après (5.1)")
        # calcul de S_Nl
        S_Nl = np.zeros((nlig, ncol))
        for i in np.arange(nlig):
            for j in np.arange(ncol):
                try:
                    S_Nl[i, j] = np.sum(phi_uvl[i, j, :] >= tau_ul[i, j])
                except ValueError:
                    pass
#        iio.write(join(cfg.repout, f"snl{l}_c{ican}.tif"), S_Nl)
        # calcul de decision_l d'après (4.1)
        decision_l = np.uint8(S_Nl == (cfg.b * cfg.b))
        decisions += [decision_l]

        # calcul de pfa_l
        pfa_l =  np.nanmean(np.exp(S_Nl - (cfg.b * cfg.b)))
        pfas += [pfa_l]

    decisions = np.array(decisions)
    pfas = np.array(pfas)
    return pfas, decisions


def calculer_pfal(kd, lambda_n, nlig, ncol):
    """
    lambda_n : float
    """
    pfal = np.zeros((nlig, ncol))

    for i in np.arange(nlig):
        for j in np.arange(ncol):
            for k in np.arange(kd[i, j] + 1):
#                print(k, lambda_n)
                pfal[i, j] += (
                    (lambda_n)**k / np.math.factorial(k) * np.exp(-lambda_n)
                )
            pfal[i, j] = 1 - pfal[i, j]
    return pfal


def calculer_alpha(epsilon, nlig, ncol, pfal):
    """
    D'après la formule de l'algorithme.
    """
    alpha = np.max((epsilon/(nlig*ncol), np.min(pfal)))
    return alpha


def algorithme(cfg, im1, im2, ican):
    """
    cfg: Namespace
    im1: np.array ndim=(nlig, ncol)
    im2: np.array ndim=(nlig, ncol)
    ican : int
        Index du canal.
    """
    nlig, ncol = im1.shape
    pfas, decisions = calculer_pfas(cfg, im1, im2, ican)
    lambda_n = np.sum(np.array(pfas))
    print(f"lambda_n {lambda_n}")
    # calcul de kd
    kd = np.sum(decisions, axis=0)

    # calcul de P_FA(x, L) pour tout x
    pfal = calculer_pfal(kd, lambda_n, nlig, ncol)

    # calcul de α
    alpha = calculer_alpha(cfg.epsilon, nlig, ncol, pfal)

    # test d'hypothèse
    h_uv = np.uint8(pfal <= alpha)
    return h_uv, pfal


def lit_parametres():
    """
    …
    """
    d = "Compute the changes between two images."
    parser = argparse.ArgumentParser(description=d)
    parser.add_argument(
        "--image1", type=str, required=True, help="First image."
    )
    parser.add_argument(
        "--image2", type=str, required=True, help="Second image."
    )
    parser.add_argument(
        "--scale", type=int, required=True, help="Nombre d'échelles."
        , default=2
    )
    parser.add_argument(
        "--b", type=int, required=True, default=3, help="Voisinage de τ."
    )
    parser.add_argument(
        "--metrique", type=str, required=False, help="Distance.",
        choices=["correlation", "l2", "ratio"], default="l2"
    )    
    parser.add_argument(
        "--epsilon", type=float, required=True, default=1.0,
        help="Nombre de fausses alarmes."
    )
    parser.add_argument(
        "--sigma", type=float, required=True, default=0.8,
        help="Écart type du noyau de flou."
    )
    parser.add_argument(
        "--repout", type=str, required=True, default="./",
        help="Répertoire de sortie."
    )

    cfg = parser.parse_args()

    return cfg


def main():
    """
    ...
    """

    cfg = lit_parametres()
    return 0
    im1 = iio.read(cfg.image1)
    im2 = iio.read(cfg.image2)

    if not exists(cfg.repout):
        os.mkdir(cfg.repout)
        
    nlig, ncol, ncan = im1.shape
    im1 = np.mean(im1, axis=2)
    im1 = im1.reshape(nlig, ncol, 1)
    im2 = np.mean(im2, axis=2)
    im2 = im2.reshape(nlig, ncol, 1)
    
    for n in np.arange(ncan):
        h_uv, pfal = algorithme(cfg, im1[:, :, n], im2[:, :, n], n)
        iio.write(join(cfg.repout, f"huvl_c{n}.png"), h_uv)
        iio.write(join(cfg.repout, f"pfal_c{n}.png"), pfal)
    return 0


if __name__ == "__main__":
    main()

    #Lignes de commandes 
    # python3 kervrann.py --image1 img1.png --image2 img2.png --scale 2 --epsilon 1 --sigma 0.8 --b 3 --metrique correlation --repout mcor_s2_b3_eps1_sig0.8
    # python3 kervrann.py --image1 img1.png --image2 img2.png --scale 2 --epsilon 1 --sigma 0.8 --b 3 --metrique ratio --repout mrat_s2_b3_eps1_sig0.8
