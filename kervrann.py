#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""
…
"""

import argparse
import numpy as np
import iio
from scipy.ndimage import gaussian_filter

def calculer_phi(u, v, l, search_side, sigma, est_uu=False):
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
    search_side: int
        Côté de la fenêtre de recherche carrée.
    sigma : float.
        Écart type de la gaussienne.
    """

    nlig, ncol = u.shape

    # résultat
    phi_uvl = np.nan * np.ones((nlig, ncol, search_side**2))

    # images filtrées
    u_rho = gaussian_filter(u, sigma)
    v_rho = gaussian_filter(v, sigma)

    # calcul pixellien
    for xi in np.arange(nlig):
#        print(xi)
        for xj in np.arange(ncol):
            try:
                uu = u[xi-l:xi+l+1, xj-l:xj+l+1] - u_rho[xi, xj]
 #               print(u.shape, "uu.shape", uu.shape, "l", l, xi, xj, xi-l,xi+l+1, xj-l, xj+l+1)
                k = 0
                for m in np.arange(search_side):
                    for n in np.arange(search_side):
                        yi = xi + m - search_side // 2
                        yj = xj + n - search_side // 2
                        if not est_uu or (est_uu and not (yi == xi and yj == xj)):
                            vv = v[yi-l:yi+l+1, yj-l:yj+l+1] - v_rho[yi, yj]
#                        print("vv.shape", vv.shape)
                            phi_uvl[xi, xj, k] = np.sum((uu - vv)**2)
                        k += 1
#                        if xi == 10 and xj == 10:
#                            print(f"{xi} {xj} {yi} {yj}")
 #               exit()
            except ValueError:
                pass
    print("calculer phi")
    return phi_uvl


def calculer_pfas(cfg, im1, im2):
    """
    Paramètres
    ----------
    cfg: Namespace
    im1: np.array(nlig, ncol)
    im2: np.array(nlig, ncol)

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
        phi_uul = calculer_phi(im1, im1, l, cfg.b, cfg.sigma, est_uu=True)
        print(phi_uul.shape)
        nlig, ncol, ncan = phi_uul.shape
        for n in np.arange(ncan):
            iio.write(f"phi_uul_{n:03}.tif", phi_uul[:, :, n])

        # calcul de φ(u, v, l)
        phi_uvl = calculer_phi(im1, im2, l, cfg.b, cfg.sigma)

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
        iio.write(f"tau_ul{l}.tif", tau_ul)
        print("# calcul de τ(u, l) d'après (5.1)")
        # calcul de S_Nl
        S_Nl = np.zeros((nlig, ncol))
        for i in np.arange(nlig):
            for j in np.arange(ncol):
                try:
                    S_Nl[i, j] = np.sum(phi_uvl[i, j, :] >= tau_ul[i, j])
                except ValueError:
                    pass
        iio.write(f"snl{l}.tif", S_Nl)
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


def algorithme(cfg, im1, im2):
    """
    cfg: Namespace
    im1: np.array ndim=(nlig, ncol)
    im2: np.array ndim=(nlig, ncol)
    """
    nlig, ncol = im1.shape
    pfas, decisions = calculer_pfas(cfg, im1, im2)
    lambda_n = np.sum(np.array(pfas))
    print(f"lambda_n {lambda_n}")
    # calcul de kd
    kd = np.sum(decisions, axis=0)

    # calcul de P_FA(x, L) pour tout x
    pfal = calculer_pfal(kd, lambda_n, nlig, ncol)
    iio.write(f"pfal.tif", pfal)
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
    )
    parser.add_argument(
        "--b", type=int, required=True, help="Voisinage de τ."
    )
    parser.add_argument(
        "--epsilon", type=float, required=True,
        help="Nombre de fausses alarmes."
    )
    parser.add_argument(
        "--sigma", type=float, required=True,
        help="Ecart type du noyau de flou."
    )

    cfg = parser.parse_args()

    return cfg


def main():
    """
    ...
    """

    cfg = lit_parametres()
    im1 = iio.read(cfg.image1)
    im2 = iio.read(cfg.image2)

    _, _, ncan = im1.shape
    for n in np.arange(ncan):
        h_uv, pfal = algorithme(cfg, im1[:, :, n], im2[:, :, n])
        iio.write(f"huvl_{n}.tif", h_uv)
        iio.write(f"pfal_{n}.tif", pfal)
    return 0


if __name__ == "__main__":
    main()
