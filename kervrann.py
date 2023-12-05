"""
…
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import argparse
import numpy as np
import iio
from scipy.ndimage import gaussian_filter

def calculer_phi(u, v, l, search_side, sigma):
    """
    D'après la formule (2.2).
    Paramètres
    ----------
    u : np.array ndim=(nlig, ncol, 1)
        Image de référence.
    v : np.array ndim=(nlig, ncol, 1)
        Image de comparaison.
    l : int
        Demi-côté de la vignette carrée.
    search_side: int
        Côté de la fenêtre de recherche carrée.
    sigma : float.
        Écart type de la gaussienne.
    """

    nlig, ncol, _ = u.shape

    # résultat
    phi_uvl = np.zeros((nlig, ncol, search_side**2))

    # images filtrées
    u_rho = gaussian_filter(u, sigma)
    v_rho = gaussian_filter(v, sigma)

    # calcul pixellien
    for xi in np.arange(nlig):
        for xj in np.arange(ncol):
            try:
                uu = u[xi-l:xi-l+1, xj-l:xj+l+1] - u_rho[xi, xj]
                k = 0
                for m in np.arange(search_side):
                    for n in np.arange(search_side):
                        yi = xi + m - search_side // 2
                        yj = xj + n - search_side // 2
                        vv = v[yi-l:yi-l+1, yj-l:yj+l+1] - v_rho[yi, yj]

                        phi_uvl[xi, xj, k] = np.sum((uu - vv)**2)
                        k += 1
            except IndexError:
                pass
    return phi_uvl


def calculer_pfas(cfg, im1, im2):
    """
    Paramètres
    ----------
    cfg: Namespace
    im1: np.array(nlig, ncol, 1)
    im2: np.array(nlig, ncol, 1)

    Retour
    ------
    decisions: np.array(L, nlig, ncol)
    pfas: np.array(L, nlig, ncol)
    """

    nlig, ncol, _ = im1.shape
    pfas = []
    decisions = []
    for l in np.arange(1, cfg.scale+1):
        # calcul de φ(u, u, l)
        phi_uul = calculer_phi(im1, im1, l, cfg.b, cfg.sigma)

        # calcul de φ(u, v, l)
        phi_uvl = calculer_phi(im1, im2, l, cfg.b, cfg.sigma)

        # calcul de τ_mean(l) d'après (5.1)
        tau_l_mean = []
        for i in np.arange(nlig):
            for j in np.arange(ncol):
                tau_l_mean += [
                    np.min(phi_uul[i-cfg.b:i+cfg.b, j-cfg.b:j+cfg.b])
                ]
        tau_l_mean = np.mean(np.array(tau_l_mean))

        # calcul de τ(u, l) d'après (5.1)
        tau_ul = np.zeros((nlig, ncol))
        for i in np.arange(nlig):
            for j in np.arange(ncol):
                tau_ul[i, j] = np.max(
                    (
                        np.max(phi_uul[i-cfg.b:i+cfg.b, j-cfg.b:j+cfg.b]),
                        tau_l_mean
                    )
                )
        # calcul de S_Nl
        S_Nl = np.zeros((nlig, ncol))
        for i in np.arange(nlig):
            for j in np.arange(ncol):
                S_Nl[i, j] = np.sum(
                    phi_uvl[i-cfg.B:i+cfg.B, j-cfg.B:j+cfg.B] >= tau_ul[i, j]
                )
        # calcul de decision_l d'après (4.1)
        decision_l = np.uint8(S_Nl == (cfg.B*cfg.B))
        decisions += [decision_l]

        # calcul de pfa_l
        pfa_l = 1.0 / (nlig * ncol) * np.sum(np.exp(S_Nl-cfg.B*cfg.B))
        pfas += [pfa_l]

    decisions = np.array(decisions)
    pfas = np.array(pfas)
    return pfas, decisions


def calculer_pfal(kd, lambda_n, nlig, ncol):
    """
    …
    """
    pfal = np.zeros((nlig, ncol))

    for i in np.arange(nlig):
        for j in np.arange(ncol):
            for k in np.arange(kd[i, j] + 1):
                pfal[i, j] += (
                    (lambda_n[i, j])**k / np.fact(k) * np.exp(-lambda_n[i, j])
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
    im1: np.array ndim=(nlig, ncol, 1)
    im2: np.array ndim=(nlig, ncol, 1)
    """
    nlig, ncol, _ = im1.shape
    pfas, decisions = calculer_pfas(cfg, im1, im2)
    lambda_n = np.sum(np.array(pfas))

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
    )
    parser.add_argument(
        "--b", type=int, required=True, help="Voisinage de τ."
    )
    parser.add_argument(
        "--B", type=int, required=True, help="Voisinage de recherche."
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
