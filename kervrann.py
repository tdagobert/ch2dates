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
from scipy.special import factorial

def gerer_bords(img):
    """
    Remplacement des valeurs NaN situées sur les bords, par les valeurs
    situées sur la frontière.
    """
    nlig, ncol, ncan = img.shape
    for k in np.arange(ncan):
        
        # remplacement des colonnes
        for i in np.arange(nlig):
            j = 0
            while j < ncol and np.isnan(img[i, j, k]):
                j += 1
            # toute la ligne est NaN
            if j == ncol:
                continue
            # remplacement des colonnes de gauche
            img[i, 0:j, k] = img[i, j, k]

            while not np.isnan(img[i, j, k]):
                j += 1
            # remplacement des colonnes de droite
            img[i, j:ncol, k] = img[i, j-1, k]

        # remplacement des lignes
        for j in np.arange(ncol):
            i = 0
            while i < nlig and np.isnan(img[i, j, k]):
                i += 1
            # toute la colonne est NaN
            if i == nlig:
                continue
            # remplacement des lignes du haut
            img[0:i, j, k] = img[i, j, k]

            while not np.isnan(img[i, j, k]):
                i += 1
            # remplacement des colonnes de droite
            img[i:nlig, j, k] = img[i-1, j, k]
    
    return img




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
    demi_b = cfg.b // 2
    
    # résultat
    phi_uvl = np.nan * np.ones((nlig, ncol, cfg.b**2))

    # images filtrées
    u_rho = gaussian_filter(u, cfg.sigma)
    v_rho = gaussian_filter(v, cfg.sigma)

    # calcul pixellien
    for xi in np.arange(nlig):
#        print(xi)
        for xj in np.arange(ncol):

            # test aux limites
            if xi-l < 0 or nlig <= xi+l or xj-l < 0 or ncol <= xj+l:
                continue
            
            # voisinage de x
            if cfg.metrique == "l2":
                uu = u[xi-l:xi+l+1, xj-l:xj+l+1] - u_rho[xi, xj]
            elif cfg.metrique == "ratio":
                uu = u[xi-l:xi+l+1, xj-l:xj+l+1]
            elif cfg.metrique == "correlation":
                uu = u[xi-l:xi+l+1, xj-l:xj+l+1]                    
            k = 0
            for m in np.arange(-demi_b, demi_b + 1):
                for n in np.arange(-demi_b, demi_b + 1):
                    yi = xi + m
                    yj = xj + n
                    
                    # test aux limites
                    if yi-l < 0 or nlig <= yi+l or yj-l < 0 or ncol <= yj+l:
                        k += 1
                        continue
                    
                    # voisinage de y
                    vv = v[yi-l:yi+l+1, yj-l:yj+l+1]

                    # calcul de la distance
                    if not est_uu or (est_uu and not (yi == xi and yj == xj)):
                        if cfg.metrique == "l2":
                            vv = vv - v_rho[yi, yj]
                            phi_uvl[xi, xj, k] = np.sum((uu - vv)**2)
                        elif cfg.metrique == "ratio":
                            vv = vv * (u_rho[xi, xj] / v_rho[yi, yj])
                            phi_uvl[xi, xj, k] = np.sum((uu - vv)**2)
                        elif cfg.metrique == "correlation":
                            phi_uvl[xi, xj, k] = (
                                np.sum(uu * vv) /
                                (np.sqrt(np.sum(uu*uu)) * np.sqrt(np.sum(vv*vv))
                                 )
                            )
                    k += 1

    phi_uvl = gerer_bords(phi_uvl)
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
        for n in np.arange(ncan):
            iio.write(f"phi_uul_{n:03}.tif", phi_uul[:, :, n])

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
                    (lambda_n)**k / factorial(k) * np.exp(-lambda_n)
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
        "--scale", type=int, required=False, help="Nombre d'échelles."
        , default=2
    )
    parser.add_argument(
        "--b", type=int, required=False, default=3, help="Voisinage de τ."
    )
    parser.add_argument(
        "--metrique", type=str, required=False, help="Distance.",
        choices=["correlation", "l2", "ratio"], default="l2"
    )
    parser.add_argument(
        "--epsilon", type=float, required=False, default=1.0,
        help="Nombre de fausses alarmes."
    )
    parser.add_argument(
        "--sigma", type=float, required=False, default=0.8,
        help="Écart type du noyau de flou."
    )
    parser.add_argument(
        "--repout", type=str, required=False, default="./",
        help="Répertoire de sortie."
    )
    parser.add_argument(
        "--ndvi-threshold", type=float, required=False, default=0.1,
        help="NDVI seuil."
    )

    cfg = parser.parse_args()

    return cfg


def normaliser_image(rgb):
    """
    …
    """
    mini = np.min(rgb)
    maxi = np.max(rgb)
    rgb = 255 * (rgb - mini) / (maxi - mini)
    rgb = np.array(rgb, dtype=np.uint8)
    return rgb


def ndvi_index_map(cfg, img):
    """
    If the image contains 4 channels, we assume it is a Sentinel-1 image with
    the B04, B03, B02, B08 channels storage in this order. We retrieve the
    B08 to compute the NDVI index…
    """
    nlig, ncol, ncan = img.shape

    if ncan == 4:
        # we compute the NDVI index, where values stand in [-1, +1]
        ndvi = (img[:, :, 3] - img[:, :, 0]) / (img[:, :, 3] + img[:, :, 0])
        ndvi = np.expand_dims(ndvi, axis=-1)
        # we normalize
        r_can = 255 * np.ones((nlig, ncol, 1))
        g_can = 255 * (1 - (ndvi + 1) / 2)
        img_ndvi = np.concatenate((g_can, r_can, g_can), axis=2)
        img = img[:, :, 0:3]
        return img, img_ndvi, ndvi
    else:
        return img, None, None


def main():
    """
    ...
    """

    cfg = lit_parametres()

    im1 = iio.read(cfg.image1)
    im2 = iio.read(cfg.image2)

    if not exists(cfg.repout):
        os.mkdir(cfg.repout)

    im1, img_ndvi1, ndvi1 = ndvi_index_map(cfg, im1)
    im2, img_ndvi2, ndvi2 = ndvi_index_map(cfg, im2)

    nlig, ncol, _ = im1.shape
    im1 = np.mean(im1, axis=2)
    im1 = im1.reshape(nlig, ncol, 1)

    im2 = np.mean(im2, axis=2)
    im2 = im2.reshape(nlig, ncol, 1)

    nlig, ncol, ncan = im1.shape
    for n in np.arange(ncan):
        h_uv, pfal = algorithme(cfg, im1[:, :, n], im2[:, :, n], n)
        h_uv = normaliser_image(h_uv)
        iio.write(join(cfg.repout, f"huvl_c{n}.png"), h_uv)
        iio.write(join(cfg.repout, f"pfal_c{n}.png"), pfal)

    print(img_ndvi1)
    # NDVI filtering if any
#    h_uv = np.ones((nlig, ncol, 1))
    if img_ndvi1 is not None:
        iio.write(join(cfg.repout, "ndvi1.png"), img_ndvi1)
        iio.write(join(cfg.repout, "ndvi2.png"), img_ndvi2)
#        iio.write(join(cfg.repout, "ndvi2.tif"), ndvi2)
        # L'idée est de supprimer tous les changements qui sont du type
        # végétation--> végétation ou du type non-végétation--> végétation.
        # i.e. dès que im2 est végétation
        img_veget = ndvi2 > cfg.ndvi_threshold
        img_veget = img_veget.squeeze()
#        iio.write(join(cfg.repout, f"veget.tif"), img_veget)
        h_uv[img_veget] = 0
        iio.write(join(cfg.repout, f"huvl_seuille.png"), h_uv)

    return 0


if __name__ == "__main__":
    main()

    #Lignes de commandes
    # python3 kervrann.py --image1 img1.png --image2 img2.png --scale 2 --epsilon 1 --sigma 0.8 --b 3 --metrique correlation --repout mcor_s2_b3_eps1_sig0.8
    # python3 kervrann.py --image1 img1.png --image2 img2.png --scale 2 --epsilon 1 --sigma 0.8 --b 3 --metrique ratio --repout mrat_s2_b3_eps1_sig0.8
