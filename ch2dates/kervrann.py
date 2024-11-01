#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""
…
"""
import os
from os.path import exists, join, basename, dirname
import timeit
import argparse
import zipfile
from ast import literal_eval
import multiprocessing
from joblib import Parallel, delayed

import numpy as np
import iio
from scipy.ndimage import gaussian_filter
from scipy.special import factorial
from matplotlib import cm

from numba import njit

@njit
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


@njit
def calculer_phi(
        imu, imv, imu_rho, imv_rho, l, b, metrique, est_uu=False
):
    """
    D'après la formule (2.2).
    Paramètres
    ----------
    imu : np.array ndim=(nlig, ncol)
        Image de référence.
    imv : np.array ndim=(nlig, ncol)
        Image de comparaison.
    l : int
        Demi-côté de la vignette carrée.
    cfg : Namespace
    """

    nlig, ncol = imu.shape
    demi_b = b // 2

    # résultat
    phi_uvl = np.nan * np.ones((nlig, ncol, b**2))

    # calcul pixellien
    for x_i in np.arange(nlig):
#        print(x_i)
        for x_j in np.arange(ncol):

            # test aux limites
            if x_i-l < 0 or nlig <= x_i+l or x_j-l < 0 or ncol <= x_j+l:
                continue

            # voisinage de x
            if metrique == "l2":
                uu = imu[x_i-l:x_i+l+1, x_j-l:x_j+l+1] - imu_rho[x_i, x_j]
            elif metrique == "ratio":
                uu = imu[x_i-l:x_i+l+1, x_j-l:x_j+l+1]
            elif metrique == "correlation":
                uu = imu[x_i-l:x_i+l+1, x_j-l:x_j+l+1]
            elif metrique == "lin":
                uu = imu[x_i-l:x_i+l+1, x_j-l:x_j+l+1]
            elif metrique == "zncc":
                uu = imu[x_i-l:x_i+l+1, x_j-l:x_j+l+1]
                muu = np.mean(uu)
            k = 0
            for m in np.arange(-demi_b, demi_b + 1):
                for n in np.arange(-demi_b, demi_b + 1):
                    y_i = x_i + m
                    y_j = x_j + n

                    # test aux limites
                    if y_i-l < 0 or nlig <= y_i+l or y_j-l < 0 or ncol <= y_j+l:
                        k += 1
                        continue

                    # voisinage de y
                    vv = imv[y_i-l:y_i+l+1, y_j-l:y_j+l+1]

                    # calcul de la distance
                    if not est_uu or (est_uu and not (y_i == x_i and y_j == x_j)):
                        if metrique == "l2":
                            vv = vv - imv_rho[y_i, y_j]
                            phi_uvl[x_i, x_j, k] = np.sum((uu - vv)**2)
                        elif metrique == "ratio":
                            vv = vv * (imu_rho[x_i, x_j] / imv_rho[y_i, y_j])
                            phi_uvl[x_i, x_j, k] = np.sum((uu - vv)**2)
                        elif metrique == "lin":
                            suu = np.sum(uu*uu)
                            svv = np.sum(vv*vv)
                            phi_uvl[x_i, x_j, k] = (
                                max(suu, svv)
                                * (1 - np.sum(uu * vv)**2 / (suu * svv))
                            )
                        elif metrique == "correlation":
                            phi_uvl[x_i, x_j, k] = (
                                1
                                - np.sum(uu * vv) /
                                (np.sqrt(np.sum(uu*uu)) * np.sqrt(np.sum(vv*vv))
                                 )
                            )
                        elif metrique == "zncc":
                            mvv = np.mean(vv)
                            phi_uvl[x_i, x_j, k] = (
                                1
                                - np.sum((uu - muu) * (vv - mvv))
                                /(vv.size * np.std(uu) * np.std(vv))
                            )

                    k += 1

    phi_uvl = gerer_bords(phi_uvl)
    return phi_uvl


def calculer_pfas(cfg, im1, im2, ante1, ante2, ican, ref_ident):
    """
    Paramètres
    ----------
    cfg: Namespace
    im1: np.array(nlig, ncol)
    im2: np.array(nlig, ncol)
    ante1: np.array ndim=(nlig, ncol)
    ante2: np.array ndim=(nlig, ncol)
    ican: int
    Retour
    ------
    decisions: np.array(L, nlig, ncol)
    pfas: np.array(L, nlig, ncol)
    """


    nlig, ncol = im1.shape
    pfas = []
    decisions = []
    im1_rho = gaussian_filter(im1, cfg.sigma)
    im2_rho = gaussian_filter(im2, cfg.sigma)
    ante1_rho = gaussian_filter(ante1, cfg.sigma)
    ante2_rho = gaussian_filter(ante2, cfg.sigma)
#    print(ante1 == ante2)
#    print(np.array_equal(ante1, ante2))

    for l in np.arange(1, cfg.scale+1):
#    for l in [cfg.scale]:
        print(f"Échelle {l}")
        # calcul de φ(u, u, l)
        phi_uul = calculer_phi(
            ante1, ante2, ante1_rho, ante2_rho, l, cfg.b,
            cfg.metrique, est_uu=ref_ident
        )
        #print(phi_uul.shape)
        nlig, ncol, ncan = phi_uul.shape

        if cfg.debug:
            for n in np.arange(ncan):
                iio.write(
                    join(cfg.repout, f"phi_uul{l}_{n:03}.tif"), phi_uul[:, :, n]
                )

        # calcul de φ(u, v, l)
        phi_uvl = calculer_phi(
            im1, im2, im1_rho, im2_rho, l, cfg.b, cfg.metrique
        )
        if cfg.debug:
            for n in np.arange(ncan):
                iio.write(
                    join(cfg.repout, f"phi_uvl{l}_{n:03}.tif"), phi_uvl[:, :, n]
                )

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

#        exit()
#        tau_l_mean =compute_tau_l_mean(phi_uul)
        print(f"# calcul d'après (5.1) de τ_mean({l}) {tau_l_mean:3.5e}")

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
        if cfg.debug:
            iio.write(join(cfg.repout, f"tau_ul_s{l}_c{ican}.tif"), tau_ul)

        print("# calcul de τ(u, l) d'après (5.1)")
        # calcul de S_Nl
        S_Nl = np.zeros((nlig, ncol))
        for i in np.arange(nlig):
            for j in np.arange(ncol):
                try:
                    S_Nl[i, j] = np.sum(phi_uvl[i, j, :] >= tau_ul[i, j])
                except ValueError:
                    pass
        if cfg.debug:
            iio.write(join(cfg.repout, f"snl{l}_c{ican}.tif"), S_Nl)

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


def algorithme(cfg, im1, im2, ante1, ante2, ican, ref_ident):
    """
    cfg: Namespace
    im1: np.array ndim=(nlig, ncol)
    im2: np.array ndim=(nlig, ncol)
    ante1: np.array ndim=(nlig, ncol)
    ante2: np.array ndim=(nlig, ncol)
    ican : int
        Index du canal.
    """

    nlig, ncol = im1.shape
    pfas, decisions = calculer_pfas(cfg, im1, im2, ante1, ante2, ican, ref_ident)
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
    subparsers = parser.add_subparsers(dest="monaction")

    a_parser = subparsers.add_parser("apparier", help="Entre deux images.")
    a_parser.add_argument(
        "--image1", type=str, required=True, help="First image."
    )
    a_parser.add_argument(
        "--image2", type=str, required=True, help="Second image."
    )
    a_parser.add_argument(
        "--ante1", type=str, required=True, help="Ante First image."
    )
    a_parser.add_argument(
        "--ante2", type=str, required=True, help="Ante Second image."
    )
    a_parser.add_argument(
        "--scale", type=int, required=False, help="Nombre d'échelles."
        , default=2
    )
    a_parser.add_argument(
        "--b", type=int, required=False, default=3, help="Voisinage de τ."
    )
    a_parser.add_argument(
        "--metrique", type=str, required=False, help="Distance.",
        choices=["correlation", "l2", "ratio", "zncc", "lin"], default="l2"
    )
    a_parser.add_argument(
        "--epsilon", type=float, required=False, default=1.0,
        help="Nombre de fausses alarmes."
    )
    a_parser.add_argument(
        "--sigma", type=float, required=False, default=0.8,
        help="Écart type du noyau de flou."
    )
    a_parser.add_argument(
        "--repout", type=str, required=False, default="./",
        help="Répertoire de sortie."
    )
    a_parser.add_argument(
        "--ndvi-threshold", type=float, required=False, default=0.1,
        help="NDVI seuil."
    )
    a_parser.add_argument(
        "--ndwi-threshold", type=float, required=False, default=0.5,
        help="NDWI seuil."
    )
    a_parser.add_argument(
        "--debug", type=bool, required=False, default=False,
        help="NDWI seuil."
    )
    #===========================================================================
    b_parser = subparsers.add_parser(
        "tripler", help="A partir d'une archive Zip."
    )
    b_parser.add_argument(
        "--zip", type=str, required=True, help="Zip contenant les images."
    )
    b_parser.add_argument(
        "--scale", type=int, required=False, help="Nombre d'échelles."
        , default=2
    )
    b_parser.add_argument(
        "--b", type=int, required=False, default=3, help="Voisinage de τ."
    )
    b_parser.add_argument(
        "--metrique", type=str, required=False, help="Distance.",
        choices=["correlation", "l2", "ratio", "zncc", "lin"], default="lin"
    )
    b_parser.add_argument(
        "--epsilon", type=float, required=False, default=1.0,
        help="Nombre de fausses alarmes."
    )
    b_parser.add_argument(
        "--sigma", type=float, required=False, default=0.8,
        help="Écart type du noyau de flou."
    )
    b_parser.add_argument(
        "--repout", type=str, required=False, default="./",
        help="Répertoire de sortie."
    )
    b_parser.add_argument(
        "--ndvi-threshold", type=float, required=False, default=0.1,
        help="NDVI seuil."
    )
    b_parser.add_argument(
        "--ndwi-threshold", type=float, required=False, default=0.5,
        help="NDWI seuil."
    )
    b_parser.add_argument(
        "--debug", type=bool, required=False, default=False,
        help="NDWI seuil."
    )
    b_parser.add_argument(
        "--identiques", type=bool, required=False, default=False,
        help="Images identiques."
    )

    cfg = parser.parse_args()

    return cfg


def normaliser_image(img, sat=None):
    """
    …
    """
    # convertir en float
    if sat is None:
        mini = np.min(img)
        maxi = np.max(img)
    else:
        val = np.sort(img.flatten())
        mini = val[int(sat*val.size)]
        maxi = val[int((1-sat)*val.size)]
        # remplacer les valeurs < mini ou > maxi par mini et maxi ... np.clip
    img = 255 * (img - mini) / (maxi - mini)
    img[img>255.0] = 255.0
    img[img<0.0] = 0.0
    #print("shape",img.shape)

    img = np.array(img, dtype=np.uint8)
    return img


def calorifier_image(img, apply_log=True):
    """
    ...
    """
    if apply_log:
        img = np.log(img)
        mini = np.min(img)
        maxi = np.max(img)

        img = 1.0 * (img - mini) / (maxi - mini)
        img = img.squeeze()
        img = np.uint8(255.0 * cm.jet(img))
        img = img[:, :, 0:3]

    return img


def compute_index_maps(img):
    """
    If the image contains 4 channels, we assume it is a Sentinel-2 image with
    the B04, B03, B02, B08 channels storage in this order. We retrieve the
    B08 to compute the NDVI index…

    """
    _, _, ncan = img.shape

    if ncan == 4:
        # we compute the NDVI index, where values stand in [-1, +1]
        img_ndvi = (img[:, :, 3] - img[:, :, 0]) / (img[:, :, 3] + img[:, :, 0])

        # we compute the NDWI index, where values stand in [-1, +1]
        img_ndwi = (img[:, :, 1] - img[:, :, 3]) / (img[:, :, 1] + img[:, :, 3])

        img = img[:, :, 0:3]

        return img, img_ndvi, img_ndwi

    return img, None, None


def comparer_une_paire(cfg):
    """
    ...
    """
    print(cfg.debug)
    im1 = iio.read(cfg.image1)
    im2 = iio.read(cfg.image2)
    ante1 = iio.read(cfg.ante1)
    ante2 = iio.read(cfg.ante2)
    if not exists(cfg.repout):
        os.mkdir(cfg.repout)

    # suppression du canal B02 si besoin
    im1, _, _ = compute_index_maps(im1)
    im2, img_ndvi2, img_ndwi2 = compute_index_maps(im2)
    ante1, _, _ = compute_index_maps(ante1)
    ante2, _, _ = compute_index_maps(ante2)
    iio.write(
        join(cfg.repout, "ante1.png"),
        normaliser_image(np.copy(ante1), sat=0.001)
    )
    iio.write(
        join(cfg.repout, "ante2.png"),
        normaliser_image(np.copy(ante2), sat=0.001)
    )
    iio.write(
        join(cfg.repout, "im1.png"), normaliser_image(np.copy(im1), sat=0.001)
    )
    iio.write(
        join(cfg.repout, "im2.png"), normaliser_image(np.copy(im2), sat=0.001)
    )
    print(im1.shape, im2.shape, ante1.shape, ante2.shape)
    assert im1.shape == im2.shape and im1.shape == ante1.shape
    nlig, ncol, _ = im1.shape
    im1 = np.mean(im1, axis=2)
    im1 = im1.reshape(nlig, ncol, 1)

    im2 = np.mean(im2, axis=2)
    im2 = im2.reshape(nlig, ncol, 1)

    ante1 = np.mean(ante1, axis=2)
    ante1 = ante1.reshape(nlig, ncol, 1)

    ante2 = np.mean(ante2, axis=2)
    ante2 = ante2.reshape(nlig, ncol, 1)

#    im2[200, 200, 0] = 2 * im2[200, 200, 0]

    iio.write(join(cfg.repout, "ante1.tif"), ante1)
    iio.write(join(cfg.repout, "ante2.tif"), ante2)
    iio.write(join(cfg.repout, "im1.tif"), im1)
    iio.write(join(cfg.repout, "im2.tif"), im2)

    nlig, ncol, ncan = im1.shape
    for n in np.arange(ncan):
        h_uv, pfal = algorithme(
            cfg, im1[:, :, n], im2[:, :, n], ante1[:, :, n], ante2[:, :, n], n,
            cfg.identiques
        )
        h_uv = normaliser_image(h_uv)
        iio.write(join(cfg.repout, f"huvl_c{n}.png"), h_uv)
        pfal = calorifier_image(pfal)
        iio.write(join(cfg.repout, f"pfal_c{n}.png"), pfal)

    ecrire_mappes(cfg, img_ndvi2, img_ndwi2, h_uv)
    return 0


def ecrire_mappes(cfg, img_ndvi, img_ndwi, h_uv):
    """
    ...
    """

    # NDVI filtering if any
#    h_uv = np.ones((nlig, ncol, 1))
    if img_ndvi is not None:
        iio.write(
            join(cfg.repout, "ndvi.png"),
            normaliser_image(np.copy(img_ndvi))
        )
        iio.write(
            join(cfg.repout, "ndwi.png"),
            normaliser_image(np.copy(img_ndwi))
        )

        # L'idée est de supprimer tous les changements qui sont du type
        # végétation--> végétation ou du type non-végétation--> végétation.
        # i.e. dès que im2 est végétation

        # Roughly the NDVI index caracterizes dense vegetation for values > 0.1
        img_veget = img_ndvi > cfg.ndvi_threshold
        img_veget = img_veget.squeeze()
#        iio.write(join(cfg.repout, f"veget.tif"), img_veget)
        himg = np.copy(h_uv)
        himg[img_veget] = 0
        img_veget = np.array(255 * img_veget, dtype=np.uint8)
        iio.write(join(cfg.repout, "ndvi_filtre.png"), img_veget)
        himg = np.array(255 * himg, dtype=np.uint8)        
        iio.write(join(cfg.repout, "huvl_par_ndvi_filtre.png"), himg)

        # Roughly the NDWI index caracterizes water for values >= 0.5
        # custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/
        img_water = img_ndwi >= cfg.ndwi_threshold
        img_water = img_water.squeeze()
        himg = np.copy(h_uv)
        himg[img_water] = 0

        img_water = np.array(255 * img_water, dtype=np.uint8)
        iio.write(join(cfg.repout, "ndwi_filtre.png"), img_water)
        himg = np.array(255 * himg, dtype=np.uint8)
        iio.write(join(cfg.repout, "huvl_par_ndwi_filtre.png"), himg)

        # Mix with the two masks
        img_mask = np.logical_or(
            img_ndvi > cfg.ndvi_threshold, img_ndwi >= cfg.ndwi_threshold
        )
        img_mask = img_mask.squeeze()
        himg = np.copy(h_uv)
        himg[img_mask] = 0

#        img_water = np.array(255 * img_mask, dtype=np.uint8)
#        iio.write(join(cfg.repout, "ndwi_filtre.png"), img_water)

        himg = np.array(255 * himg, dtype=np.uint8)
        iio.write(join(cfg.repout, "huvl_tout_filtre.png"), himg)
        
    return 0


def calculer_triplet_dates(cfg):
    """
    ...
    """

    # liste ordonnées des fichiers contenus dans le zip
    repzip = cfg.repout + "_zip"
    if not exists(repzip):
        os.mkdir(repzip)
    print("répertoire existe :", exists(repzip))

    with zipfile.ZipFile(cfg.zip, 'r') as monzip:
        fichiers = sorted([basename(f) for f in monzip.namelist()])
        pfxrep = [dirname(f) for f in monzip.namelist()][0]
        monzip.extractall(path=repzip)

    print("pfxrep:", pfxrep)
    print("contenu du répertoire:", sorted(os.listdir(repzip)))

#    exit()
    # récupération des dates concernés et conversion des dates au format
    # numériques
    triplets = []
    mesdates = [
        literal_eval(f.split('_')[0].replace('-', '')) for f in sorted(fichiers)
    ]
    if len(mesdates) == 6:
        triplets += [(join(repzip, pfxrep, fichiers[4]), join(repzip, pfxrep, fichiers[5]))]
        triplets += [(join(repzip, pfxrep, fichiers[2]), join(repzip, pfxrep, fichiers[3]))]
        triplets += [(join(repzip, pfxrep, fichiers[0]), join(repzip, pfxrep, fichiers[1]))]
        return triplets
#    06 04 18 51 35
    print(mesdates)
    date1 = mesdates[-2]
    date2 = mesdates[-1]
    print(date1, date2, fichiers[-2], fichiers[-1])

    triplets += [(join(repzip, pfxrep, fichiers[-2]), join(repzip, pfxrep, fichiers[-1]))]

    for annee in [10000, 20000]:
        # recherche des dates bornes : la date antérieure la plus proche du min
        # et la date ultérieure la plus proche du max
        ante1 = date1 - annee
        ante2 = date2 - annee

        # date antérieure la plus proche
        borne1i = None
        for i, madate in enumerate(mesdates):
            if madate < ante1:
                borne1i = i
            else:
                break
        # date ultérieure la plus proche
        borne2i = None
        for i, madate in enumerate(mesdates):
            if madate > ante2:
                borne2i = i
                break

        # petit algorithme
        optimal_dist, optimal_d1, optimal_d2 = float('inf'), 0, 0
        optimal_i1, optimal_i2 = 0, 0
        for i_1 in range(borne1i, borne2i):
            for i_2 in range(i_1+1, borne2i+1):
                # calcul de la distance
                dist = (
                    abs((ante2 - ante1) - (mesdates[i_2] - mesdates[i_1]))
                    + abs(ante2 - mesdates[i_2]) + abs(ante1 - mesdates[i_1])
                )
                if dist < optimal_dist:
                    optimal_i1 = i_1
                    optimal_i2 = i_2
                    optimal_d1 = mesdates[i_1]
                    optimal_d2 = mesdates[i_2]
                    optimal_dist = dist

        # récupération des résultats
#        print(ante1, ante2, (ante2 - ante1))
        print(
            optimal_d1, optimal_d2, optimal_dist,
            fichiers[optimal_i1], fichiers[optimal_i2]
        )
        triplets += [(
            join(repzip, pfxrep, fichiers[optimal_i1]),
            join(repzip, pfxrep, fichiers[optimal_i2])
        )]

# com    if cfg.debug:
# com        iio.write(
# com            join(cfg.repout, "im1.png"),
# com            normaliser_image(np.copy(triplets[0][1]), sat=0.001)
# com        )
# com        iio.write(
# com            join(cfg.repout, "im2.png"),
# com            normaliser_image(np.copy(triplets[0][2]), sat=0.001)
# com        )
# com        iio.write(
# com            join(cfg.repout, "antea1.png"),
# com            normaliser_image(np.copy(triplets[1][1]), sat=0.001)
# com        )
# com        iio.write(
# com            join(cfg.repout, "antea2.png"),
# com            normaliser_image(np.copy(triplets[1][2]), sat=0.001)
# com        )
# com        iio.write(
# com            join(cfg.repout, "anteb1.png"),
# com            normaliser_image(np.copy(triplets[2][1]), sat=0.001)
# com        )
# com        iio.write(
# com            join(cfg.repout, "anteb2.png"),
# com            normaliser_image(np.copy(triplets[2][2]), sat=0.001)
# com        )
# com
    print(triplets)
    return triplets


def comparer_par_triplet(cfg):
    """
    ...
    """


    (im1, im2), (antea1, antea2), (anteb1, anteb2) = calculer_triplet_dates(cfg)
    # im1, im2, antea1, antea2, anteb1, anteb2
    # im1, im2, im1, im2
    # im1, im2, ante11, ante12

    with multiprocessing.Pool(processes=3) as pool:
        results = Parallel(n_jobs=3)(
            delayed(calculer_mappe_binaire)(
                cfg, ima, imb, refa, refb, ref_ident
            )
            for ima, imb, refa, refb, ref_ident in zip(
                [im1, im1, im1], [im2, im2, im2], [im1, antea1, anteb1],
                [im1, antea2, anteb2], [True, False, False]
            )
        )
        [res1, res2, res3] = results
        print(results)
        print(type(res1), type(res2), type(res3))
    [h_uv, img_ndvi, img_ndwi] = res1
    
#        all_blocks = results[0][0] + results[1][0] + results[2][0]
#        red_blocks = results[0][1] + results[1][1] + results[2][1]
        
    mappes_binaires = []
#com    print("paire 1...")
#com    cfg.identiques = True
#com    [h_uv, img_ndvi, img_ndwi] = calculer_mappe_binaire(
#com        cfg, im1, im2, im1, im1, ref_identiques
#com    )
    mappes_binaires += [h_uv]
    huvl_original = np.array(255 * h_uv, dtype=np.uint8)
    iio.write(join(cfg.repout, "huvl_original.png"), huvl_original)

    print("paire 2...")
    cfg.identiques = False
    h_uv = res2
#com    h_uv = calculer_mappe_binaire(
#com        cfg, im1, im2, antea1, antea2, ref_identiques
#com    )
    mappes_binaires += [h_uv]
    print("paire 3...")
    cfg.identiques = False
#com    h_uv = calculer_mappe_binaire(
#com        cfg, im1, im2, anteb1, anteb2, ref_identiques
#com    )
    h_uv = res3
    mappes_binaires += [h_uv]
    mappes = np.array(mappes_binaires)

    # vote majoritaire
    mappe = 1 * np.sum(mappes, axis=0) >= 2

    huvl_triplet = np.array(255 * mappe, dtype=np.uint8)
    iio.write(join(cfg.repout, "huvl_triplet.png"), huvl_triplet)

    ecrire_mappes(cfg, img_ndvi, img_ndwi, mappe)
#    iio.write(join(cfg.repout, "mappe.png"), mappe)
    ficimgs = [im1, im2, antea1, antea2, anteb1, anteb2]
    noms = [
        "im1.png", "im2.png", "antea1.png",
        "antea2.png", "anteb1.png", "anteb2.png"
    ]

    for ficimg, nom in zip(ficimgs, noms):
        print("ficimgs:", ficimgs)
        print("ficimg:",ficimg, nom)
        img = iio.read(ficimg)[:,:,0:3]
        iio.write(join(cfg.repout, nom), normaliser_image(img, sat=0.001))
    return 0


def calculer_mappe_binaire(cfg, image1, image2, ante1, ante2, ref_ident):
    """
    ...
    """
    print("quadruplet:", image1, image2, ante1, ante2)
    print("exists(image1)", exists(image1))
    im1 = iio.read(image1)
    im2 = iio.read(image2)
    ante1 = iio.read(ante1)
    ante2 = iio.read(ante2)
    if not exists(cfg.repout):
        os.mkdir(cfg.repout)

    # suppression du canal B08 si besoin
    im1, _, _ = compute_index_maps(im1)
    im2, img_ndvi2, img_ndwi2 = compute_index_maps(im2)
    ante1, _, _ = compute_index_maps(ante1)
    ante2, _, _ = compute_index_maps(ante2)

    im1 = np.mean(im1, axis=2)
    im2 = np.mean(im2, axis=2)
    ante1 = np.mean(ante1, axis=2)
    ante2 = np.mean(ante2, axis=2)
    assert im1.shape == im2.shape and im1.shape == ante1.shape

    h_uv, pfal = algorithme(cfg, im1, im2, ante1, ante2, 0, ref_ident)

    if cfg.debug:
        g_uv = normaliser_image(h_uv)
        iio.write(join(cfg.repout, "huvl.png"), g_uv)
        g_pfal = calorifier_image(pfal)
        iio.write(join(cfg.repout, "pfal.png"), g_pfal)

    if ref_ident:
        return [h_uv, img_ndvi2, img_ndwi2]

    return h_uv


def main():
    """
    ...
    """

    cfg = lit_parametres()

    if cfg.monaction == "apparier":
        comparer_une_paire(cfg)
    elif cfg.monaction == "tripler":
        comparer_par_triplet(cfg)

    return 0


if __name__ == "__main__":
    execution_time = timeit.timeit(main, number=1)
    print(f"Execution time: {execution_time:.6f} seconds")
    #main()

    #Lignes de commandes
    # python3 kervrann.py --image1 img1.png --image2 img2.png \
    # --scale 2 --epsilon 1 --sigma 0.8 --b 3 --metrique correlation \
    # --repout mcor_s2_b3_eps1_sig0.8
    # python3 kervrann.py --image1 img1.png --image2 img2.png \
    # --scale 2 --epsilon 1 --sigma 0.8 --b 3 --metrique ratio
    # --repout mrat_s2_b3_eps1_sig0.8
# python kervrann.py \
#     --image1 2018-07-12_S2B_orbit_008_tile_31TDF_L1C_band_RGBI.tif \
#     --image2 2019-01-03_S2A_orbit_008_tile_31TDF_L1C_band_RGBI.tif \
#     --ante1 2017-07-12_S2A_orbit_008_tile_31TDF_L1C_band_RGBI.tif \
#     --ante2 2018-01-18_S2A_orbit_008_tile_31TDF_L1C_band_RGBI.tif \
#     --metrique lin --scale 6 --b 2 --repout multiscale;
#
