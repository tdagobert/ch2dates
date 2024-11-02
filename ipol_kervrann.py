#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# BSD 3-Clause License
#
# Copyright (c) 2024, Tristan Dagobert  tristan.dagobert@ens-paris-saclay.fr
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""
This program computes the changes bewteen two RBG images according to the
approach by Kervrann et al. described in the paper "Multiscale neighborhood-wise
decision fusion for redundancy detection in image pairs".
"""

import os
from os.path import exists, join
import argparse
import timeit
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import factorial
from matplotlib import cm

from numba import njit
import imageio as iio

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
def calculer_phi(u, v, u_rho, v_rho, l, b, sigma, metrique, est_uu=False):
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
    demi_b = b // 2

    # résultat
    phi_uvl = np.nan * np.ones((nlig, ncol, b**2))

    # images filtrées
#    u_rho = gaussian_filter(u, sigma)
#    v_rho = gaussian_filter(v, sigma)

    # calcul pixellien
    for xi in np.arange(nlig):
#        print(xi)
        for xj in np.arange(ncol):

            # test aux limites
            if xi-l < 0 or nlig <= xi+l or xj-l < 0 or ncol <= xj+l:
                continue

            # voisinage de x
            if metrique == "l2":
                uu = u[xi-l:xi+l+1, xj-l:xj+l+1] - u_rho[xi, xj]
            elif metrique == "ratio":
                uu = u[xi-l:xi+l+1, xj-l:xj+l+1]
            elif metrique == "correlation":
                uu = u[xi-l:xi+l+1, xj-l:xj+l+1]
            elif metrique == "lin":
                uu = u[xi-l:xi+l+1, xj-l:xj+l+1]
            elif metrique == "zncc":
                uu = u[xi-l:xi+l+1, xj-l:xj+l+1]
                muu = np.mean(uu)
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
                        if metrique == "l2":
                            vv = vv - v_rho[yi, yj]
                            phi_uvl[xi, xj, k] = np.sum((uu - vv)**2)
                        elif metrique == "ratio":
                            vv = vv * (u_rho[xi, xj] / v_rho[yi, yj])
                            phi_uvl[xi, xj, k] = np.sum((uu - vv)**2)
                        elif metrique == "lin":
                            suu = np.sum(uu*uu)
                            svv = np.sum(vv*vv)
                            phi_uvl[xi, xj, k] = (
                                max(suu, svv)
                                * (1 - np.sum(uu * vv)**2 / (suu * svv))
                            )
                        elif metrique == "correlation":
                            phi_uvl[xi, xj, k] = (
                                1
                                - np.sum(uu * vv) /
                                (np.sqrt(np.sum(uu*uu)) * np.sqrt(np.sum(vv*vv))
                                 )
                            )
                        elif metrique == "zncc":
                            mvv = np.mean(vv)
                            phi_uvl[xi, xj, k] = (
                                1
                                - np.sum((uu - muu) * (vv - mvv))
                                /(vv.size * np.std(uu) * np.std(vv))
                            )

                    k += 1

    phi_uvl = gerer_bords(phi_uvl)
    return phi_uvl


# alt@njit
# altdef compute_tau_l_mean(phi_uul):
# alt    nlig, ncol, _ = phi_uul.shape
# alt    tau_l_mean = nb.typed.List.empty_list(nb.f8)
# alt    for i in np.arange(nlig):
# alt        for j in np.arange(ncol):
# alt            # recherche du minimum sur le voisinage b(x)
# alt            tau_l_mean.append(np.nanmin(phi_uul[i, j, :]))
# alt
# alt    tau_l_mean = np.nanmean(np.array(tau_l_mean))
# alt
# alt    return tau_l_mean


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
#    for l in [cfg.scale]:
        print(f"Échelle {l}")
        # calcul de φ(u, u, l)
        im1_rho = gaussian_filter(im1, cfg.sigma)
        phi_uul = calculer_phi(
            im1, im1, im1_rho, im1_rho, l,
            cfg.b, cfg.sigma, cfg.metric, est_uu=True
        )
        print(phi_uul.shape)
        nlig, ncol, ncan = phi_uul.shape
        for n in np.arange(ncan):
            iio.imwrite(f"phi_uul_{n:03}.tif", phi_uul[:, :, n])

        # calcul de φ(u, v, l)
        im2_rho = gaussian_filter(im2, cfg.sigma)
        phi_uvl = calculer_phi(
            im1, im2, im1_rho, im2_rho, l, cfg.b, cfg.sigma, cfg.metric
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
#        iio.imwrite(join(cfg.dirout, f"tau_ul_s{l}_c{ican}.tif"), tau_ul)
        print("# calcul de τ(u, l) d'après (5.1)")
        # calcul de S_Nl
        S_Nl = np.zeros((nlig, ncol))
        for i in np.arange(nlig):
            for j in np.arange(ncol):
                try:
                    S_Nl[i, j] = np.sum(phi_uvl[i, j, :] >= tau_ul[i, j])
                except ValueError:
                    pass
#        iio.imwrite(join(cfg.dirout, f"snl{l}_c{ican}.tif"), S_Nl)
        # calcul de decision_l d'après (4.1)
        decision_l = np.uint8(S_Nl == (cfg.b * cfg.b))
        decisions += [decision_l]

        # calcul de pfa_l
        pfa_l =  np.nanmean(np.exp(S_Nl - (cfg.b * cfg.b)))
        pfas += [pfa_l]

    decisions = np.array(decisions)
    pfas = np.array(pfas)
    return pfas, decisions


def calculer_pfal(kd, lambdaa, nlig, ncol):
    """
    Computation of the probability of false alarms.
    kd: np.array ndim=(nlig, ncol)
    lambdaa : float
    nlig : int
    ncol : int
    """
    pfal = np.zeros((nlig, ncol))

    for i in np.arange(nlig):
        for j in np.arange(ncol):
            for k in np.arange(kd[i, j] + 1):
#                print(k, lambdaa)
                pfal[i, j] += (
                    (lambdaa)**k / factorial(k) * np.exp(-lambdaa)
                )
            pfal[i, j] = 1 - pfal[i, j]
    return pfal


def calculer_alpha(epsilon, nlig, ncol, pfal):
    """
    Compute the alpha threshold.
    """
    alpha = np.max((epsilon/(nlig*ncol), np.min(pfal)))
    return alpha


def algorithm(cfg, im1, im2, ican):
    """
    cfg: Namespace
    im1: np.array ndim=(nlig, ncol)
    im2: np.array ndim=(nlig, ncol)
    ican : int
        Channel index.
    """
    nlig, ncol = im1.shape
    pfas, decisions = calculer_pfas(cfg, im1, im2, ican)
    lambda_n = np.sum(np.array(pfas))
    print(f"lambda_n {lambda_n}")
    # compute the positive decisions kd
    kd = np.sum(decisions, axis=0)

    # compute P_FA(x, L) for all x
    pfal = calculer_pfal(kd, lambda_n, nlig, ncol)

    # Computation of the uniform threshold α to detect meaningful changes
    alpha = calculer_alpha(cfg.epsilon, nlig, ncol, pfal)

    # Computation of the change detection map
    h_uv = np.uint8(pfal <= alpha)
    return h_uv, pfal


def load_parameters():
    """
    …
    """

    desc = "Compute the changes between two images."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--image1", type=str, required=True, help="First image."
    )
    parser.add_argument(
        "--image2", type=str, required=True, help="Second image."
    )
    parser.add_argument(
        "--scale", type=int, required=False, help="Number of scales.", default=2
    )
    parser.add_argument(
        "--b", type=int, required=False, default=3,
        help="Side of the square neighborhood of x."
    )
    parser.add_argument(
        "--B", type=int, required=False, default=3,
        help="Side of the square search window related to x."
    )
    parser.add_argument(
        "--metric", type=str, required=False, help="Dissimilarity measure.",
        choices=["correlation", "l2", "ratio", "zncc", "lin"], default="l2"
    )
    parser.add_argument(
        "--epsilon", type=float, required=False, default=1.0,
        help="Number of false alarms threshold."
    )
    parser.add_argument(
        "--sigma", type=float, required=False, default=0.8,
        help="Standard deviation of the blur kernel."
    )
    parser.add_argument(
        "--dirout", type=str, required=False, default="./",
        help="Output directory."
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
    print("shape",img.shape)

    img = np.array(img, dtype=np.uint8)
    return img


def calorifier_image(img, apply_log=True):
    """
    Make a jetcolor image map.
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

def convert_to_gray_image(img):
    """
    Convert an RGB image into a gray level one. If the image contains 4
    channels, we assume it is a Sentinel-2 image with the B04, B03, B02, B08
    channels storage in this order.
    """
    img = img[:, :, 0:3]
    img = np.mean(img, axis=1)
    return img
#com
#com
#comdef compute_index_maps(cfg, img):
#com    """
#com    If the image contains 4 channels, we assume it is a Sentinel-1 image with
#com    the B04, B03, B02, B08 channels storage in this order. We retrieve the
#com    B08 to compute the NDVI index…
#com    """
#com    nlig, ncol, ncan = img.shape
#com
#com    if ncan == 4:
#com        # we compute the NDVI index, where values stand in [-1, +1]
#com        ndvi = (img[:, :, 3] - img[:, :, 0]) / (img[:, :, 3] + img[:, :, 0])
#com        ndvi = np.expand_dims(ndvi, axis=-1)
#com        # we normalize
#com        img_ndvi = normaliser_image(ndvi)
#com#        g_can = 255 * np.ones((nlig, ncol, 1))
#com#        can = 255 * (1 - (ndvi + 1) / 2)
#com#        img_ndvi = np.concatenate((can, g_can, can), axis=2)
#com
#com        # we compute the NDWI index, where values stand in [-1, +1]
#com        ndwi = (img[:, :, 1] - img[:, :, 3]) / (img[:, :, 1] + img[:, :, 3])
#com        ndwi = np.expand_dims(ndwi, axis=-1)
#com        # we normalize
#com        img_ndwi = normaliser_image(ndwi)
#com#        b_can = 255 * np.ones((nlig, ncol, 1))
#com#        can = 255 * (1 - (ndwi + 1) / 2)
#com#        img_ndwi = np.concatenate((can, can, b_can), axis=2)
#com
#com        img = img[:, :, 0:3]
#com
#com        return img, img_ndvi, ndvi, img_ndwi, ndwi
#com    else:
#com        return img, None, None, None, None
#com

def main():
    """
    ...
    """

    cfg = load_parameters()

    im1 = iio.imread(cfg.image1)
    im2 = iio.imread(cfg.image2)
    im1 = convert_to_gray_image(im1)
    im2 = convert_to_gray_image(im2)
#    im1 = im1.reshape(nlig, ncol, 1)
#    im2 = im2.reshape(nlig, ncol, 1)
    if not exists(cfg.dirout):
        os.mkdir(cfg.dirout)

    iio.imwrite(
        join(cfg.dirout, "im1.png"), normaliser_image(np.copy(im1), sat=0.001)
    )
    iio.imwrite(
        join(cfg.dirout, "im2.png"), normaliser_image(np.copy(im2), sat=0.001)
    )

    h_uv, pfal = algorithm(cfg, im1, im2, 0)
    h_uv = normaliser_image(h_uv)
    iio.imwrite(join(cfg.dirout, "huvl.png"), h_uv)
    pfal = calorifier_image(pfal)
    iio.imwrite(join(cfg.dirout, "pfal.png"), pfal)
    return 0

if __name__ == "__main__":
    execution_time = timeit.timeit(main, number=1)
    print(f"Execution time: {execution_time:.6f} seconds")
    #main()

    #Lignes de commandes
    # python3 kervrann.py --image1 img1.png --image2 img2.png --scale 2 --epsilon 1 --sigma 0.8 --b 3 --metrique correlation --dirout mcor_s2_b3_eps1_sig0.8
    # python3 kervrann.py --image1 img1.png --image2 img2.png --scale 2 --epsilon 1 --sigma 0.8 --b 3 --metrique ratio --dirout mrat_s2_b3_eps1_sig0.8
