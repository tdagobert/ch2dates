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
def handle_boundaries(img):
    """
    Replacement of NaN values located on the edges,
    by the values located on the boundaries.
    Parameters
    ----------
    img : np.array ndim=(nrow, ncol, ncan)
    """
    nrow, ncol, ncan = img.shape
    for k in np.arange(ncan):
        # replacement of columns
        for i in np.arange(nrow):
            j = 0
            while j < ncol and np.isnan(img[i, j, k]):
                j += 1
            # entire line is NaN
            if j == ncol:
                continue
            # replacement of left columns
            img[i, 0:j, k] = img[i, j, k]

            while not np.isnan(img[i, j, k]):
                j += 1
            # replacement of right columns
            img[i, j:ncol, k] = img[i, j-1, k]

        # replacement of lines
        for j in np.arange(ncol):
            i = 0
            while i < nrow and np.isnan(img[i, j, k]):
                i += 1
            # entire column is NaN
            if i == nrow:
                continue
            # replacement of top lines
            img[0:i, j, k] = img[i, j, k]

            while not np.isnan(img[i, j, k]):
                i += 1
            # replacement of right colums
            img[i:nrow, j, k] = img[i-1, j, k]

    return img


@njit
def compute_phi(imu, imv, u_rho, v_rho, l, b, metric, is_uu=False):
    """
    Parameters
    ----------
    imu : np.array ndim=(nrow, ncol)
        Reference image.
    imv : np.array ndim=(nrow, ncol)
        Compared image.
    l : int
        Half side of the square patch.
    b : int
        Side of the square search window.
    metric : str
        Name of metric used.
    is_uu : bool
        Indicate if pair of parameters (u, v) is (u, u) or not.
    """

    nrow, ncol = imu.shape
    half_b = b // 2

    # initialization
    phi_uvl = np.nan * np.ones((nrow, ncol, b**2))

    # computation per pixel
    for x_i in np.arange(nrow):
        for x_j in np.arange(ncol):
            # limits tests
            if x_i-l < 0 or nrow <= x_i+l or x_j-l < 0 or ncol <= x_j+l:
                continue
            # neighborhood of x
            if metric == "l2":
                tilu = imu[x_i-l:x_i+l+1, x_j-l:x_j+l+1] - u_rho[x_i, x_j]
            elif metric == "ratio":
                tilu = imu[x_i-l:x_i+l+1, x_j-l:x_j+l+1]
            elif metric == "correlation":
                tilu = imu[x_i-l:x_i+l+1, x_j-l:x_j+l+1]
            elif metric == "lin":
                tilu = imu[x_i-l:x_i+l+1, x_j-l:x_j+l+1]
            elif metric == "zncc":
                tilu = imu[x_i-l:x_i+l+1, x_j-l:x_j+l+1]
                muu = np.mean(tilu)
            k = 0
            for m in np.arange(-half_b, half_b + 1):
                for n in np.arange(-half_b, half_b + 1):
                    y_i = x_i + m
                    y_j = x_j + n

                    # limits tests
                    if y_i-l < 0 or nrow <= y_i+l or y_j-l < 0 or ncol <= y_j+l:
                        k += 1
                        continue

                    # neighborhood of y
                    tilv = imv[y_i-l:y_i+l+1, y_j-l:y_j+l+1]

                    # calcul de la distance
                    if not is_uu or (is_uu and not (y_i == x_i and y_j == x_j)):
                        if metric == "l2":
                            tilv = tilv - v_rho[y_i, y_j]
                            phi_uvl[x_i, x_j, k] = np.sum((tilu - tilv)**2)
                        elif metric == "ratio":
                            tilv = tilv * (u_rho[x_i, x_j] / v_rho[y_i, y_j])
                            phi_uvl[x_i, x_j, k] = np.sum((tilu - tilv)**2)
                        elif metric == "lin":
                            suu = np.sum(tilu*tilu)
                            svv = np.sum(tilv*tilv)
                            phi_uvl[x_i, x_j, k] = (
                                max(suu, svv)
                                * (1 - np.sum(tilu * tilv)**2 / (suu * svv))
                            )
                        elif metric == "correlation":
                            phi_uvl[x_i, x_j, k] = (
                                1
                                - np.sum(tilu * tilv) /
                                (np.sqrt(np.sum(tilu*tilu)) * np.sqrt(np.sum(tilv*tilv))
                                 )
                            )
                        elif metric == "zncc":
                            mvv = np.mean(tilv)
                            phi_uvl[x_i, x_j, k] = (
                                1
                                - np.sum((tilu - muu) * (tilv - mvv))
                                /(tilv.size * np.std(tilu) * np.std(tilv))
                            )
                    k += 1

    phi_uvl = handle_boundaries(phi_uvl)
    return phi_uvl


# alt@njit
# altdef compute_tau_l_mean(phi_uul):
# alt    nrow, ncol, _ = phi_uul.shape
# alt    tau_l_mean = nb.typed.List.empty_list(nb.f8)
# alt    for i in np.arange(nrow):
# alt        for j in np.arange(ncol):
# alt            # recherche du minimum sur le voisinage b(x)
# alt            tau_l_mean.append(np.nanmin(phi_uul[i, j, :]))
# alt
# alt    tau_l_mean = np.nanmean(np.array(tau_l_mean))
# alt
# alt    return tau_l_mean

@njit
def compute_measures_phi(cfg, im1, im2, scale):
    """
    ...
    """
    # computation of φ(u, u, s)
    im1_rho = gaussian_filter(im1, cfg.sigma)
    phi_uus = compute_phi(
        im1, im1, im1_rho, im1_rho, scale, cfg.b, cfg.metric, is_uu=True
    )
    print(phi_uus.shape)
    _, _, ncan = phi_uus.shape
    for n in np.arange(ncan):
        iio.imwrite(f"phi_uus_{n:03}.tif", phi_uus[:, :, n])

    # computation of φ(u, v, s)
    im2_rho = gaussian_filter(im2, cfg.sigma)
    phi_uvs = compute_phi(im1, im2, im1_rho, im2_rho, scale, cfg.b, cfg.metric)
    return phi_uus, phi_uvs


@njit
def compute_theta_us(phi_uus):
    """
    ...
    """
    nrow, ncol, _ = phi_uus.shape
    # computation of θ_us
    theta_us = []
    for i in np.arange(nrow):
        for j in np.arange(ncol):
            try:
                # search of the minimum in b(x)
                theta_us += [np.nanmin(phi_uus[i, j, :])]
            except ValueError:
                pass
    theta_us = np.nanmean(np.array(theta_us))
    print(f"# θ_us {theta_us:3.5e}")
    return theta_us


@njit
def compute_tau_us(phi_uus, theta_us):
    """
    ...
    """
    nrow, ncol, _ = theta_us.shape
    tau_us = np.zeros((nrow, ncol))
    for i in np.arange(nrow):
        for j in np.arange(ncol):
            try:
                tau_us[i, j] = np.nanmax(
                    (np.nanmax(phi_uus[i, j, :]), theta_us)
                )
            except ValueError:
                pass
    return tau_us


@njit
def compute_number_of_decisions(phi_uvs, phi_vus, tau_s, side_b):
    """
    ...
    """
    varphi = np.minimum(phi_uvs, phi_vus)
    nrow, ncol, _ = varphi.shape
    f_s = np.zeros((nrow, ncol))
    for i in np.arange(nrow):
        for j in np.arange(ncol):
            try:
                f_s[i, j] = np.sum(varphi[i, j, :] >= tau_s[i, j])
            except ValueError:
                pass
    #        iio.imwrite(join(cfg.dirout, f"snl{scale}.tif"), f_s)
    # computation of the positive decisions
    decision_s = np.uint8(f_s == (side_b * side_b))
    pfa_s =  np.nanmean(np.exp(f_s - (side_b * side_b)))
    return decision_s, pfa_s


def compute_pfas(cfg, im1, im2):
    """
    Compute the probability of positive detections under H_0 at each scale.

    Parameters
    ----------
    cfg: Namespace
    im1: np.array(nrow, ncol)
    im2: np.array(nrow, ncol)
    ican: int
    Return
    ------
    decisions: np.array(L, nrow, ncol)
    pfas: np.array(L, nrow, ncol)
    """

#com    nrow, ncol = im1.shape
    pfas = []
    decisions = []

    for scale in np.arange(1, cfg.scale+1):
        print(f"Scale {scale}")
        phi_uus, phi_uvs = compute_measures_phi(cfg, im1, im2, scale)
        phi_vvs, phi_vus = compute_measures_phi(cfg, im2, im1, scale)
#com         # computation of φ(u, u, s)
#com         im1_rho = gaussian_filter(im1, cfg.sigma)
#com         phi_uus = compute_phi(
#com             im1, im1, im1_rho, im1_rho, scale,
#com             cfg.b, cfg.metric, is_uu=True
#com         )
#com         print(phi_uus.shape)
#com         _, _, ncan = phi_uus.shape
#com         for n in np.arange(ncan):
#com             iio.imwrite(f"phi_uus_{n:03}.tif", phi_uus[:, :, n])
#com
#com         # computation of φ(u, v, s)
#com         im2_rho = gaussian_filter(im2, cfg.sigma)
#com         phi_uvs = compute_phi(
#com             im1, im2, im1_rho, im2_rho, scale, cfg.b, cfg.metric
#com         )
#com
        theta_us = compute_theta_us(phi_uus)
        theta_vs = compute_theta_us(phi_vvs)
#com         # computation of θ_us
#com         theta_us = []
#com         for i in np.arange(nrow):
#com             for j in np.arange(ncol):
#com                 try:
#com                     # search of the minimum in b(x)
#com                     theta_us += [np.nanmin(phi_uus[i, j, :])]
#com                 except ValueError:
#com                     pass
#com         theta_us = np.nanmean(np.array(theta_us))
#com
#com         print(f"# θ_us {theta_us:3.5e}")

        # computation of τ(s)
        tau_us = compute_tau_us(phi_uus, theta_us)
        tau_vs = compute_tau_us(phi_vvs, theta_vs)
        tau_s = np.minimum(tau_us, tau_vs)

#        iio.imwrite(join(cfg.dirout, f"tau_ul_s{scale}.tif"), tau_ul)
        print("# calcul de τ(u, s) d'après (5.1)")

#com        # computation of F_s
#com        f_s = np.zeros((nrow, ncol))
#com        for i in np.arange(nrow):
#com            for j in np.arange(ncol):
#com                try:
#com                    f_s[i, j] = np.sum(phi_uvs[i, j, :] >= tau_us[i, j])
#com                except ValueError:
#com                    pass
#com#        iio.imwrite(join(cfg.dirout, f"snl{scale}.tif"), f_s)
#com        # computation of the positive decisions
#com        decision_s = np.uint8(f_s == (cfg.b * cfg.b))
        decision_s, pfa_s = compute_number_of_decisions(
            phi_uvs, phi_vus, tau_s, cfg.b
        )
        decisions += [decision_s]

        # computation of pfa_l
#com        pfa_s =  np.nanmean(np.exp(f_s - (cfg.b * cfg.b)))
        pfas += [pfa_s]

    decisions = np.array(decisions)
    pfas = np.array(pfas)
    return pfas, decisions


def calculer_pfal(k_d, lambdaa, nrow, ncol):
    """
    Computation of the probability of false alarms.
    kd: np.array ndim=(nrow, ncol)
    lambdaa : float
    nrow : int
    ncol : int
    """
    pfal = np.zeros((nrow, ncol))

    for i in np.arange(nrow):
        for j in np.arange(ncol):
            for k in np.arange(k_d[i, j] + 1):
#                print(k, lambdaa)
                pfal[i, j] += (
                    (lambdaa)**k / factorial(k) * np.exp(-lambdaa)
                )
            pfal[i, j] = 1 - pfal[i, j]
    return pfal


def calculer_alpha(epsilon, nrow, ncol, pfal):
    """
    Compute the alpha threshold.
    """
    alpha = np.max((epsilon/(nrow*ncol), np.min(pfal)))
    return alpha


def algorithm(cfg, im1, im2):
    """
    cfg: Namespace
    im1: np.array ndim=(nrow, ncol)
    im2: np.array ndim=(nrow, ncol)
    """
    nrow, ncol = im1.shape
    # compute the probability of positive detections under H_0 at each scale
    pfas, decisions = compute_pfas(cfg, im1, im2)

    lambda_n = np.sum(np.array(pfas))
    print(f"lambda_n {lambda_n}")
    # compute the positive decisions kd
    k_d = np.sum(decisions, axis=0)

    # compute P_FA(x, L) for all x
    pfal = calculer_pfal(k_d, lambda_n, nrow, ncol)

    # Computation of the uniform threshold α to detect meaningful changes
    alpha = calculer_alpha(cfg.epsilon, nrow, ncol, pfal)

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


def normalize_image(img, sat=None):
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


def convert_to_jetcolor_image(img, apply_log=True):
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
#com    nrow, ncol, ncan = img.shape
#com
#com    if ncan == 4:
#com        # we compute the NDVI index, where values stand in [-1, +1]
#com        ndvi = (img[:, :, 3] - img[:, :, 0]) / (img[:, :, 3] + img[:, :, 0])
#com        ndvi = np.expand_dims(ndvi, axis=-1)
#com        # we normalize
#com        img_ndvi = normalize_image(ndvi)
#com#        g_can = 255 * np.ones((nrow, ncol, 1))
#com#        can = 255 * (1 - (ndvi + 1) / 2)
#com#        img_ndvi = np.concatenate((can, g_can, can), axis=2)
#com
#com        # we compute the NDWI index, where values stand in [-1, +1]
#com        ndwi = (img[:, :, 1] - img[:, :, 3]) / (img[:, :, 1] + img[:, :, 3])
#com        ndwi = np.expand_dims(ndwi, axis=-1)
#com        # we normalize
#com        img_ndwi = normalize_image(ndwi)
#com#        b_can = 255 * np.ones((nrow, ncol, 1))
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
#    im1 = im1.reshape(nrow, ncol, 1)
#    im2 = im2.reshape(nrow, ncol, 1)
    if not exists(cfg.dirout):
        os.mkdir(cfg.dirout)

    iio.imwrite(
        join(cfg.dirout, "im1.png"), normalize_image(np.copy(im1), sat=0.001)
    )
    iio.imwrite(
        join(cfg.dirout, "im2.png"), normalize_image(np.copy(im2), sat=0.001)
    )

    h_uv, pfal = algorithm(cfg, im1, im2)
    h_uv = normalize_image(h_uv)
    iio.imwrite(join(cfg.dirout, "huvl.png"), h_uv)
    pfal = convert_to_jetcolor_image(pfal)
    iio.imwrite(join(cfg.dirout, "pfal.png"), pfal)
    return 0

if __name__ == "__main__":
    execution_time = timeit.timeit(main, number=1)
    print(f"Execution time: {execution_time:.6f} seconds")
    #main()

    #Lignes de commandes
    # python3 kervrann.py --image1 img1.png --image2 img2.png --scale 2 --epsilon 1 --sigma 0.8 --b 3 --metrique correlation --dirout mcor_s2_b3_eps1_sig0.8
    # python3 kervrann.py --image1 img1.png --image2 img2.png --scale 2 --epsilon 1 --sigma 0.8 --b 3 --metrique ratio --dirout mrat_s2_b3_eps1_sig0.8
