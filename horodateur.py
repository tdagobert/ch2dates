#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""
…
"""

import os
from os.path import exists, join
import argparse
import zipfile


def lit_parametres():
    """
    …
    """

    d = "Compute the changes between two images."
    parser = argparse.ArgumentParser(description=d)
    parser.add_argument(
        "--zip", type=str, required=True, help="Zip contenant les images."
    )
    parser.add_argument(
        "--d1", type=str, required=False, help="Date au format AAAA-MM-JJ."
    )
    parser.add_argument(
        "--d2", type=str, required=False, help="Date au format AAAA-MM-JJ."
    )

    cfg = parser.parse_args()

    return cfg


def main():
    """
    ...
    """

    cfg = lit_parametres()

    # liste ordonnées des fichiers contenus dans le zip
    zip = zipfile.ZipFile(cfg.zip)
    fichiers = sorted(zip.namelist())
#    print("fichiers:", fichiers)
#    exit()
    # récupération des dates concernés et conversion des dates au format
    # numériques
    mesdates = [
        eval(f.split('_')[0].replace('-', ''))
        for f in fichiers
    ]
    
#    print(mesdates)
    date1 = mesdates[-2] if cfg.d1 is None else eval(cfg.d1.replace('-', ''))
    date2 = mesdates[-1] if cfg.d2 is None else eval(cfg.d2.replace('-', ''))    
    #print(date1, date2, fichiers[-2], fichiers[-1])
    print(join(cfg.zip.replace(".zip", ""), fichiers[-2]))
    print(join(cfg.zip.replace(".zip", ""), fichiers[-1]))
    
    for an in [10000, 20000]:
        # recherche des dates bornes : la date antérieure la plus proche du min
        # et la date ultérieure la plus proche du max
        ante1 = date1 - an
        ante2 = date2 - an

        # date antérieure la plus proche
        borne1 = None
        borne1i = None
        for i, madate in enumerate(mesdates):
            if madate < ante1:
                borne1 = madate
                borne1i = i
            else:
                break
        # date ultérieure la plus proche
        borne2 = None
        borne2i = None
        for i, madate in enumerate(mesdates):
            if madate > ante2:
                borne2 = madate
                borne2i = i
                break
        
        # petit algorithme
        optimal_dist, optimal_d1, optimal_d2 = float('inf'), 0, 0
        optimal_i1, optimal_i2 = 0, 0
        for i1 in range(borne1i, borne2i):
            for i2 in range(i1+1, borne2i+1):
                # calcul de la distance
                dist = (
                    abs((ante2 - ante1) - (mesdates[i2] - mesdates[i1]))
                    + abs(ante2 - mesdates[i2]) + abs(ante1 - mesdates[i1])
                )
                if dist < optimal_dist:
                    optimal_i1 = i1
                    optimal_i2 = i2
                    optimal_d1 = mesdates[i1]
                    optimal_d2 = mesdates[i2]
                    optimal_dist = dist

        # récupération des résultats
#        print(ante1, ante2, (ante2 - ante1))
#        print(
#            optimal_d1, optimal_d2, optimal_dist,
#            fichiers[optimal_i1], fichiers[optimal_i2]
#        )
        print(join(cfg.zip.replace(".zip", ""), fichiers[optimal_i1]))
        print(join(cfg.zip.replace(".zip", ""), fichiers[optimal_i2]))
        

    return


if __name__ == "__main__":
    main()
