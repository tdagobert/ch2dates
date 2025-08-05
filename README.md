# A Brief Analysis of the Change Detector by Kervrann et al.

## Abstract
This work describes the symmetric method by Kervrann et al. for change detection. The
algorithm processes a pair of images using a hypothesis testing technique with an a contrario
approach. We perform a brief analysis of the results produced by the method and evaluate its
quality and limitations on the Sentinel-2 OSCD dataset.

## Installation
Install a virtual environment with the required packages to execute the program :
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Usage
```
ipol_kervrann.py [-h] [--zip ZIP] [--image1 IMAGE1] [--image2 IMAGE2] [--scale SCALE] [--b B] [--B B] [--metric {corr,rho,mult,zncc,lin}] [--epsilon EPSILON] [--sigma SIGMA]
                        [--dirout DIROUT]

Compute the changes between two images.

options:
  -h, --help            show this help message and exit
  --zip ZIP             Contains the image pair.
  --image1 IMAGE1       First image.
  --image2 IMAGE2       Second image.
  --scale SCALE         Number of scales.
  --b B                 Side of the square neighborhood of x.
  --B B                 Side of the square search window related to x.
  --metric {corr,rho,mult,zncc,lin}
                        Dissimilarity measure.
  --epsilon EPSILON     Number of false alarms threshold.
  --sigma SIGMA         Standard deviation of the blur kernel.
  --dirout DIROUT       Output directory.
```
## Example
```
(venv) python ipol_kervrann.py --image1 pair/2016-02-01_S2A_orbit_050_tile_34SGH_L1C_band_RGBI.tif --image2 pair/2016-03-02_S2A_orbit_050_tile_34SGH_L1C_band_RGBI.tif --scale 3 --b 5 --B 5 --metric rho --sigma 0.1 --epsilon 1 --dirout result/
(venv) python ipol_kervrann.py --zip pair.zip --scale 3 --b 5 --B 5 --metric rho --sigma 0.1 --epsilon 1 --dirout result/
```
where the archive `pair.zip` contains both images.

## Online demo
You can try the method online in the following <a href="https://ipolcore.ipol.im/demo/clientApp/demo.html?id=602">IPOL demo</a>.
