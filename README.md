# LidarPC-KDTree
Comparison of KDTree implementations for Lidar PointClouds (PC) and Structure-from-Motion (SfM) dataset.

One of the core processing steps for irregular PC is to understand the neighborhood for each point. This is often done using [KD Trees](https://en.wikipedia.org/wiki/K-d_tree). There exist myriad of implementations for various applications and KD Trees have become an important tool for Deep Learning that have been implemented in [kNN (k-nearest neighbor)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) algorithms. Many of the approaches have been optimized for multi-dimensional datasets (n>5 and up to 50). In the recent months and years, KD-Trees relying on [CUDA](https://en.wikipedia.org/wiki/CUDA) or [OpenCL](https://en.wikipedia.org/wiki/OpenCL) implementations have become more coming and easily approachable through Python or Matlab interfaces.

**Here, we briefly explore existing algorithms and test, which one perform favorable for 3D lidar PC (or SfM PC). We only use three dimensions (x, y, z), but future implementation may rely on four (intensity) or higher dimensional lidar PC. We focus on implementations accessible through Python or C.**

We note that there exist other algorithm and parameter comparisons (e.g. [knn-benchmarking in python](https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/) and [knn-benchmarking](http://mccormickml.com/2017/09/08/knn-benchmarks-part-1/)) and these are very useful and helpful -- but these are neither tailored for lidar/SfM PC nor have been using recent implementations. Most comparison also focus on the general applicability of KD-Tree algorithm and explore the impact of leaf sizes and dimensionality - both parameters do not change for lidar PC.

## Approach
We time the generation of a KD-Tree and the queries separately. We first generated the search tree and then query the k=5,10,50,100,500,1000 closest neighbors (kNN) for each point. These calculation, for example, can be used to estimate point-density or perform further classification on the neighborhood structure of points.

We use the following algorithms:

| Name | Reference and Documentation | Comments |
|:---|:---|:---|
|scipy.spatial.KDTree| [Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree) | Pure Python implementation of KD tree |
|scipy.spatial.cKDTree | [Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html#scipy.spatial.cKDTree) | KDTree implementation in Cython |
|sklearn.neighbors.KDTree | [Manual](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html) | KDTree implementation in sklearn |
|pyKDTree | [github page](https://github.com/storpipfugl/pykdtree) and [pypi project page](https://pypi.org/project/pykdtree/#description) |  fast implementation for common use cases (low dimensions and low number of neighbours) for both tree construction and queries. The implementation is based on scipy.spatial.cKDTree and libANN by combining the best features from both and focus on implementation efficiency.
| pyflann | [github](https://github.com/primetang/pyflann) | pyflann is the python bindings for [FLANN - Fast Library for Approximate Nearest Neighbors](http://www.cs.ubc.ca/research/flann/)
| cyflann | [github](https://github.com/dougalsutherland/cyflann) | cyflann is the a cythin interface for [FLANN - Fast Library for Approximate Nearest Neighbors](http://www.cs.ubc.ca/research/flann/)
| kNN-CUDA | ??? | GPU implementaiton of knn


# Preliminary results
A first test using various algorithms is shown in the table below. Note that the KDTree calculations from *scipy.spatial.KDTree* have note been included, because they are too slow. All results show times in seconds (s).

|Algorithm | Generate KDTree (s) |Query k=5 (s) |  Query k=10 (s) | Query k=50 (s)|
|:---|---:|---:|---:|---:|
KDTree        |13.695143  |# |       # |       #
cKDTree        |5.555057  | 5.109167 | 7.070755  |  20.352178
sklearnKDTree  |6.576532 | 12.202257 | 18.981463| 38.920412
pyKDTree       |0.448567  | 1.294014| 1.828395 |  8.235128
pyflannKDTree  |0.979680  | 0.840919 |1.288056 |  4.726238
cyflannKDTree  |0.855574  | 0.859662| 1.236899 |  4.833241

# Test datasets
## Airborne Lidar data from Santa Cruz Island, California
We perform example calculations with various algorithms and approaches on a point cloud from the Santa Cruz Island in southern California. We use a small subset of the Pozo catchment in the southwestern part of the island. These data are openly accessibly and available from [opentopography](https://opentopography.org/) and were originally acquired by the USGS. The geologic and geomorphic environment and setting of the Santa Cruz Island has been described in several peer-reviewed scientific publications (e.g., [Perroy et al., 2010](https://doi.org/10.1016/j.geomorph.2010.01.009),  [Perroy et al., 2012](https://doi.org/10.1080/00045608.2012.715054), [Neely et al., 2017](https://doi.org/10.1002/2017JF004250), [Rheinwalt et al., 2019](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018JF004827), [Clubb et al., 2019](https://doi.org/10.1029/2019JF005025)).

The dataset contains 3,348,668 points with a point-density of 7.2 pts/m^2 and has been ground-classified using [LAStools](https://rapidlasso.com/lastools/). The points have been colored using an airphoto from the same time as the lidar flight.

## Airborne lidar data from Campus Golm (University of Potsdam)

## Structure-from-Motion based data from Campus Golm (University of Potsdam)

# Implementation
Before running, ensure that you have an up-to-date Python environment. For the sake of compatibility and comparability, we rely on the following `conda` environment. We realize that timing may change in the future, if versions are updated, but the order-of-magnitude comparison should still hold. All tests have been performed in Ubuntu 18.04, but other systems (including Mac OS X and Windows) should work equally well. Note that CUDA implementation on Mac OS X may be limited.

Install your `conda` environment (we rely on [miniconda](https://docs.conda.io/en/latest/miniconda.html), see also a [blog](https://bodobookhagen.github.io/posts/2018/12/conda-install/) entry):
