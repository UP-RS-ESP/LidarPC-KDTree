# LidarPC-KDTree
Comparison of KDTree implementations for Lidar PointClouds (PC) and Structure-from-Motion (SfM) dataset.

One of the core processing steps for irregular PC is to understand the neighborhood for each point (kNN - k-Nearest-Neighbors). This is often done using [KD Trees](https://en.wikipedia.org/wiki/K-d_tree). There exist myriad of implementations for various applications and KD Trees have become an important tool for Deep Learning that have been implemented in [kNN (k-nearest neighbor)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) algorithms. Many of the approaches have been optimized for multi-dimensional datasets (n>5 and up to 50). In the recent months and years, KD-Trees relying on [CUDA](https://en.wikipedia.org/wiki/CUDA) or [OpenCL](https://en.wikipedia.org/wiki/OpenCL) implementations have become more coming and easily approachable through Python or Matlab interfaces.

**Here, we briefly explore existing algorithms and test, which one perform favorable for 3D lidar PC (or SfM PC). We only use three dimensions (x, y, z), but future implementation may rely on four (intensity) or higher dimensional lidar PC. We focus on implementations accessible through Python or C.**

We note that there exist other algorithm and parameter comparisons (e.g. [knn-benchmarking in python](https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/) and [knn-benchmarking](http://mccormickml.com/2017/09/08/knn-benchmarks-part-1/)) and these are very useful and helpful -- but these are neither tailored for lidar/SfM PC nor have been using recent implementations. Most comparison also focus on the general applicability of KD-Tree algorithm and explore the impact of leaf sizes and dimensionality - both parameters do not change for lidar PC.

**A detailed analysis is described and illustrated in [Comparing Python KD-Tree Implementations with Focus on Point Cloud Processing](docs/Compare_KDTree_implementations.pdf).**

## Methods and Approach
We construct the following scenarios:
1. Deriving k=5,10,50,100,500,1000 nearest neighbors from four lidar/SfM point clouds with 14e6, 38e6, 69e6, and 232e6 (million) points.
2. We time the generation of a KD-Tree and the queries separately for each.
3. Searching for neighbors within a given search radius/sphere (not supported by all algorithms).
4. The k-nearest neighbors can be used to estimate point-density or perform further classification on the neighborhood structure of points (e.g., curvature)


| Name | Reference and Documentation | Comments |
|:---|:---|:---|
|scipy.spatial.KDTree| [Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree) | Pure Python implementation of KD tree. Querying is very slow and usage is not suggested. Not used.*Single core CPU processing.* |
|scipy.spatial.cKDTree | [Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html#scipy.spatial.cKDTree) | KDTree implementation in Cython. *Single and Multi-core CPU processing.* |
|sklearn.neighbors.KDTree | [Manual](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html) | KDTree implementation in sklearn. *Single core CPU processing.* |
|pyKDTree | [github page](https://github.com/storpipfugl/pykdtree) and [pypi project page](https://pypi.org/project/pykdtree/#description) |  fast implementation for common use cases (low dimensions and low number of neighbours) for both tree construction and queries. The implementation is based on scipy.spatial.cKDTree and libANN by combining the best features from both and focus on implementation efficiency. *Multi-core CPU processing.*|
| pyflann | [github](https://github.com/primetang/pyflann) | pyflann is the python bindings for [FLANN - Fast Library for Approximate Nearest Neighbors](http://www.cs.ubc.ca/research/flann/) and [FLANN Manual 1.8.4](https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf) *Multi-core CPU processing.* |
| cyflann | [github](https://github.com/dougalsutherland/cyflann) | cyflann is the a cython interface for [FLANN - Fast Library for Approximate Nearest Neighbors](http://www.cs.ubc.ca/research/flann/) and [FLANN Manual 1.8.4](https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf). *Multi-core CPU processing.* |
| NearestNeighbors | [cuml-kNN](https://docs.rapids.ai/api/cuml/nightly/api.html#nearest-neighbors) | GPU implementaiton of knn via [rapidsai](https://github.com/rapidsai), *currently only supports brute-force algorithm and is not competitive* |

# Datasets
## Subset of Pozo catchment: Airborne Lidar data from Santa Cruz Island, California
The airborne point cloud from Santa Cruz Island, California represents a natural terrain without buildings, but lower density. The  dataset contains 3,348,668 points with a point-density of 7.2 pts/m^2 and has been ground-classified using [LAStools](https://rapidlasso.com/lastools/). The points have been colored using an airphoto from the same time as the lidar flight.
The test area is from a small subset of the Pozo catchment in the southwestern part of the island. These data are openly accessibly and available from [opentopography](https://opentopography.org/) and were originally acquired by the USGS in 2010. The geologic and geomorphic environment and setting of the Santa Cruz Island has been described in several peer-reviewed scientific publications (e.g., [Perroy et al., 2010](https://doi.org/10.1016/j.geomorph.2010.01.009),  [Perroy et al., 2012](https://doi.org/10.1080/00045608.2012.715054), [Neely et al., 2017](https://doi.org/10.1002/2017JF004250), [Rheinwalt et al., 2019](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018JF004827), [Clubb et al., 2019](https://doi.org/10.1029/2019JF005025)).


# Results
We ran tests on a AMD Ryzen Threadripper 2970WX 24-Core Processor (2019) with a NVIDIA Titan RTX 24 GB running Ubuntu 18.04 (CUDA 10.1) and a AMD Ryzen 9 3900X 12-Core Processor with a NVIDIA GeForce RTX 2080 SUPER running Ubuntu 20.04 (CUDA 11.0).

## Subset of Pozo catchments (n=3,348,668 points)
A first test using standard single-core and multi core algorithms for n=3,348,668 queries for n=3,348,668 points. Note that the KDTree calculations from _scipy.spatial.KDTree_ have note been included, because they were too slow. Also, for the single core _sklearnKDTree_ approach, no higher number of neighbors have been included (too slow). All results show times in seconds (s) and have been averaged over n=3 runs.

### Comparing single-core sklearnKDTree and multi-core cKDTree
Comparing the traditional and widely used _sklearnKDTree_ (single core) and _cKDTree_ (multi core) approaches we note the following results:
1. The leaf size is an important parameter to speed up single-core querying trees. Depending on point cloud structure, different leaf sizes provide very different results and can improve query times. We note that the default leaf size does not generate useful results for real-world airborne lidar data and that there exists a minimum time representing an optimal leaf size.
2. The _sklearnKDTree_ (single core) is slow on these massive queries. The option `dualtree=True` has been used to speed up processing.
3. _cKDTree_ with `jobs=-1` set for querying outperforms single-core approaches - especially on modern multi-core systems. Leaf size does not have a significant impact on multi-core processing, but some for larger neighborhood queries (k>100).
4. There are minimal difference between different approach. For example. the max. difference between _sklearnKDTree_ and _cKDTree_ is 0.2m and the median difference is 0.0.
5. Comparing _cKDTree_ with 12, 24, and 48 core processors indicates a clear advantage of multi-threading processes. We emphasize that in order to take full advantage of multi-threading processes, an increase in available DRAM is needed (i.e., more cores require more DRAM). We note that _pyKDTree_ has lower peak memory requirement than _cKDTree_.
6. The FLANN (Fast Library for Approximate Nearest Neighbors) family of approaches provides additional advancements, especially for large datasets and massive queries.
7. Initial tests with cuML (CUDA RAPIDS) show that the implemented brute-force approach for nearest neighbor searches is not competitive against the multi-core approaches (_cKDTree_ and _pyKDTree_) and highly optimized FLANN approaches. But there are other processing advantages of data analysis using  CUDA Dataframes (cudf).


![Generation and query times for single-core sklearnKDTree for varying leafsizes. \label{pc_sklearnKDTree_AMD3900X_12cores}](https://github.com/UP-RS-ESP/LidarPC-KDTree/raw/master/docs/figs/pc_sklearnKDTree_AMD3900X_12cores.png)

![The multi-core cKDTree implementation in _scipy.spatial.cKDTree_ performs well - but you need to set the 'jobs=-1' parameter in the query to achieve best results and use all available cores (only during queries). \label{pc_cKDTree_AMD3900X_12cores}](https://github.com/UP-RS-ESP/LidarPC-KDTree/raw/master/docs/figs/pc_cKDTree_AMD3900X_12cores.png)

|  Algorithm     |   Generate KDTree (s) |   Query k=5 (s) |   Query k=10 (s) |   Query k=50 (s) |   Query k=100 (s) |   Query k=500 (s) |   Query k=1000 (s) |
| :--------------|----------------------:|----------------:|-----------------:|-----------------:|------------------:|------------------:|-------------------:|
|  KDTree        |                5.25    |          nan    |           nan    |           nan    |            nan    |            nan    |             nan    |
|  sklearnKDTree |                  1.51 |           36.93 |           nan    |           nan    |            nan    |            nan    |             nan    |
|  **cKDTree**       |                  0.32 |            0.23 |             0.31 |             0.91 |              1.67 |              7.47 |              15.13 |
|  pyKDTree      |                  0.07 |            0.23 |             0.32 |             1.1  |              2.58 |             35.17 |             129.81 |
|  pyflannKDTree |                  0.19 |            0.17 |             0.24 |             0.97 |              2.11 |             12.54 |              27.4  |
|  cyflannKDTree |                  0.26 |            0.2  |             0.26 |             1    |              2.2  |              9.69 |              20.01 |

Table: Comparison of fastest processing times (any leaf size) for all implemented algorithms in seconds. Note that _KDTree_ has not been processed due to the very slow processing times. All times are the average of 3 iterations.


|  Algorithm     |   Generate KDTree (# leafsize) |   Query k=5 (# leafsize) |   Query k=10 (# leafsize) |   Query k=50 (# leafsize) |   Query k=100 (# leafsize) |   Query k=500 (# leafsize) |   Query k=1000 (# leafsize) |
| :--------------|-------------------------------:|-------------------------:|--------------------------:|--------------------------:|---------------------------:|---------------------------:|----------------------------:|
|  KDTree        |                             10 |                        8 |                         8 |                         8 |                          8 |                          8 |                           8 |
|  cKDTree       |                             36 |                       14 |                        16 |                        24 |                         14 |                         38 |                          38 |
|  sklearnKDTree |                             28 |                       12 |                         8 |                         8 |                          8 |                          8 |                           8 |
|  pyKDTree      |                             36 |                       16 |                        20 |                        16 |                         28 |                         26 |                          32 |

Table: Best leaf sizes (fastest times). Note the differences for varying numbers of neighbors.
