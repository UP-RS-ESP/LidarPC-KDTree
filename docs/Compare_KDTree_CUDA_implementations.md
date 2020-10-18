---
title: "Comparing Python KD-Tree Implementations with Focus on Point Cloud Processing"
subtitle: What is the most effective way to process lidar and SfM point clouds?
author: "Bodo Bookhagen, [bodo.bookhagen@uni-potsdam.de](bodo.bookhagen@uni-potsdam.de), University of Potsdam"
date: "Oct-18-2020"
footnotes-pretty: true
listings-disable-line-numbers: false
titlepage: true
toc-own-page: true
book: false
header-left: "Comparing Python KD-Tree Implementations"
footer-left: "University of Potsdam - Bodo Bookhagen"
logo-width: 350
disable-header-and-footer: false
lang: "en"
header-includes: |
  \usepackage[table]{xcolor}
  \usepackage{booktabs,caption,threeparttable}
  \captionsetup{labelfont=bf, justification=raggedright, singlelinecheck=false}

...
\tableofcontents
\newpage
\listoffigures
\newpage
\listoftables
\newpage

# LidarPC-KDTree
Comparison of KDTree implementations for Lidar PointClouds (PC) and Structure-from-Motion (SfM) dataset.

One of the core processing steps for irregular PC is to understand the neighborhood for each point (kNN - k-Nearest-Neighbors). This is often done using [KD Trees](https://en.wikipedia.org/wiki/K-d_tree). There exist myriad of implementations for various applications and KD Trees have become an important tool for Deep Learning that have been implemented in [kNN (k-nearest neighbor)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) algorithms. Many of the approaches have been optimized for multi-dimensional datasets (n>5 and up to 50). In the recent months and years, KD-Trees relying on [CUDA](https://en.wikipedia.org/wiki/CUDA) or [OpenCL](https://en.wikipedia.org/wiki/OpenCL) implementations have become more coming and easily approachable through Python or Matlab interfaces.

**Here, we briefly explore existing algorithms and test, which one perform favorable for 3D lidar PC (or SfM PC). We only use three dimensions (x, y, z), but future implementation may rely on four (intensity) or higher dimensional lidar PC. We focus on implementations accessible through Python or C.**

We note that there exist other algorithm and parameter comparisons (e.g. [knn-benchmarking in python](https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/) and [knn-benchmarking](http://mccormickml.com/2017/09/08/knn-benchmarks-part-1/)) and these are very useful and helpful -- but these are neither tailored for lidar/SfM PC nor have been using recent implementations. Most comparison also focus on the general applicability of KD-Tree algorithm and explore the impact of leaf sizes and dimensionality - both parameters do not change for lidar PC.

# Environment Installation
See [miniconda installation instructions](https://up-rs-esp.github.io/posts/2020/10/conda-install/) to setup an environment for processing. The installation of the python codes is done through conda on a Ubuntu 18.04 LTS system (also tested on 20.04 LTS).

Conda installation:
```
cconda create -y -n PC_py3 -c anaconda -c conda-forge -c defaults ipython spyder python=3.8 gdal=3 numpy scipy dask h5py pandas pytables hdf5 cython matplotlib tabulate scikit-learn pykdtree pyflann cyflann scikit-image opencv ipywidgets scikit-learn gmt=6* imagemagick
```
Next:
```
conda activate PC_py3
pip install laspy
pip install tables
```

# Methods and Approach
We construct the following scenarios:
1. Deriving k=5,10,50,100,500,1000 nearest neighbors from four lidar/SfM point clouds with 14e6, 38e6, 69e6, and 232e6 (million) points.
2. We time the generation of a KD-Tree and the queries separately for each.
3. Searching for neighbors within a given search radius/sphere (not supported by all algorithms).
4. The k-nearest neighbors can be used to estimate point-density or perform further classification on the neighborhood structure of points (e.g., curvature)

**We note that we query the tree with all points (e.g., k=50 neighbors for all points) and thus create large queries for neighborhood statistical analysis.**

An incomplete list of available algorithms and implementations. *Note: We have not used all of them for the tests, because some implementations are very slow and mostly for instructive/teaching purposes.*
Also, in all instances we have used the standard options and parameters, but these may not always be the most useful ones.

| Name | Reference and Documentation | Comments |
|:---|:---|:---|
|scipy.spatial.KDTree| [Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree) | Pure Python implementation of KD tree. Querying is very slow and usage is not suggested. Not used.*Single core CPU processing.* |
|scipy.spatial.cKDTree | [Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html#scipy.spatial.cKDTree) | KDTree implementation in Cython. *Single and Multi-core CPU processing.* |
|sklearn.neighbors.KDTree | [Manual](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html) | KDTree implementation in sklearn. *Single core CPU processing.* |
|pyKDTree | [github page](https://github.com/storpipfugl/pykdtree) and [pypi project page](https://pypi.org/project/pykdtree/#description) |  fast implementation for common use cases (low dimensions and low number of neighbours) for both tree construction and queries. The implementation is based on scipy.spatial.cKDTree and libANN by combining the best features from both and focus on implementation efficiency. *Multi-core CPU processing.*|
| pyflann | [github](https://github.com/primetang/pyflann) | pyflann is the python bindings for [FLANN - Fast Library for Approximate Nearest Neighbors](http://www.cs.ubc.ca/research/flann/) and [FLANN Manual 1.8.4](https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf) *Multi-core CPU processing.* |
| cyflann | [github](https://github.com/dougalsutherland/cyflann) | cyflann is the a cython interface for [FLANN - Fast Library for Approximate Nearest Neighbors](http://www.cs.ubc.ca/research/flann/) and [FLANN Manual 1.8.4](https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf). *Multi-core CPU processing.* |
| NearestNeighbors | [cuml-kNN](https://docs.rapids.ai/api/cuml/nightly/api.html#nearest-neighbors) | GPU implementaiton of knn via [rapidsai](https://github.com/rapidsai), *currently only supports brute-force algorithm and is not competitive* |

# Test datasets
## Lidar and SfM Point clouds from the University of Potsdam Campus Golm

The dense and high-resolution point clouds have been generated between 2018-2020 for parts of the University of Potsdam Campus Golm. These represent mixed-urban environments with building and vegetation. The files are too large for github and have been stored on Dropbox (links provided below).

| Name | PC Type | # of points | Point Density [pts/m2] | Link
|:---|:---|---:|---:|
| Golm_May06_2018_Milan_UTM33N_WGS84_6digit_cl_clip.laz | Airborne Lidar | 14,437,532 | 61 | [Dropbox Link](https://www.dropbox.com/s/ocfj1kpfsap63pz/ALS_Golm_May06_2018_Milan_UTM33N_WGS84_6digit_cl_clip.laz?dl=0)
|mavicpro2_nadir_15deg_highq_dense_PC_10cm_aligned2.laz | Mavic Pro2, Agisoft Photoscan, high quality processing setting, images from nadiar and 15 degree angle taken | 38,334,551 | 219 | [Dropbox Link](https://www.dropbox.com/s/yxgpmxmz5bv2pnh/mavicpro2_nadir_15deg_highq_dense_PC_10cm_aligned2.laz?dl=0)
| Golm_sept06_2019_highquality_agressivefiltering_aligned.laz | Mavic Pro2, Agisoft Photoscan, high quality processing setting| 69,482,218 | 707 | [Dropbox Link](https://www.dropbox.com/s/ydbez1tsnqiacjk/Golm_sept06_2019_highquality_agressivefiltering_aligned.laz?dl=0)
| inspire2_1031cameras_highq_dense_pc.laz | Inspire 2 | 232,269,911 | 988 | [Dropbox Link 2 GB!](https://www.dropbox.com/s/r8kjcxxs01a9g5h/inspire2_1031cameras_highq_dense_pc.laz?dl=0)

## Airborne Lidar data from Santa Cruz Island, California
The airborne point cloud from Santa Cruz Island, California represents a natural terrain without buildings, but lower density. The  dataset contains 3,348,668 points with a point-density of 7.2 pts/m^2 and has been ground-classified using [LAStools](https://rapidlasso.com/lastools/). The points have been colored using an airphoto from the same time as the lidar flight.
The test area is from a small subset of the Pozo catchment in the southwestern part of the island. These data are openly accessibly and available from [opentopography](https://opentopography.org/) and were originally acquired by the USGS in 2010. The geologic and geomorphic environment and setting of the Santa Cruz Island has been described in several peer-reviewed scientific publications (e.g., [Perroy et al., 2010](https://doi.org/10.1016/j.geomorph.2010.01.009),  [Perroy et al., 2012](https://doi.org/10.1080/00045608.2012.715054), [Neely et al., 2017](https://doi.org/10.1002/2017JF004250), [Rheinwalt et al., 2019](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018JF004827), [Clubb et al., 2019](https://doi.org/10.1029/2019JF005025)).


# Results
We ran tests on a AMD Ryzen Threadripper 2970WX 24-Core Processor (2019) with a NVIDIA Titan RTX 24 GB running Ubuntu 18.04 (CUDA 10.1) and a AMD Ryzen 9 3900X 12-Core Processor with a NVIDIA GeForce RTX 2080 SUPER running Ubuntu 20.04 (CUDA 11.0).

## Subset of Pozo catchments (n=3,348,668 points)
A first test using standard single-core and multi-core algorithms for n=3,348,668 queries for n=3,348,668 points. Note that the KDTree calculations from *scipy.spatial.KDTree* have note been included, because they were too slow. Also, for the single-core *sklearnKDTree* approach, no higher number of neighbors have been included (too slow). All results show times in seconds (s) and have been averaged over n=3 runs.

### Comparing single-core sklearnKDTree and multi-core cKDTree
Comparing the traditional and widely used _sklearnKDTree_ (single core) and _cKDTree_ (multi core) approaches we note the following results:
1. The leaf size is an important parameter to speed up single-core querying trees. Depending on point cloud structure, different leaf sizes provide very different results and can improve query times. We note that the default leaf size does not generate useful results for real-world airborne lidar data and that there exists a minimum time representing an optimal leaf size (cf. Figure \ref{pc_sklearnKDTree_AMD3900X_12cores}).
2. The _sklearnKDTree_ (single core) is slow on these massive queries. The option `dualtree=True` has been used to speed up processing.
3. _cKDtree_ with `jobs=-1` set for querying outperforms single-core approaches - especially on modern multi-core systems. Leaf size does not have a significant impact on multi-core processing, but some for larger neighborhood queries (k>100) (cf. Figures \ref{pc_cKDTree_AMD3900X_12cores} and {pc_sklearnKDTree_cKDTree_k5_AMD3900X_12cores}).
4. There are minimal difference between different approach. For example. the max. difference between _sklearnKDTree_ and _cKDTree_ is 0.2m and the median difference is 0.0 (see Figure \ref{pc_mean_distance_sklearnKDTree_cKDTree_leafsize16_k5_AMD3900X_12cores}).
5. Comparing _cKDTree_ with 12, 24, and 48 core processors indicates a clear advantage of multi-threading processes. (cf. Figure \ref{pc_pyKDTree_k5_k50_vcores}). We emphasize that in order to take full advantage of multi-threading processes, an increase in available DRAM is needed (i.e., more cores require more DRAM). We note that _pyKDTree_ has lower peak memory requirement than _cKDTree_.
6. The FLANN (Fast Library for Approximate Nearest Neighbors) family of approaches provides additional advancements, especially for large datasets and massive queries.


[Generation and query times for single-core sklearnKDTree for varying leafsizes. \label{pc_sklearnKDTree_AMD3900X_12cores}](figs/pc_sklearnKDTree_AMD3900X_12cores.png)

[The multi-core cKDTree implementation in scipy.spatial.cKDTree performs very well - but you need to set the 'jobs=-1' parameter in the query to achieve best results and use all available cores. \label{pc_cKDTree_AMD3900X_12cores}](figs/pc_cKDTree_AMD3900X_12cores.png)

[Direct comparison of single-core sklearnKDTree and multi-core cKDTree approaches.  \label{pc_sklearnKDTree_cKDTree_k5_AMD3900X_12cores}](figs/pc_sklearnKDTree_cKDTree_k5_AMD3900X_12cores.png)

[Direct comparison of single-core sklearnKDTree and multi-core cKDTree approaches.  \label{pc_mean_distance_sklearnKDTree_cKDTree_leafsize16_k5_AMD3900X_12cores}](figs/pc_mean_distance_sklearnKDTree_cKDTree_leafsize16_k5_AMD3900X_12cores.png)

### Comparing pyKDTree and cKDTree for 12, 24, and 40 cores
[Varying leaf size for pyKDTree and cKDTree. The leafsize does not have an significant impact on querying time. There is an advantage of multiple cores for higher ks. The cKDTree algorithm appears to be the fastest for searches of large k \label{pc_pyKDTree_k5_k50_vcores}](figs/pc_pyKDTree_k5_k50_vcores.png)


```python
#Read in data:
#kant: 3.9 GHz
inps.cpuname= 'AMDRyzen_3900X_12cores'

#macon: 2.9 GHz
inps.cpuname= 'AMDRyzen_2970WX_24cores'

#aconcagua: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz, 2x20 cores, 2x40 threads
inps.cpuname= 'Xeon_Gold6230_40cores'

inps.cpuname= 'AMDRyzen_3900X_12cores'
pc_pyKDTree_time_df_12cores = pd.read_hdf('pc_pyKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_pyKDTree_time_df')
pc_cKDTree_time_df_12cores = pd.read_hdf('pc_cKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_cKDTree_time_df')

inps.cpuname= 'AMDRyzen_2970WX_24cores'
pc_pyKDTree_time_df_24cores = pd.read_hdf('pc_pyKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_pyKDTree_time_df')
pc_cKDTree_time_df_24cores = pd.read_hdf('pc_cKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_cKDTree_time_df')

inps.cpuname= 'Xeon_Gold6230_40cores'
pc_pyKDTree_time_df_40cores = pd.read_hdf('pc_pyKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_pyKDTree_time_df')
pc_cKDTree_time_df_40cores = pd.read_hdf('pc_cKDTree_time_df_%s.hdf'%(inps.cpuname), key='pc_cKDTree_time_df')

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(leafrange, (pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k5_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 'x-', c='k', label='create + query pyKDTree k=5, 12 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k5_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 'o-', c='k', label='create + query pyKDTree k=5, 24 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k5_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 's-', c='k', label='create + query pyKDTree k=5, 40 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k50_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 'x-', c='darkblue', label='create + query pyKDTree k=50, 12 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k50_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 'o-', c='darkblue', label='create + query pyKDTree k=50, 24 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k50_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 's-', c='darkblue', label='create + query pyKDTree k=50, 40 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_12cores['pc_query_pyKDTree_k500_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_12cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 'x-', c='darkred', label='create + query pyKDTree k=500, 12 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_24cores['pc_query_pyKDTree_k500_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_24cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 'o-', c='darkred', label='create + query pyKDTree k=500, 24 cores')
ax.plot(leafrange, (pc_pyKDTree_time_df_40cores['pc_query_pyKDTree_k500_time'])/inps.nr_of_repetitions+pc_pyKDTree_time_df_40cores['pc_generate_pyKDTree_time']/inps.nr_of_repetitions_generate, 's-', c='darkred', label='create + query pyKDTree k=500, 40 cores')
ax.plot(leafrange, (pc_cKDTree_time_df_12cores['pc_query_cKDTree_k500_time'])/inps.nr_of_repetitions+pc_cKDTree_time_df_12cores['pc_generate_cKDTree_time']/inps.nr_of_repetitions_generate, 'x-', c='red', label='create + query cKDTree k=500, 12 cores')
ax.plot(leafrange, (pc_cKDTree_time_df_24cores['pc_query_cKDTree_k500_time'])/inps.nr_of_repetitions+pc_cKDTree_time_df_24cores['pc_generate_cKDTree_time']/inps.nr_of_repetitions_generate, 'o-', c='red', label='create + query cKDTree k=500, 24 cores')
ax.plot(leafrange, (pc_cKDTree_time_df_40cores['pc_query_cKDTree_k500_time'])/inps.nr_of_repetitions+pc_cKDTree_time_df_40cores['pc_generate_cKDTree_time']/inps.nr_of_repetitions_generate, 's-', c='red', label='create + query cKDTree k=500, 40 cores')
ax.set_title('pyKDTree+cKDTree and various cores comparison')
ax.grid()
ax.set_xlabel('leafsize')
ax.set_ylabel('Query time (s)')
ax.set_xlim([8,30])
ax.set_xticks(np.arange(8,30,step=2))
ax.set_yscale('log')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.7])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=False, shadow=False, ncol=2,fontsize=8)
fig.savefig('figs/pc_pyKDTree_k5_k50_vcores.png', dpi=300, orientation='landscape')
```

### Comparing multi-core cKDTree and multi-core pyKDTree approaches
We note the following results:

1. The leaf size is an important parameter to speed up single-core querying trees. Depending on point cloud structure, different leaf sizes provide very different results. *We note that the default leaf size does not generate useful results for real-world airborne lidar data and that there exists a minimum time representing an optimal leaf size.*
2. For multi-core algorithms, leaf size does not affect speed in the same degree.


### Comparing multi-core cKDTree and multi-core FLANN (Fast Library for Approximate Nearest Neighbors) approaches

|  Algorithm     |   Generate KDTree (s) |   Query k=5 (s) |   k=10 (s) |   k=50 (s) | k=100 (s) |   k=500 (s) | k=1000 (s) |
| :--------------|----------------------:|----------------:|-----------:|-----------:|----------:|------------:|-----------:|
  KDTree | 13.71 | *not run* |  *not run* |  *not run* |
  |  cKDTree       |  5.37 | 4.52 |  6.4  | 20.18 |
  |  sklearnKDTree |  6.17 | 8.04 | 10.12 | 27.08 |
  |  pyKDTree      |  0.35 | 0.65 |  0.91 |  4.03 |
  |  pyflannKDTree |  0.78 | 0.59 |  0.86 |  4.33 |
  |  cyflannKDTree |  0.83 | 0.59 |  0.91 |  3.53 |


# Installation
## Prerequisites
Before running, ensure that you have an up-to-date Python environment. For the sake of compatibility and comparability, we rely on the following `conda` environment. We realize that timing may change in the future, if versions are updated, but the order-of-magnitude comparison should still hold. All tests have been performed in Ubuntu 18.04, but other systems (including Mac OS X and Windows) should work equally well. Note that CUDA implementation on Mac OS X may be limited.

You will need to have a working CUDA environment. Currently, we achieve the best results with CUDA 9.2, but likely will switch to CUDA 10.x soon. Find out about your CUDA version with `nvidia-smi` or `nvcc --version`. Please see relevant webpages to get your proper NVIDIA driver and CUDA compilation tools running.

Install your `conda` environment (we rely on [miniconda](https://docs.conda.io/en/latest/miniconda.html), see also a [blog](https://bodobookhagen.github.io/posts/2018/12/conda-install/) entry):

```bash
conda create -y -n PC_py36 -c anaconda -c conda-forge ipython spyder python=3 \
  gdal=3 numpy scipy dask h5py pandas pytables hdf5 cython matplotlib tabulate \
  scikit-learn pyflann cyflann
conda activate PC_py3
pip install laspy
conda install -y -c nvidia -c rapidsai -c numba -c conda-forge -c pytorch \
  -c defaults cudf=0.8 cuml=0.8 cugraph=0.8 python=3.6 cudatoolkit=9.2
```

Alternatively, you can create a separate environment for CUDA processing, if you plan to do only GPU/CUDA processing:
```bash
conda create -y -n PC_py36_cuda -c nvidia -c rapidsai -c numba -c conda-forge \
  -c pytorch -c defaults cudf=0.8 cuml=0.8 cugraph=0.8 python=3.6 cudatoolkit=9.2
```
## Example codes from LidarPC-KDTree

# Implementation
