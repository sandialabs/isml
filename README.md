# ISML: In Situ Machine Learning

ISML is a python-based framework for automatically detecting spatial and temporal events of interest in situ while running simulations in high performance computing environments. Capturing events that are of interest to scientists in complex, high-fidelity HPC simulations is rarely feasible due to the vast memory requirements of exporting and storing simulation state for every time step. The ISML framework, composed from *signature*, *measure*, and *decision* building blocks with well-defined semantics, is tailored to identify events of interest during simulation in an inexpensive and unsupervised fashion. More details about the ISML framework's methodology can be found in the following publication:

> T. M. Shead, I. K. Tezaur, W. L. Davis IV, M. L. Carlson, D. M. Dunlavy, E. J. Parish, P. J. Blonigan, J. Tencer, F. Rizzi, and H. Kolla. “A Novel In Situ Machine Learning Framework for Intelligent Data Capture and Event Detection”. In: Swaminathan, N., Parente, A. (eds) Machine Learning and Its Application to Reacting Flows. Lecture Notes in Energy, vol 44. Springer, Cham. January, 2023.

## Methodology

The ISML framework for event detection works as follows. A given simulation domain is divided into a set of analysis partitions that correspond to a spatially contiguous subset of mesh points of the simulation domain. For each analysis partition, ISML computes a **signature**, a fixed-length vector representing the simulation state within that partition. Conceptually, a signature is a compressed, low-dimensional representation of an analysis partition's content. Given a set of signatures, ISML can then compute spatial or temporal **measures** to identify events. Measures are functions applied to a set of signatures that detect changes across space or time. Finally, **decisions** are functions used to convert continuous per-analysis-partition measures into boolean values to indicate whether the partitions should be flagged as containing events of interest for the current timestep.

![framework_diagram](https://github.com/sandialabs/isml/assets/12075822/4558a17c-5c26-4eb6-adc4-3f29371989b2)

## Available Signatures, Measure, and Decisions

Table 1: Signature functions
| Signature | Description |
| --------- | ----------- |
| `fieda` | Vector of feature importance values described in Ling et al. (2017) |
| `fmm` | Vector based on the Feature Moment Metric algorithm in Konduri et al. (2018) |
| `mean` | Vector of mean values for each simulation feature |
| `minimax` | Vector of minimum and maximum values for each simulation feature |
| `quartile` | Vector of quartile boundaries for each simulation feature (a generalization of minimax) |
| `svd` | Vector of singular values computed using an SVD on the flattened partition state matrix |

Table 2: Spatial measures
| Measure | Description |
| -------------- | ----------- |
| `dbscan` | Uses DBSCAN (Ester et al. 1996) to flag outlier signatures as events |
| `m1` | M1 metric described in Ling et al. (2017) |
| `m1-hellinger` | Modified version of the Hellinger distance used as a spatial metric, described in Konduri et al. (2018) |
| `msd` | Compares the distance between one signature and the mean signature for all partitions |
| `sigscal` | Normalizes the signature matrix using the product of the inverse of each signature and measures the ability of each signature to drive values in the product to zero |

Table 3: Temporal measures
| Measure | Description |
| --------- | ----------- |
| `changefreq` | Counts the number of changes (dramatic increases or decreases across any two timesteps within a temporal window) |
| `maxchange` | Uses the maximum change across any timesteps within the current temporal window |
| `mse` | Based on mean squared error between the current temporal block (S × T matrix) and previous temporal block |
| `psd` | Estimates power spectral density (power spectrum) for each feature within the temporal block (S × T matrix) using Welch’s method (Welch 1967) |
| `svd` | Ratio of the largest non-zero singular value to the smallest non-zero singular value |

Table 4: Decision functions
| Measure | Description |
| --------- | ----------- |
| `threshold` | Flag a partition when its measure exceeds a fixed threshold |
| `percentile` | Flag a partition when its measure exceeds the nth percentile of the measure for all partitions |
| `memory` | Decision modifier which continues to flag a partition for a fixed number of timesteps after another decision function has flagged it |
