/////////////////////////////////////////////////////////////////////////////////
///
/// Author    : Anirudh Iyengar 
/// Date      : June 15, 2025
///
/// This is an implementation of the k-means++ algorithm of EigenVectors based
/// clustering. Given initial centroids, cluster using Eigenvectors that are
/// orthogonal. Orthogonality is ensure by the caller with a psd matrix.
/////////////////////////////////////////////////////////////////////////////////

#include "kmeans.h"

#include <map>
#include <random>
#include <limits>
#include <algorithm>
#include <cmath>

// kmeans clustering - uses Eigen library
// data         Affinity matrix
// nCenters	Number of clusters
// Returns      ordered set of data row indices (the path id) for each cluster
// (order is based on distance to cluster center)

ClusterId2Clust KMeans::kmcluster(Eigen::MatrixXf& data, const CentroidVec& iCv, int nCenters)
{
    int nDims = data.cols();
    int nData = data.rows();

    size_t icvSz = iCv.size();
    assert(nCenters > 0 && nData >= nCenters);

    Eigen::VectorXf dataSqNorms = data.rowwise().squaredNorm();

    double bestGlobalErr = std::numeric_limits<double>::max();
    Eigen::VectorXi bestAssignments = Eigen::VectorXi::Zero(nData);
    Eigen::VectorXf bestMinVals = Eigen::VectorXf::Zero(nData);

    constexpr int nRestarts = 12;
    constexpr int niters = 300;
    std::mt19937 rng(42);

    for (int restart = 0; restart < nRestarts; ++restart) {
        Eigen::MatrixXf centers = Eigen::MatrixXf::Zero(nCenters, nDims);
        std::vector<bool> chosen(nData, false);

        // K-Means++ (D^2) initialization
        int firstCenter = 0;
        if (restart == 0 && icvSz > 0 && iCv[0] >= 0 && iCv[0] < nData) {
            firstCenter = iCv[0];
        } else {
            std::uniform_int_distribution<int> uni(0, nData - 1);
            firstCenter = uni(rng);
        }
        centers.row(0) = data.row(firstCenter);
        chosen[firstCenter] = true;

        Eigen::VectorXf minDistSq(nData);
        for (int k = 0; k < nData; ++k) {
            minDistSq[k] = (data.row(k) - centers.row(0)).squaredNorm();
        }

        for (int c = 1; c < nCenters; ++c) {
            double sumDistSq = 0.0;
            for (int k = 0; k < nData; ++k) {
                if (!chosen[k]) {
                    sumDistSq += minDistSq[k];
                }
            }

            int nextCenter = -1;
            if (sumDistSq > 1e-12) {
                if (restart == 0) {
                    // Deterministic farthest-point seeding on first restart
                    float maxD = -1.0f;
                    for (int k = 0; k < nData; ++k) {
                        if (!chosen[k] && minDistSq[k] > maxD) {
                            maxD = minDistSq[k];
                            nextCenter = k;
                        }
                    }
                } else {
                    std::uniform_real_distribution<double> dist(0.0, sumDistSq);
                    double r = dist(rng);
                    double accum = 0.0;
                    for (int k = 0; k < nData; ++k) {
                        if (!chosen[k]) {
                            accum += minDistSq[k];
                            if (accum >= r) {
                                nextCenter = k;
                                break;
                            }
                        }
                    }
                }
            }

            if (nextCenter < 0) {
                for (int k = 0; k < nData; ++k) {
                    if (!chosen[k]) {
                        nextCenter = k;
                        break;
                    }
                }
            }

            centers.row(c) = data.row(nextCenter);
            chosen[nextCenter] = true;

            for (int k = 0; k < nData; ++k) {
                float d2 = (data.row(k) - centers.row(c)).squaredNorm();
                if (d2 < minDistSq[k]) {
                    minDistSq[k] = d2;
                }
            }
        }

        Eigen::MatrixXf oldCenters(nCenters, nDims);
        Eigen::VectorXi assignments = Eigen::VectorXi::Zero(nData);
        Eigen::VectorXf minVals = Eigen::VectorXf::Zero(nData);
        double oldErrVal = 0.0;

        for (int n = 0; n < niters; ++n) {
            oldCenters = centers;

            // Fast vectorized squared Euclidean distance matrix:
            // ||x_k - c_j||^2 = ||x_k||^2 + ||c_j||^2 - 2 * x_k . c_j^T
            Eigen::VectorXf centerSqNorms = centers.rowwise().squaredNorm();
            Eigen::MatrixXf d2 = (-2.0f * (data * centers.transpose())).colwise() + dataSqNorms;
            d2.rowwise() += centerSqNorms.transpose();

            std::vector<int> counts(nCenters, 0);
            for (int k = 0; k < nData; ++k) {
                Eigen::Index bestJ = 0;
                float mv = d2.row(k).minCoeff(&bestJ);
                minVals[k] = std::max(0.0f, mv);
                assignments[k] = static_cast<int>(bestJ);
                counts[bestJ]++;
            }

            // Handle any empty clusters by splitting the point with maximum residual error
            for (int j = 0; j < nCenters; ++j) {
                if (counts[j] == 0) {
                    int farthestIdx = 0;
                    float maxErr = -1.0f;
                    for (int k = 0; k < nData; ++k) {
                        if (counts[assignments[k]] > 1 && minVals[k] > maxErr) {
                            maxErr = minVals[k];
                            farthestIdx = k;
                        }
                    }
                    counts[assignments[farthestIdx]]--;
                    assignments[farthestIdx] = j;
                    counts[j] = 1;
                    minVals[farthestIdx] = 0.0f;
                }
            }

            // Recompute cluster centers
            centers.setZero();
            for (int k = 0; k < nData; ++k) {
                centers.row(assignments[k]) += data.row(k);
            }
            for (int j = 0; j < nCenters; ++j) {
                if (counts[j] > 0) {
                    centers.row(j) /= static_cast<float>(counts[j]);
                }
            }

            double errVal = minVals.sum();
            double ediff = std::fabs(oldErrVal - errVal);
            double cdiff = (centers - oldCenters).cwiseAbs().maxCoeff();

            if (n > 1 && cdiff < 1e-7 && ediff < 1e-7) {
                break;
            }
            oldErrVal = errVal;
        }

        double totalErr = minVals.sum();
        if (totalErr < bestGlobalErr) {
            bestGlobalErr = totalErr;
            bestAssignments = assignments;
            bestMinVals = minVals;
        }
    }

    //------- kmeans done ---------

    // Find the item closest to the center for each cluster
    ClusterId2Clust clusteredItems(nCenters);
    for (int j = 0; j < nCenters; ++j) {
        std::multimap<double, int> cluster;
        for (int k = 0; k < nData; ++k) {
            if (bestAssignments[k] == j) {
                cluster.insert(std::make_pair(static_cast<double>(bestMinVals[k]), k));
            }
        }
        std::vector<int>& units = clusteredItems[j];
        units.reserve(cluster.size());
        for (auto it = cluster.begin(); it != cluster.end(); ++it) {
            units.push_back(it->second);
        }
    }
    return clusteredItems;
}
