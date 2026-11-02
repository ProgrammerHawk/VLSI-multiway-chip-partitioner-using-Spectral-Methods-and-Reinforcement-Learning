/////////////////////////////////////////////////////////////////////////////////
///
/// Author    : Anirudh Iyengar 
/// Date      : June 15, 2025
///
/// This is an implementation of the k-means++ algorithm of EigenVectors based
/// clustering. Given initial centroids, cluster using Eigenvectors that are
/// orthogonal. Orthogonality is ensure by the caller with a psd matrix.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Eigen/Eigen>
#include <cstdlib>
#include <iostream>
#include <vector>

using ClusterId2Clust = std::vector<std::vector<int>>;
using CentroidVec = std::vector<int>;

class KMeans {
    public:
	KMeans() {}
	virtual ~KMeans() {}

	// kmeans clustering
	// 
	// data: Affinity matrix (in our case eigenvector matrix of the normalized Laplacian)
	// nCenters: Number of clusters
	// Returns:  ordered set of data row indices (id) for each cluster
	// 	     (order is based on distance to cluster center)
	//
	static ClusterId2Clust kmcluster(Eigen::MatrixXf& data, const CentroidVec& iCv, int nCenters);
};
