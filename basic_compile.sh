#!/bin/sh

g++ -std=c++14 -O3 -IEigen -o spectral_part spectral_partitioner.cpp kmeans.cpp
