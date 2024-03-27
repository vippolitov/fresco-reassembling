# Fresco reassembling
## Overview
This repository contains code that was written as a part of my master's thesis.
Given set of fragments of destroyed fresco, the aim is to recommend pairs of fragments that can be joint in one one fragment along their common edge.

## Pipeline
0) Preprocessing (extending fragments with inpainting model, computing edge coordinates using Canny, etc.)
1) Filtering pairs of fragments that are not similar enough (based on color histograms and color quantization using KNN)
2) Fast coarse fragments aligning for each pair of fragments (using adopted Longest Common Subsequence searching algorithm)
3) Slow refining of obtained alignments (looping over coarse estimation neighbourhood and computing different scores for each option)
4) Filtering false positive predictions (according to triplets formed based on refined pairwise alignments)

## Structure ans usage
Python code is kept in /code/src directory. 
Directory /code contains several notebooks with examples of src usage. Each notebook has number inside its name, this number corresponds to one of the pipeline stages.

## Examples
<p align="center">
  Example of found common edge of two fragments
</p>
<p align="center">
  <img src="https://github.com/vippolitov/fresco-reassembling/blob/main/illustrations/example_common_edge.png" width="700">
</p>


<p align="center">
  Example of found alignment between two fragments
</p>
<p align="center">
  <img src="https://github.com/vippolitov/fresco-reassembling/blob/main/illustrations/example_refined.png" width="500">
</p>


<p align="center">
  Example of triplet formed to filter pair matches without any other fragment to construct triplet
</p>
<p align="center">
  <img src="https://github.com/vippolitov/fresco-reassembling/blob/main/illustrations/example_triplet.png" width="500">
</p>
