# Fresco reassembling
## Overview
This repository contains code that was written as a part of my master's thesis.
Given set of fragments of destroyed fresco, the aim is to recommend pairs of fragments that can be joint in one one fragment along their common edge.

## Pipeline
1) Filtering pairs of fragments that are not similar enough
2) Fast coarse fragments aligning for each pair of fragments using adopted Longest Common Subsequence searching algorithm
3) Slow refining of obtained alignments
4) Filtering false positive predictions (according to triplets formed based on refined pairwise alignments)

## Usage
Code directory contains several notebooks with examples of code usage. Each notebook has number inside its name, this number corresponds to one of the pipeline stages.

## Examples
<p align="center">
  Example of found common edge of two fragments
</p>
<p align="center">
  <img src="https://github.com/Ippolitov2909/fresco-reassembling/blob/main/illustrations/example_common_edge.png" with="800">
</p>


<p align="center">
  Example of found alignment between two fragments
</p>
<p align="center">
  <img src="https://github.com/Ippolitov2909/fresco-reassembling/blob/main/illustrations/example_refined.png" with="800">
</p>
