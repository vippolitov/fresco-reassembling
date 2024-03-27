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
