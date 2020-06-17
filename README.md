# Scalable Hashing-Based Network Discovery
Version 1.0.

## About

This repository hosts the code for our IEEE ICDM 2017 paper and follow-up KAIS journal paper on inferring networks from time series efficiently:
> Tara Safavi, Chandra Sripada, Danai Koutra. _Scalable Hashing-Based Network Discovery_. IEEE International Conference on Data Mining, 2017.

> Tara Safavi, Chandra Sekhar Sripada, Danai Koutra: Fast network discovery on sequence data via time-aware hashing. Knowl. Inf. Syst. 61(2): 987-1017 (2019)

*Link to the conference paper*: https://gemslab.github.io/papers/safavi-2017-scalable.pdf
*Link to the journal paper*:  https://gemslab.github.io/papers/safavi-2018-fast.pdf

<p align="center">
<img src="https://github.com/GemsLab/hashing-based-network-discovery/blob/master/overview.png" width="700"  alt="Overview of hashing-based network discovery">
</p>

```
@INPROCEEDINGS{SafaviSK17,
  author    = {T. {Safavi} and C. {Sripada} and D. {Koutra}},
  title     = {Scalable Hashing-Based Network Discovery}, 
  booktitle = {IEEE International Conference on Data Mining (ICDM)}, 
  year      = {2017},
  pages     = {405-414},
  }
```
```
@INPROCEEDINGS{SafaviSK19journal,
  author    = {T. {Safavi} and C. {Sripada} and D. {Koutra}},
  title     = {Fast network discovery on sequence data via time-aware hashing}, 
  journal   = {Knowl. Inf. Syst.},
  volume    = {61},
  number    = {2},
  pages     = {987--1017},
  year      = {2019},
  }
```

## Run
To run, change directories to the ```discovery``` directory and type ```make```.
This will generate a sample graph from synthetic data using the pairwise correlation and window LSH methods.

## Contact
If you have questions about the code, please email Tara Safavi at tsafavi@umich.edu.


