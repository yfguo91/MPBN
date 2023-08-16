# Membrane Potential Batch Normalization for Spiking Neural Networks

Official implementation of [MPBN ICCV2023](https://arxiv.org/abs/2308.06787).

## Introduction

The spiking neuron is much more complex with the spatio-temporal dynamics. The regulated data flow after the BN layer will be disturbed again by the membrane potential updating operation before the firing function, i.e., the nonlinear activation. Therefore, we advocate adding another BN layer before the firing function to normalize the membrane potential again, called MPBN.

### Dataset

The dataset will be download automatically.

## Get Started

```
cd imloss
python main_train.py --spike --step 4 
```

## Citation

```bash
@inproceedings{
guo2022imloss,
title={{IM}-Loss: Information Maximization Loss for Spiking Neural Networks},
author={Yufei Guo and Yuanpei Chen and Liwen Zhang and Xiaode Liu and Yinglei Wang and Xuhui Huang and Zhe Ma},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022}
}
```
