# Membrane Potential Batch Normalization for Spiking Neural Networks

Official implementation of [Membrane Potential Batch Normalization for Spiking Neural Networks (ICCV2023)](https://arxiv.org/abs/2308.06787).

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
@article{guo2023membrane,
      title={Membrane Potential Batch Normalization for Spiking Neural Networks}, 
      author={Yufei Guo and Yuhan Zhang and Yuanpei Chen and Weihang Peng and Xiaode Liu and Liwen Zhang and Xuhui Huang and Zhe Ma},
      year={2023},
      journal={arXiv preprint arXiv:2308.08359},
}
```
