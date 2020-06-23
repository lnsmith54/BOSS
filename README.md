# BOSS
This repository provides the code for replicating the experiments in the paper "Building One-Shot Semi-supervised (BOSS) Learning up to Fully Supervised Performance"

This repository contains two sets of codes: one for the TensorFlow version that is contained in the folder TF-FixMatch and the other for the PyTorch version that is contained in PT-FixMatch.  Both folders contain README files that describe how to install dependencies, setup the datasets, and run the codes to replicate our experiments.

Additional information is available in the Appendix of our paper.

## License

This code is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
1. That the code comes “AS IS”, without express or implied warranty. Although every effort has been made to ensure accuracy, we (NRL) do not accept any responsibility for errors or omissions.
2. That you include a reference to the BOSS paper in any work that makes use of this code.
3. You may not use the codebase or any derivative work for commercial purposes such as, for example, licensing or selling the software, or using the software with a purpose to procure a commercial gain.
4. That all rights not expressly granted to you are reserved by us (NRL).

## Citation

When using the dataset or code, please cite our [paper](https://arxiv.org/abs/2006.09363): 
```
@article{smith2020building,
  title={Building One-Shot Semi-supervised (BOSS) Learning up to Fully Supervised Performance},
  author={Smith, Leslie N and Conovaloff, Adam},
  journal={arXiv preprint arXiv:2006.09363},
  year={2020}
}
```



## Acknowledgements

The codebase is heavily based off [FixMatch](https://github.com/google-research/fixmatch) and [FixMatch-pytorch](https://github.com/CoinCheung/fixmatch-pytorch). Both are great repositories - have a look!
