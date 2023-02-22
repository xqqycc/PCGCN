# PCGCN
This is the official Pytorch-version code of PCGCN (Preference Contrastive Graph Convolutional Network for
Collaborative Filtering Recommendation, submitted to SIGIR2023).

## Requirements
torch >= 1.7.0

numba == 0.53.1

numpy == 1.20.3

scipy == 1.6.2

## Usage
The Movielens datasets are provided in the **dataset** folder.

The running parameters can be tuned in the **conf** folder.

You can use the following run command to obtain the result on Movielens dataset:
```
python main.py
```
