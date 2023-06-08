
# Robust Bayesian Tensor Factorization with Zero-Inflated Poisson Model and Consensus Aggregation

This paper proposes a new method for **Bayesian Tensor Factorization with Zero-Inflated Poisson Model and Consensus Aggregation**. Included here are the code for implementing bayesian tensor factorization with various models and consensus aggregation, and code for running the experiments on simulated and real data.

The main implementation is found in BayesianCP class in `./scBTF/scBTF/bayesian_parafac.py` The class implements a generic probabilistic tensor factorization with variational inference using pyro. Models which specify the generative model including the distribution of the data and the variational distribution (_guide_ in pyro terminology) are selected from the list of options provided in the class.

The rest of the scBTF package contains code specifically designed to run bayesian tensor factorization on single cell RNA-Seq data. For the simulation experiments we use only the BayesianCP class. For the real scRNA-Seq data analysis, we use the whole package. 



## Requirements

Requirements given in `./scBTF/requirements.txt`. Additionally, we install cNMF from its github source to get the latest version.

Running the following commands from a `continuumio/miniconda3:22.11.1` docker image would be the easiest way to get the desired environment:

```commandline
# Install scBTF
conda create -n scBTF python=3.9 -y
conda activate scBTF
pip install -U adpbulk gseapy rich statannotations scanpy fastcluster torch
cd scBTF
pip install .

# Install cNMF
git clone https://github.com/dylkot/cNMF.git
cd cNMF
pip install .
```



## Running

- `experiment_1a.py` and `experiment_1bc.py` contain the script to run synthetic tensor experiments described in section 4.1
- `experiment_2.py` contains the script to run the synthetic single-cell RNA-Seq data analysis given in section 4.2
- `ifnb_tensor_factorization.ipynb` - jupyter notebook for downloading preprocessing and running C-ZIPTF on IFN-B dataset