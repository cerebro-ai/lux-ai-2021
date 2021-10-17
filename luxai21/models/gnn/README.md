Based on torch_geometric

## Installation
### Using conda
```shell
conda install pyg -c pyg -c conda-forge
```
### Or with pip

Ensure that at least PyTorch 1.4.0 is installed:
```shell
python -c "import torch; print(torch.__version__)"
> 1.9.0
```

Find the CUDA version PyTorch was installed with:

```shell
python -c "import torch; print(torch.version.cuda)"
> 11.1
```

Install the relevant packages:

```shell
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```
where ${CUDA} and ${TORCH} should be replaced by the specific CUDA version (cpu, cu92, cu101, cu102, cu110, cu111)
and PyTorch version (1.4.0, 1.5.0, 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1), respectively. For example, for PyTorch 1.9.0/1.9.1 and CUDA 11.1, type:
