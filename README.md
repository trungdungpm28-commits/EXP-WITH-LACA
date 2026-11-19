# LACE experiment reproduction

Source code, Azure Functions 2019 input data used in the experiment, trained model weights, replay arrays, measured results and figures are in [`lace-reproduction/`](lace-reproduction/README.md).

## Download all data

Install [Git LFS](https://git-lfs.com/), then run:

```bash
git lfs install
git clone https://github.com/trungdungpm28-commits/EXP-WITH-LACA.git
cd EXP-WITH-LACA
git lfs pull
```

Large arrays, weights and source CSV files are stored with Git LFS. The Python virtual environment and runtime plotting caches are excluded. Personal machine paths in text evidence are replaced with `<PROJECT_ROOT>` and `<USER_HOME>`. Experiment measurements remain unchanged.

The source Azure trace files in this snapshot cover days 1–4 used by this experiment. The full Microsoft dataset and attribution are described in the project documentation.
