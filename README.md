# Counseling Generator Backend

## Setup
### Install Miniconda

Miniconda installer
This [cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) contains all comments I use with conda.

After installing miniconda, create an env `mi`.
```
conda env create --name mi python=3.10
```

Run this to activate the env every time you work on this project.
```
conda activate mi
```

For the first time you use the env, run this inside (`counseling-generator-backend/`) to install the dependencies.
```
pip install -r requirements.txt
```

After you installed anything new via `pip`, replace the `requirements.txt` with a new one via:
```
pip list --format=freeze > requirements.txt
```
