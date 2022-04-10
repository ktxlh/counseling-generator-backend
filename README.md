# Counseling Generator Backend

## Setup
### Install Miniconda

Miniconda installer
This [cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) contains all comments I use with conda.

After installing miniconda, run this inside (`counseling-generator-backend/`) to create an env `mi` from `requirements.txt`.
```
conda env create --file requirements.txt --name mi
```

Run this to activate the env every time.
```
conda activate mi
```

After you install anything new via `pip`, replace the `requirements.txt` with a new one via:
```
pip list --format=freeze > requirements.txt
```
