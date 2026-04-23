# Vector-Search-FAISS

This repository is for FAISS-based vector search experiments.

## 1) Clone and enter the project

```powershell
git clone <YOUR_REPO_URL>
cd Vector-Search-FAISS
```

## 2) Create a Python environment and install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3) Create the required folders

The raw data is not committed to Git (`data/raw/` is ignored in `.gitignore`), so each user must create and initialize local data folders.

```powershell
mkdir data, data\noisy_queries, data\processed, data\raw, data\raw\sift, data\raw\gist, data\raw\glove, results, results\exports, results\plots, results\tables, src, notebooks
```

## 4) Initialize raw data (download datasets)

Run the following commands exactly:

```powershell
curl.exe -o data/raw/sift/sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
curl.exe -o data/raw/gist/gist.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
curl.exe -L -o data/raw/glove/glove.twitter.27B.zip http://nlp.stanford.edu/data/glove.twitter.27B.zip
```

## 5) Extract downloaded files

```powershell
tar -xzf data/raw/sift/sift.tar.gz -C data/raw/sift
tar -xzf data/raw/gist/gist.tar.gz -C data/raw/gist
Expand-Archive -Path data/raw/glove/glove.twitter.27B.zip -DestinationPath data/raw/glove -Force
```

## 6) Start working

```powershell
jupyter notebook notebooks/experiments.ipynb
```

## Notes

- `data/raw/` is intentionally ignored by Git to avoid committing large datasets.
- Keep generated artifacts in `results/exports`, `results/plots`, and `results/tables`.
