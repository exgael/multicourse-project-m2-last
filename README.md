# multicourse-project-m2-last

Graphe de connaissances sur les smartphones pour recommandation selon les besoins des utilisateurs.

## Requirements

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
# Install dependencies
uv sync
```

## Usage

```python
from load_data import load_datasets

df_smartphones, df_amazon_reviews = load_datasets()
```

Downloads on first run, uses cache afterward.