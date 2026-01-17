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
## Build Knowledge Graph

1. Generate facts
2. Combine both ontologie and facts
3. Materialize inverse relation + classes that should be inferred.

```sh
make build_kg
```

or 

```sh
uv run src/kg_build_pipeline.py
```