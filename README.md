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

If you need to rebuild knowledge graph, see `data/README.md`

You can run the following:
```sh
make data
```

## Clean up

```sh
make clean
```