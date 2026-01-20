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

## Structure

- Data: `data/raw_prety`, `data/preprocessed` and `data/users`
- Ontology+KG: `data/rdf`
- Recommandation: `src/recommandation`
- GraphRag: `src/graphrag`
- sparql competency: `src/sparql`
- UI: `streamlite/`


## Build Knowledge Graph

If you need to rebuild knowledge graph, see `data/README.md`. Note that the RML Mapper jar seen during the course is required. 

You can run the following:
```sh
make data
```


## Web App

The UI is done via streamlite in Python.

Run the following in cmd line to launch UI.

```sh
make ui
```

## Clean up

Warning: doing this will require a FULL data processing + KG build. This can take a lot of time due to linking and alignment steps.
```sh
make clean
```

