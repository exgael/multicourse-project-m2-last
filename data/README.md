# Data

This directory is about processing and generating rdf data.

## Requirements

1. The `uv` package manager.

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. RML Mapper Executable

Download from: https://github.com/RMLio/rmlmapper-java/releases.

Put the `jar` file the **`data`** folder.

## How to generate the if necessary

### Process the raw data. 

This will create simpler structures to work with and do some aggregations and conversions.

```sh
uv run data/preprocess.py
```
- output: `preprocessed/*`


### Generate the user data.

```sh
uv run generate_users.py
```
- output: `users/*`

### Create the base rdf

```sh
uv run data_to_rdf.py
```
- output: `rdf/subgraphs/data.ttl`

### Full RDF Graph

You can now run the `full_knowledge_graph.ipynb` notebook
- output: `rdf/subgraphs/*`
- output: `rdf/knwoledge_graph_full`
