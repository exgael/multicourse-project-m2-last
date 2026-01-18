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

### Requires RML MAPPER

Download from: https://github.com/RMLio/rmlmapper-java/releases
And put jar at the root of the folder. Called `rmlmapper.jar`

### Build

1. Convert prices to eur
2. Generate users
3. Generate facts
4. Materialize inverse relation + classes that should be inferred.
5. Linkage
6. Combine everything into Kg

```sh
make run
```

## Clean up

```sh
make clean
```