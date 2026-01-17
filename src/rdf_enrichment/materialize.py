from pathlib import Path
from typing import List, Tuple
from rdflib import Graph
import owlrl


def load_graph(files_to_load: List[Path]) -> Graph:
    graph = Graph()
    for path in files_to_load:
        graph.parse(path, format="turtle")
    return graph


def apply_sparql_constructs(graph: Graph, constructs: List[Tuple[str, str]]) -> None:
    for _, query in constructs:
        result = graph.query(query).graph
        if result:
            graph += result


def apply_owl_reasoning(graph: Graph) -> None:
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(graph)


def materialize(
    input_files: List[Path],
    sparql_constructs: List[Tuple[str, str]],
    output_materialized: Path,
    output_inferred: Path
) -> None:
    """Execute materialization pipeline."""
    graph: Graph = load_graph(input_files)
    apply_sparql_constructs(graph, sparql_constructs)
    output_materialized.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=output_materialized, format="turtle")

    # Print number of triples materialized
    with open(output_materialized, "r", encoding="utf-8") as f:
        triple_count = sum(1 for line in f if line.strip() and not line.startswith("@"))
    print(f"Materialized {triple_count} triples in {output_materialized}")

    graph = load_graph(input_files)
    apply_sparql_constructs(graph, sparql_constructs)
    apply_owl_reasoning(graph)
    output_inferred.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=output_inferred, format="turtle")

    # Print number of triples inferred
    with open(output_inferred, "r", encoding="utf-8") as f:
        triple_count = sum(1 for line in f if line.strip() and not line.startswith("@"))
    print(f"Inferred {triple_count} triples in {output_inferred}")