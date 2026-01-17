#!/usr/bin/env python3
"""Materialize inferences using OWL-RL reasoner + SPARQL CONSTRUCT."""

from pathlib import Path
from rdflib import Graph
import owlrl

KG_DIR = Path("knowledge_graph")
OUTPUT_PATH = KG_DIR / "materialized.ttl"

INPUT_FILES = [
    "smartphone.ttl",
    "smartphone-skos.ttl",
    "facts.ttl",
]

SPARQL_CONSTRUCTS = [
    ("HighResolutionCameraPhone", """
        PREFIX sp: <http://example.org/smartphone#>
        CONSTRUCT { ?phone a sp:HighResolutionCameraPhone }
        WHERE {
            ?phone a sp:Smartphone ; sp:mainCameraMP ?mp .
            FILTER(?mp >= 100)
        }
    """),
    ("LargeBatteryPhone", """
        PREFIX sp: <http://example.org/smartphone#>
        CONSTRUCT { ?phone a sp:LargeBatteryPhone }
        WHERE {
            ?phone a sp:Smartphone ; sp:batteryCapacityMah ?mah .
            FILTER(?mah >= 5000)
        }
    """),
    ("InMarketPhone", """
        PREFIX sp: <http://example.org/smartphone#>
        CONSTRUCT { ?phone a sp:InMarketPhone }
        WHERE {
            ?phone a sp:Smartphone .
            ?variant sp:variantOf ?phone ; sp:hasPrice ?price .
        }
    """),
]


def load_graph() -> Graph:
    """Load all input TTL files into a single graph."""
    g = Graph()
    print("Loading knowledge graph...")
    for filename in INPUT_FILES:
        path = KG_DIR / filename
        if path.exists():
            g.parse(path, format="turtle")
            print(f"  {filename}")
    print(f"Loaded {len(g):,} triples")
    return g


def run_owl_inference(g: Graph) -> int:
    """Run OWL-RL reasoning. Returns number of new triples."""
    print("\nRunning OWL-RL inference...")
    before = len(g)
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
    added = len(g) - before
    print(f"  +{added:,} triples")
    return added


def run_sparql_constructs(g: Graph) -> int:
    """Run SPARQL CONSTRUCT queries for numeric restrictions. Returns number of new triples."""
    print("\nRunning SPARQL CONSTRUCT queries...")
    total = 0
    for name, query in SPARQL_CONSTRUCTS:
        result_graph = g.query(query).graph
        count = len(result_graph) if result_graph else 0
        if result_graph:
            g += result_graph
        print(f"  {name}: +{count}")
        total += count
    return total


def main():
    g = load_graph()
    before = len(g)

    run_owl_inference(g)
    run_sparql_constructs(g)

    print(f"\nTotal inferred: {len(g) - before:,} new triples")
    print(f"Final: {len(g):,} triples")

    print(f"\nSaving to {OUTPUT_PATH}...")
    g.serialize(destination=OUTPUT_PATH, format="turtle")
    print("Done.")


if __name__ == "__main__":
    main()
