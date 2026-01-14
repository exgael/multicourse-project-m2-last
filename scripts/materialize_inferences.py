from rdflib import Graph

INVERSE_QUERIES = [
    # hasBrand -> manufactures
    ("manufactures", """
    PREFIX sp: <http://example.org/smartphone#>
    CONSTRUCT { ?brand sp:manufactures ?phone }
    WHERE { ?phone sp:hasBrand ?brand }
    """),
    # variantOf -> hasVariant
    ("hasVariant", """
    PREFIX sp: <http://example.org/smartphone#>
    CONSTRUCT { ?phone sp:hasVariant ?variant }
    WHERE { ?variant sp:variantOf ?phone }
    """),
    # reviewOf -> hasReview
    ("hasReview", """
    PREFIX sp: <http://example.org/smartphone#>
    CONSTRUCT { ?phone sp:hasReview ?review }
    WHERE { ?review sp:reviewOf ?phone }
    """),
    # writtenBy -> wroteReview
    ("wroteReview", """
    PREFIX sp: <http://example.org/smartphone#>
    CONSTRUCT { ?user sp:wroteReview ?review }
    WHERE { ?review sp:writtenBy ?user }
    """),
]

CLASS_QUERIES = [
    # FiveGPhone: supports5G = true
    ("FiveGPhone", """
    PREFIX sp: <http://example.org/smartphone#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    CONSTRUCT { ?phone rdf:type sp:FiveGPhone }
    WHERE { ?phone sp:supports5G true }
    """),
    # LargeBatteryPhone: batteryCapacityMah >= 5000
    ("LargeBatteryPhone", """
    PREFIX sp: <http://example.org/smartphone#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    CONSTRUCT { ?phone rdf:type sp:LargeBatteryPhone }
    WHERE { ?phone sp:batteryCapacityMah ?mah FILTER(?mah >= 5000) }
    """),
    # HighResolutionCameraPhone: mainCameraMP >= 100
    ("HighResolutionCameraPhone", """
    PREFIX sp: <http://example.org/smartphone#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    CONSTRUCT { ?phone rdf:type sp:HighResolutionCameraPhone }
    WHERE { ?phone sp:mainCameraMP ?mp FILTER(?mp >= 100) }
    """),
]

def run_queries(g: Graph, queries: list[tuple[str, str]], label: str) -> int:
    print(f"\n{label}...")
    total = 0
    for name, query in queries:
        result = g.query(query)
        count = 0
        for triple in result:
            g.add(triple) # type: ignore
            count += 1
        print(f"  {name}: +{count:,}")
        total += count
    return total

def main():
    kg_path = "knowledge_graph/kg.ttl"

    print("Loading KG...")
    g = Graph()
    g.parse(kg_path, format="turtle")
    before = len(g)
    print(f"Loaded {before:,} triples")

    run_queries(g, INVERSE_QUERIES, "Materializing inverse properties")
    run_queries(g, CLASS_QUERIES, "Materializing inferred classes")

    after = len(g)
    print(f"\nTotal inferred: {after - before:,} new triples")
    print(f"Final: {after:,} triples")

    print(f"\nSaving to {kg_path}...")
    g.serialize(destination=kg_path, format="turtle")
    print("Done.")

if __name__ == "__main__":
    main()
