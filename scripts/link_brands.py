import requests
from rdflib import Graph, Namespace
from rdflib.namespace import RDF

SP = Namespace("http://example.org/smartphone#")
SPOTLIGHT_URL = "https://api.dbpedia-spotlight.org/en/annotate"


def spotlight(text: str) -> str | None:
    """Get DBpedia URI for text via Spotlight API."""
    try:
        r = requests.post(SPOTLIGHT_URL, data={"text": f"{text} company", "confidence": 0.8},
                          headers={"Accept": "application/json"}, timeout=10)
        for res in r.json().get("Resources", []):
            if text.lower() in res.get("@surfaceForm", "").lower():
                return res["@URI"]
    except Exception:
        pass
    return None


def main():
    g = Graph()
    g.parse("knowledge_graph/kg.ttl", format="turtle")

    brands: list[str] = list(g.subjects(RDF.type, SP.Brand)) # type: ignore
    print(f"Found {len(brands)} brands\n")

    links: list[tuple[str, str]] = []
    for uri in brands:
        name = str(uri).split("/")[-1].replace("_", " ")
        dbpedia = spotlight(name)
        if dbpedia:
            links.append((uri, dbpedia))
            print(f"  {name} -> {dbpedia.split('/')[-1]}")
        else:
            print(f"  {name} -> not found")

    with open("knowledge_graph/brand-links.ttl", "w") as f:
        f.write("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n\n")
        f.write("# Brand linking to DBpedia (automatic via Spotlight)\n\n")
        for uri, dbpedia in links:
            f.write(f"<{uri}> owl:sameAs <{dbpedia}> .\n")

    print(f"\nSaved {len(links)} links to knowledge_graph/brand-links.ttl")


if __name__ == "__main__":
    main()
