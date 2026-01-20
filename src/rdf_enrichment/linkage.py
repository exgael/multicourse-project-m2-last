from pathlib import Path
import requests
from rdflib import Graph, Namespace
from rdflib.namespace import RDF

SP = Namespace("http://example.org/smartphone#")
SPOTLIGHT_URL = "https://api.dbpedia-spotlight.org/en/annotate"


def spotlight(text: str) -> str | None:
    """Get DBpedia URI for text via Spotlight API."""
    try:
        r = requests.post(
            SPOTLIGHT_URL,
            data={"text": f"{text} company", "confidence": 0.8},
            headers={"Accept": "application/json"},
            timeout=10
        )
        for res in r.json().get("Resources", []):
            if text.lower() in res.get("@surfaceForm", "").lower():
                return res["@URI"]
    except Exception:
        pass
    return None


def perform_linkage(input_file: Path, output_file: Path) -> None:
    """Link smartphone brands to DBpedia using Spotlight API."""
    g = Graph()
    g.parse(input_file, format="turtle")

    brands: list[str] = list(g.subjects(RDF.type, SP.Brand))  # type: ignore
    links: list[tuple[str, str]] = []

    for uri in brands:
        name = str(uri).split("/")[-1].replace("_", " ")
        dbpedia = spotlight(name)
        if dbpedia:
            links.append((uri, dbpedia))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n\n")
        f.write("# Brand linking to DBpedia (automatic via Spotlight)\n\n")
        for uri, dbpedia in links:
            f.write(f"<{uri}> owl:sameAs <{dbpedia}> .\n")

    # Print number of links created
    print(f"Created {len(links)} links to DBpedia in {output_file}")