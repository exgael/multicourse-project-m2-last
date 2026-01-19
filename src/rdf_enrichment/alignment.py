from pathlib import Path
import json
import requests
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL

SP = Namespace("http://example.org/smartphone#")

LOV_API_URL = "https://lov.linkeddata.es/dataset/lov/api/v2/term/search"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

# Trusted vocabulary prefixes for LOV API
TRUSTED_VOCABS = [
    "schema.org",
    "dbpedia.org",
    "wikidata.org",
    "purl.org/goodrelations",
    "xmlns.com/foaf",
    "purl.org/dc",
    "w3.org/2004/02/skos",
]

# =============================================================================
# MANUAL ALIGNMENTS (for domain-specific terms not covered by LOV)
# =============================================================================

MANUAL_ALIGNMENTS = """
# Classes
<http://example.org/smartphone#BasePhone> rdfs:subClassOf <http://schema.org/Product> .
<http://example.org/smartphone#BasePhone> skos:exactMatch <http://www.wikidata.org/entity/Q22645> .
<http://example.org/smartphone#User> rdfs:subClassOf <http://schema.org/Person> .
<http://example.org/smartphone#User> rdfs:subClassOf <http://xmlns.com/foaf/0.1/Person> .
<http://example.org/smartphone#TagSentiment> skos:closeMatch <http://schema.org/Rating> .

# Properties
<http://example.org/smartphone#manufactures> skos:closeMatch <http://schema.org/manufacturer> .
<http://example.org/smartphone#manufactures> skos:closeMatch <http://dbpedia.org/ontology/manufacturer> .
<http://example.org/smartphone#likes> skos:closeMatch <http://xmlns.com/foaf/0.1/interest> .
<http://example.org/smartphone#batteryCapacityMah> skos:closeMatch <http://dbpedia.org/ontology/batteryCapacity> .
<http://example.org/smartphone#ramGB> skos:closeMatch <http://dbpedia.org/ontology/ram> .
<http://example.org/smartphone#storageGB> skos:closeMatch <http://dbpedia.org/ontology/storage> .
<http://example.org/smartphone#supports5G> skos:closeMatch <http://dbpedia.org/ontology/frequencyBand> .
<http://example.org/smartphone#supportsNFC> skos:closeMatch <http://dbpedia.org/ontology/feature> .
"""

# =============================================================================


def query_lov(term: str, term_type: str = "class", limit: int = 20) -> list[dict]:
    """Query LOV API for candidates."""
    candidates = []
    try:
        params = {"q": term, "type": term_type, "page_size": limit}
        r = requests.get(LOV_API_URL, params=params, timeout=10)
        results = r.json().get("results", [])

        for res in results:
            uri = res.get("uri", [None])[0]
            if not uri:
                continue
            if not any(vocab in uri for vocab in TRUSTED_VOCABS):
                continue
            candidates.append({
                "uri": uri,
                "label": res.get("prefixedName", [""])[0],
                "score": res.get("score", 0),
            })

        # Also try with spaces (camelCase -> spaced)
        spaced = ''.join(' ' + c.lower() if c.isupper() else c for c in term).strip()
        if spaced != term.lower():
            params = {"q": spaced, "type": term_type, "page_size": limit}
            r = requests.get(LOV_API_URL, params=params, timeout=10)
            for res in r.json().get("results", []):
                uri = res.get("uri", [None])[0]
                if uri and any(vocab in uri for vocab in TRUSTED_VOCABS):
                    if uri not in [c["uri"] for c in candidates]:
                        candidates.append({
                            "uri": uri,
                            "label": res.get("prefixedName", [""])[0],
                            "score": res.get("score", 0),
                        })
    except Exception as e:
        print(f"  LOV error: {e}")
    return candidates[:limit]


def ask_llm(term_name: str, term_type: str, label: str, comment: str, candidates: list[dict]) -> list[dict]:
    """Ask Ollama to choose best alignments."""
    if not candidates:
        return []

    candidates_text = "\n".join([
        f"  {i+1}. {c['uri']} (label: {c['label']}, score: {c['score']:.2f})"
        for i, c in enumerate(candidates)
    ])

    prompt = f"""You are an ontology alignment expert. Given a term from a smartphone ontology, choose the best matching terms from external vocabularies.

TERM: {term_name} ({term_type})
Label: {label or 'N/A'}
Description: {comment or 'N/A'}
Domain: Smartphone/mobile phone specifications and e-commerce

CANDIDATES:
{candidates_text}

Select 0-3 best matches. For each, specify relation type:
- "equivalent": exact same meaning
- "subclass"/"subproperty": our term is more specific
- "close": similar but not exact
- "exact": same concept, different vocabulary

Respond ONLY with JSON array. Example:
[{{"uri": "http://schema.org/Brand", "relation": "equivalent"}}]

If no good match: []
"""

    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        text = r.json().get("response", "[]")
        start, end = text.find("["), text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as e:
        print(f"  LLM unavailable, using top LOV match")
        # Fallback: use top candidate as "close" match
        if candidates:
            return [{"uri": candidates[0]["uri"], "relation": "close"}]
    return []


def extract_classes_and_properties(g: Graph) -> tuple[list[URIRef], list[URIRef]]:
    """Extract classes and properties from ontology."""
    classes = [s for s in g.subjects(RDF.type, OWL.Class)
               if isinstance(s, URIRef) and str(s).startswith(str(SP))]
    props = [s for s in g.subjects(RDF.type, OWL.ObjectProperty)
             if isinstance(s, URIRef) and str(s).startswith(str(SP))]
    props += [s for s in g.subjects(RDF.type, OWL.DatatypeProperty)
              if isinstance(s, URIRef) and str(s).startswith(str(SP))]
    return classes, props


def get_local_name(uri: URIRef) -> str:
    s = str(uri)
    return s.split("#")[-1] if "#" in s else s.split("/")[-1]


def get_label_comment(g: Graph, uri: URIRef) -> tuple[str, str]:
    label = next((str(o) for o in g.objects(uri, RDFS.label)), "")
    comment = next((str(o) for o in g.objects(uri, RDFS.comment)), "")
    return label, comment


def make_triple(uri: URIRef, ext_uri: str, relation: str, is_class: bool) -> str:
    if is_class:
        pred = {"equivalent": "owl:equivalentClass", "subclass": "rdfs:subClassOf",
                "exact": "skos:exactMatch"}.get(relation, "skos:closeMatch")
    else:
        pred = {"equivalent": "owl:equivalentProperty", "subproperty": "rdfs:subPropertyOf",
                "exact": "skos:exactMatch"}.get(relation, "skos:closeMatch")
    return f"<{uri}> {pred} <{ext_uri}> ."


def perform_alignment(input_file: Path, output_file: Path) -> None:
    """Align ontology with external vocabularies (automatic + manual)."""
    g = Graph()
    g.parse(input_file, format="turtle")

    classes, props = extract_classes_and_properties(g)
    auto_alignments: list[str] = []

    print(f"Aligning {len(classes)} classes and {len(props)} properties...\n")

    # Automatic alignment for classes
    for uri in classes:
        name = get_local_name(uri)
        label, comment = get_label_comment(g, uri)
        print(f"[CLASS] {name}")
        candidates = query_lov(name, "class")
        print(f"  {len(candidates)} LOV candidates")
        if candidates:
            choices = ask_llm(name, "class", label, comment, candidates)
            print(f"  LLM selected {len(choices)}")
            for c in choices:
                if c.get("uri"):
                    auto_alignments.append(make_triple(uri, c["uri"], c.get("relation", "close"), True))
                    print(f"    -> {c.get('relation', 'close')}: {c['uri']}")

    # Automatic alignment for properties
    for uri in props:
        name = get_local_name(uri)
        label, comment = get_label_comment(g, uri)
        print(f"[PROP] {name}")
        candidates = query_lov(name, "property")
        print(f"  {len(candidates)} LOV candidates")
        if candidates:
            choices = ask_llm(name, "property", label, comment, candidates)
            print(f"  LLM selected {len(choices)}")
            for c in choices:
                if c.get("uri"):
                    auto_alignments.append(make_triple(uri, c["uri"], c.get("relation", "close"), False))
                    print(f"    -> {c.get('relation', 'close')}: {c['uri']}")

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n")
        f.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")
        f.write("@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n\n")

        f.write("\n\n# AUTOMATIC ALIGNMENTS (via LOV API + LLM)\n\n")
        for t in auto_alignments:
            f.write(f"{t}\n")
        
        f.write("\n\n# MANUAL ALIGNMENTS (domain-specific terms not covered by LOV)\n")
        f.write(MANUAL_ALIGNMENTS)

    manual_count = len([l for l in MANUAL_ALIGNMENTS.strip().split("\n") if l and not l.startswith("#")])
    print(f"\nCreated {len(auto_alignments)} automatic + {manual_count} manual alignments in {output_file}")
