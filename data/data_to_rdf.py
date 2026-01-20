import subprocess
from pathlib import Path
from rdflib import Graph

def generate_facts(mapping: Path, output: Path, rml_mapper_jar: Path) -> None:
    """Generate RDF facts from JSON data using RMLMapper."""

    if not mapping.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping}")

    if not rml_mapper_jar.exists():
        raise FileNotFoundError(
            f"RMLMapper not found: {rml_mapper_jar}\n"
            "Download from: https://github.com/RMLio/rmlmapper-java/releases"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["java", "-jar", str(rml_mapper_jar), "-m", str(mapping), "-o", str(output), "-s", "turtle"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"RMLMapper failed:\n{result.stderr}")

    g = Graph()
    g.parse(output, format="turtle")
    print(f"Generated {len(g)} triples in {output}")


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent.parent / "data"
    RML_MAPPER = DATA_DIR / "rml_mappings.ttl"
    RDF_OUTPUT = DATA_DIR / "rdf" / "subgraphs" / "data.ttl"
    jars = list(DATA_DIR.glob("*.jar"))
    if not jars:
        raise FileNotFoundError("No RMLMapper .jar found in DATA_DIR")
    RML_MAPPER_JAR = jars[0]

    generate_facts(
        mapping=RML_MAPPER,
        output=RDF_OUTPUT,
        rml_mapper_jar=RML_MAPPER_JAR,
    )