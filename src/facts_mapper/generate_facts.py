import subprocess
from pathlib import Path

def ensure_path(path: Path | str) -> Path:
    """Ensure the input is a Path object."""
    if isinstance(path, str):
        return Path(path)
    return path


def generate_facts(mapping: Path | str, output: Path | str, rml_mapper_jar: Path | str) -> None:
    """Generate RDF facts from JSON data using RMLMapper."""
    mapping = ensure_path(mapping)
    output = ensure_path(output)
    rml_mapper_jar = ensure_path(rml_mapper_jar)

    if not mapping.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping}")

    if not rml_mapper_jar.exists():
        raise FileNotFoundError(
            f"RMLMapper not found: {rml_mapper_jar}\n"
            "Download from: https://github.com/RMLio/rmlmapper-java/releases"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = ["java", "-jar", str(rml_mapper_jar), "-m", str(mapping), "-o", str(output), "-s", "turtle"]
    result: subprocess.CompletedProcess[str] = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"RMLMapper failed: {result.stderr}")
    
    # Print number of triples generated
    with open(output, "r", encoding="utf-8") as f:
        triple_count = sum(1 for line in f if line.strip() and not line.startswith("@"))
    print(f"Generated {triple_count} triples in {output}")

