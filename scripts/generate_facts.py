"""
Generate RDF tripples via rml mapper.

Usage:
    uv run generate_facts.py <mapping.ttl> [--output <output.ttl>]

Examples:
    uv run generate_facts.py knowledge_graph/smartphone-rml.ttl
    uv run generate_facts.py knowledge_graph/smartphone-rml.ttl --output knowledge_graph/facts.ttl
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
RMLMAPPER_JAR = ROOT_DIR / "rmlmapper.jar"


def main():
    parser = argparse.ArgumentParser(description="Generate RDF using RMLMapper")
    parser.add_argument("mapping", help="RML mapping file (.ttl)")
    parser.add_argument("-o", "--output", default="knowledge_graph/facts.ttl", help="Output file")
    args = parser.parse_args()

    mapping = Path(args.mapping)
    output = Path(args.output)

    if not mapping.exists():
        print(f"Error: {mapping} not found")
        sys.exit(1)

    if not RMLMAPPER_JAR.exists():
        print(f"Error: {RMLMAPPER_JAR} not found")
        print("Download from: https://github.com/RMLio/rmlmapper-java/releases")
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["java", "-jar", str(RMLMAPPER_JAR), "-m", str(mapping), "-o", str(output), "-s", "turtle"]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"Output: {output}")


if __name__ == "__main__":
    main()
