#!/bin/bash
# Build complete Knowledge Graph:
# 1. Generate facts from RML
# 2. Merge T-Box (ontology) + A-Box (facts)
# 3. Materialize inferences

set -e
cd "$(dirname "$0")/.."

OUTPUT="knowledge_graph/facts.ttl"

echo "[1] Generating facts (RMLMapper)..."
uv run scripts/generate_facts.py rml_mappings/smartphone-rml.ttl -o knowledge_graph/facts.ttl

echo ""
echo "[3] Materializing missing triples..."
uv run scripts/materialize.py

echo ""
echo "[4] Linking brands to DBpedia..."
uv run scripts/link_brands.py
cat knowledge_graph/brand-links.ttl >> "$OUTPUT"
rm -f knowledge_graph/brand-links.ttl

echo ""
echo "BUILD COMPLETE"
wc -l "$OUTPUT"
