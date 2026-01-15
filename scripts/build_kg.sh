#!/bin/bash
# Build complete Knowledge Graph:
# 1. Generate facts from RML
# 2. Merge T-Box (ontology) + A-Box (facts)
# 3. Materialize inferences

set -e
cd "$(dirname "$0")/.."

OUTPUT="knowledge_graph/kg.ttl"

echo "[1] Generating facts (RMLMapper)..."
uv run scripts/generate_facts.py rml_mappings/smartphone-rml.ttl -o knowledge_graph/facts.ttl

echo ""
echo "[2] Merging T-Box + A-Box..."
# T-Box: Ontology + SKOS + SHACL
cat knowledge_graph/smartphone.ttl > "$OUTPUT"
echo "" >> "$OUTPUT"
cat knowledge_graph/smartphone-skos.ttl >> "$OUTPUT"
echo "" >> "$OUTPUT"
cat knowledge_graph/smartphone-shacl.ttl >> "$OUTPUT"
echo "" >> "$OUTPUT"

# A-Box: Instance data (facts)
cat knowledge_graph/facts.ttl >> "$OUTPUT"
rm -f knowledge_graph/facts.ttl

echo "Merged: $OUTPUT"
wc -l "$OUTPUT"

echo ""
echo "[3] Materializing inferences..."
uv run scripts/materialize_inferences.py

echo ""
echo "[4] Linking brands to DBpedia..."
uv run scripts/link_brands.py
cat knowledge_graph/brand-links.ttl >> "$OUTPUT"
rm -f knowledge_graph/brand-links.ttl

echo ""
echo "BUILD COMPLETE"
wc -l "$OUTPUT"
