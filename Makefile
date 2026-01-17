clean_generated_data:
	rm -f data/review_tags.json

build_kg:
	uv run "src/kg_build_pipeline.py"

clean_kg:
	rm -f knowledge_graph/final_knowledge_graph.ttl
	rm -rf knowledge_graph/data