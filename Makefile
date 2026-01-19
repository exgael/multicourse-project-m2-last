.PHONY: clean data

clean:
	rm -rf data/preprocessed
	rm -rf data/rdf/subgraphs
	rm -f data/rdf/knowledge_graph_full.ttl
	rm -rf data/users

data:
	uv run data/preprocess.py
	uv run data/generate_users.py
	uv run data/data_to_rdf.py
