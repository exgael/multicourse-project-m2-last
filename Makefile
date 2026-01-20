clean:
	rm -rf output

run:
	uv run "src/kg_build_pipeline.py"