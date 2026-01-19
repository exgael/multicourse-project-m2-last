"""
GraphRAG Approach 1: Natural Language to SPARQL
Transforms natural language questions into SPARQL queries using LLM (Ollama).
"""

import requests
from pathlib import Path
from typing import Any
from pydantic import BaseModel
from rdflib import Graph

ROOT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
FINAL_KG_TTL = ROOT_DIR / "data" / "rdf" / "knowledge_graph_full.ttl"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

# Schema context for the LLM
ONTOLOGY_CONTEXT = """
# Smartphone KG - SPARQL Guide

## Prefixes (ALWAYS include)
PREFIX sp: <http://example.org/smartphone#>
PREFIX spv: <http://example.org/smartphone/vocab/>

## Classes
- sp:Smartphone - A phone (includes storage/RAM variant info)
- sp:Brand - Manufacturer (Apple, Samsung, etc.)
- sp:Store - Retail store (Amazon.de, Amazon.uk, etc.)
- sp:PriceOffering - A price for a phone at a specific store
- sp:User - User with preferences

## Object Properties
- sp:hasBrand (Phone -> Brand)
- sp:forPhone (PriceOffering -> Phone) - Links price to phone
- sp:offeredBy (PriceOffering -> Store) - Links price to store
- sp:interestedIn (User -> spv:Concept)
- sp:likes (User -> Phone)

## Datatype Properties
- sp:phoneName (string) - On Phone, ALWAYS select this
- sp:brandName (string) - On Brand
- sp:storeName (string) - On Store
- sp:priceValue (decimal) - On PriceOffering, price in EUR
- sp:releaseYear (integer) - On Phone
- sp:batteryCapacityMah (integer) - On Phone
- sp:mainCameraMP (integer) - On Phone
- sp:selfieCameraMP (integer) - On Phone
- sp:refreshRateHz (integer) - On Phone
- sp:supports5G (boolean) - On Phone, true/false, NO quotes
- sp:supportsNFC (boolean) - On Phone, true/false, NO quotes
- sp:displayType (string) - On Phone, "AMOLED", "LCD", etc.
- sp:storageGB (integer) - On Phone, Storage in GB
- sp:ramGB (integer) - On Phone, RAM in GB
- sp:userId (string) - On User

## Use Cases (spv: namespace)
- spv:Photography, spv:Gaming, spv:Business, spv:EverydayUse, spv:Vlogging, spv:Minimalist

## Price Segments (spv: namespace)
- spv:Flagship (>900 EUR), spv:MidRange (400-900), spv:Budget (<400), spv:AfterMarket (no price)

## Example Queries

# Samsung phones with battery > 5000mAh
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName ?battery WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName ;
         sp:hasBrand ?brand ;
         sp:batteryCapacityMah ?battery .
  ?brand sp:brandName "Samsung" .
  FILTER(?battery > 5000)
}

# Gaming phones (high refresh + big battery)
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName ?refreshRate ?battery WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName ;
         sp:refreshRateHz ?refreshRate ;
         sp:batteryCapacityMah ?battery .
  FILTER(?refreshRate >= 120 && ?battery >= 5000)
}

# 5G phones (boolean without quotes)
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName ;
         sp:supports5G true .
}

# AMOLED display (use CONTAINS)
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName ;
         sp:displayType ?display .
  FILTER(CONTAINS(LCASE(?display), "amoled"))
}

# Top 5 by camera (ORDER BY after WHERE brace)
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName ?camera WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName ;
         sp:mainCameraMP ?camera .
}
ORDER BY DESC(?camera)
LIMIT 5

# Phones liked by user
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName WHERE {
  ?user a sp:User ;
        sp:userId "pro_gamer_0000" ;
        sp:likes ?phone .
  ?phone sp:phoneName ?phoneName .
}

# Users interested in gaming
PREFIX sp: <http://example.org/smartphone#>
PREFIX spv: <http://example.org/smartphone/vocab/>
SELECT DISTINCT ?userId WHERE {
  ?user a sp:User ;
        sp:userId ?userId ;
        sp:interestedIn spv:Gaming .
}

# Flagship phones (price > 900 EUR) - uses PriceOffering
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName ?price WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName .
  ?offering sp:forPhone ?phone ;
            sp:priceValue ?price .
  FILTER(?price > 900)
}
ORDER BY DESC(?price)
LIMIT 10

# Phones with 256GB storage under 500 EUR
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName ?storage ?price WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName ;
         sp:storageGB ?storage .
  ?offering sp:forPhone ?phone ;
            sp:priceValue ?price .
  FILTER(?storage >= 256 && ?price < 500)
}

# Find cheapest store for a phone
PREFIX sp: <http://example.org/smartphone#>
SELECT ?storeName ?price WHERE {
  ?phone sp:phoneName "Apple iPhone 14" .
  ?offering sp:forPhone ?phone ;
            sp:offeredBy ?store ;
            sp:priceValue ?price .
  ?store sp:storeName ?storeName .
}
ORDER BY ?price
LIMIT 1

# Compare prices across stores
PREFIX sp: <http://example.org/smartphone#>
SELECT ?phoneName ?storeName ?price WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName .
  ?offering sp:forPhone ?phone ;
            sp:offeredBy ?store ;
            sp:priceValue ?price .
  ?store sp:storeName ?storeName .
}
ORDER BY ?phoneName ?price
"""

SYSTEM_PROMPT = """You are a SPARQL query generator for a smartphone knowledge graph.

Given a natural language question, generate a valid SPARQL query that answers it.

CRITICAL RULES:
1. ALWAYS start with PREFIX declarations:
   PREFIX sp: <http://example.org/smartphone#>
   PREFIX spv: <http://example.org/smartphone/vocab/>
2. ALWAYS include sp:phoneName when selecting phone information
3. Use SELECT DISTINCT to avoid duplicates
4. Booleans: use true/false WITHOUT quotes (sp:supports5G true)
5. Integers: use numbers WITHOUT quotes (sp:releaseYear 2023)
6. Text matching: use FILTER(CONTAINS(LCASE(?var), "text"))
7. ORDER BY and LIMIT must come AFTER the closing brace of WHERE
8. All properties (price, storage, RAM) are directly on Phone
9. Return ONLY the SPARQL query, no explanations

Schema:
{schema}
"""

USER_PROMPT = """Question: {question}

Generate a SPARQL query to answer this question. Return ONLY the query."""


class QueryResult(BaseModel):
    """Result of a GraphRAG query."""
    question: str
    sparql_query: str
    results: list[dict[str, Any]]
    success: bool
    error: str | None = None


class NLToSPARQL:
    """Natural Language to SPARQL converter using Ollama LLM."""

    def __init__(
        self,
        kg_path: Path = FINAL_KG_TTL,
        ollama_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL
    ):
        self.kg_path = kg_path
        self.ollama_url = ollama_url
        self.model = model
        self.graph: Graph | None = None

        # Check Ollama is running
        try:
            r = requests.get(f"{ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"Ollama connected. Available models: {models}")
            if not any(model in m for m in models):
                print(f"Warning: Model '{model}' may not be available. Available: {models}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama at {ollama_url}. Is it running? Error: {e}")

    def load_graph(self) -> None:
        """Load the knowledge graph."""
        if self.graph is None:
            print(f"Loading knowledge graph from {self.kg_path}...")
            self.graph = Graph()
            self.graph.parse(self.kg_path, format="turtle")
            print(f"Loaded {len(self.graph)} triples")

    def generate_sparql(self, question: str) -> str:
        """Generate SPARQL query from natural language question using Ollama."""
        prompt = f"""{SYSTEM_PROMPT.format(schema=ONTOLOGY_CONTEXT)}

{USER_PROMPT.format(question=question)}"""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1000
                }
            },
            timeout=120
        )
        response.raise_for_status()

        sparql = response.json()["response"].strip()

        # Clean up the response (remove markdown code blocks if present)
        if "```" in sparql:
            lines = sparql.split("\n")
            in_code_block = False
            code_lines: list[str] = []
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    code_lines.append(line)
            sparql = "\n".join(code_lines) if code_lines else sparql

        return sparql

    def execute_sparql(self, sparql: str) -> list[dict[str, Any]]:
        """Execute SPARQL query on the knowledge graph."""
        self.load_graph()

        if self.graph is None:
            return []

        results: list[dict[str, Any]] = []
        query_results = self.graph.query(sparql)

        for row in query_results:
            result_dict: dict[str, Any] = {}
            if query_results.vars:
                for var in query_results.vars:
                    value = getattr(row, str(var), None)
                    result_dict[str(var)] = str(value) if value else None
            results.append(result_dict)

        return results

    def query(self, question: str) -> QueryResult:
        """Process a natural language question."""
        sparql = ""
        try:
            sparql = self.generate_sparql(question)
            print(f"\nGenerated SPARQL:\n{sparql}\n")

            results = self.execute_sparql(sparql)

            return QueryResult(
                question=question,
                sparql_query=sparql,
                results=results,
                success=True
            )

        except Exception as e:
            return QueryResult(
                question=question,
                sparql_query=sparql,
                results=[],
                success=False,
                error=str(e)
            )

    def format_results(self, result: QueryResult) -> str:
        """Format query results for display."""
        output: list[str] = []
        output.append(f"Question: {result.question}")
        output.append(f"\nGenerated SPARQL:\n{result.sparql_query}")

        if not result.success:
            output.append(f"\nError: {result.error}")
            return "\n".join(output)

        output.append(f"\nResults ({len(result.results)} rows):")

        if not result.results:
            output.append("  No results found.")
        else:
            headers = list(result.results[0].keys())
            output.append("  " + " | ".join(headers))
            output.append("  " + "-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)))

            for row in result.results[:20]:
                values = [str(row.get(h, ""))[:50] for h in headers]
                output.append("  " + " | ".join(values))

            if len(result.results) > 20:
                output.append(f"  ... and {len(result.results) - 20} more rows")

        return "\n".join(output)


def interactive_mode() -> None:
    """Run interactive question-answering session."""
    print("=" * 60)
    print("GraphRAG: Natural Language to SPARQL (Ollama)")
    print("=" * 60)
    print("\nType your questions in natural language.")
    print("Type 'quit' or 'exit' to stop.\n")

    try:
        nl2sparql = NLToSPARQL()
    except ConnectionError as e:
        print(f"Error: {e}")
        return

    while True:
        try:
            question = input("\nQuestion: ").strip()

            if not question:
                continue

            if question.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            result = nl2sparql.query(question)
            print("\n" + nl2sparql.format_results(result))

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def demo() -> None:
    """Run demo with sample questions."""
    questions = [
        "What Samsung phones have a battery capacity greater than 5000mAh?",
        "Find phones good for gaming (high refresh rate and big battery)",
        "What are the top 5 phones with the best camera (highest megapixels)?",
        "Which brands manufacture phones with 5G support?",
        "Find phones released in 2023 with AMOLED display",
        "What phones does user pro_gamer_0000 like?",
        "List flagship phones with price over 900 EUR",
        "Show users interested in Gaming",
    ]

    print("=" * 60)
    print("GraphRAG Demo: Natural Language to SPARQL (Ollama)")
    print("=" * 60)

    try:
        nl2sparql = NLToSPARQL()
    except ConnectionError as e:
        print(f"Error: {e}")
        return

    for question in questions:
        print("\n" + "=" * 60)
        result = nl2sparql.query(question)
        print(nl2sparql.format_results(result))
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GraphRAG: NL to SPARQL (Ollama)")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample questions")
    parser.add_argument("--question", "-q", type=str, help="Ask a single question")
    parser.add_argument("--model", "-m", type=str, default=OLLAMA_MODEL, help="Ollama model to use")

    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.question:
        try:
            nl2sparql = NLToSPARQL(model=args.model)
            result = nl2sparql.query(args.question)
            print(nl2sparql.format_results(result))
        except ConnectionError as e:
            print(f"Error: {e}")
    else:
        interactive_mode()
