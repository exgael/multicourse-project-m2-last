"""
GraphRAG Approach 1: Natural Language to SPARQL
Transforms natural language questions into SPARQL queries using LLM (Ollama).
"""

import json
import requests
from pathlib import Path
from dataclasses import dataclass
from rdflib import Graph

ROOT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
FINAL_KG_TTL = OUTPUT_DIR / "final_knowledge_graph.ttl"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"  

# Schema context for the LLM
ONTOLOGY_CONTEXT = """
# Smartphone Knowledge Graph Schema

## Prefixes
PREFIX sp: <http://example.org/smartphone#>
PREFIX spv: <http://example.org/smartphone/vocab/>
PREFIX spi: <http://example.org/smartphone/instance/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

## Classes
- sp:Smartphone - A smartphone device
- sp:Brand - Phone manufacturer (e.g., Apple, Samsung)
- sp:Variant - Storage/RAM configuration of a phone
- sp:User - Synthetic user with preferences

## Defined Classes (inferred)
- sp:LargeBatteryPhone - Smartphone with battery >= 5000mAh
- sp:FiveGPhone - Smartphone with 5G support
- sp:HighResolutionCameraPhone - Smartphone with camera >= 100MP

## Object Properties
- sp:hasBrand (Smartphone -> Brand) - Phone's manufacturer
- sp:manufactures (Brand -> Smartphone) - Inverse of hasBrand
- sp:hasVariant (Phone -> Variant) - Phone's storage configurations
- sp:variantOf (Variant -> Phone) - Inverse of hasVariant
- sp:interestedIn (User -> skos:Concept) - User's use-case interests
- sp:likes (User -> Smartphone) - User likes a phone
- sp:suitableFor (Smartphone -> skos:Concept) - Phone is suitable for a use-case

## Datatype Properties
- sp:phoneName (xsd:string) - Phone model name
- sp:brandName (xsd:string) - Brand name
- sp:releaseYear (xsd:integer) - Year of release
- sp:batteryCapacityMah (xsd:integer) - Battery in mAh
- sp:mainCameraMP (xsd:integer) - Main camera megapixels
- sp:selfieCameraMP (xsd:integer) - Selfie camera megapixels
- sp:refreshRateHz (xsd:integer) - Screen refresh rate
- sp:supports5G (xsd:boolean) - Has 5G support
- sp:supportsNFC (xsd:boolean) - Has NFC support
- sp:processorName (xsd:string) - Processor/chipset name
- sp:displayType (xsd:string) - Display technology (AMOLED, LCD, etc.)
- sp:storageGB (xsd:integer) - Storage in GB (on Variant)
- sp:ramGB (xsd:integer) - RAM in GB (on Variant)
- sp:priceEUR (xsd:decimal) - Price in Euros (on Variant)
- sp:userId (xsd:string) - User identifier
- sp:persona (xsd:string) - User persona type

## SKOS Concepts (Use Cases) - spv: namespace
- spv:Photography, spv:CasualPhotography, spv:ProPhotography
- spv:Gaming, spv:CasualGaming, spv:ProGaming
- spv:Business, spv:EverydayUse, spv:Vlogging
- spv:VintageCollector, spv:BasicPhone

## SKOS Concepts (Price Segments) - spv: namespace
- spv:Flagship (> 900 EUR)
- spv:MidRange (400-900 EUR)
- spv:Budget (< 400 EUR)
- spv:AfterMarket (no current price)

## Instance URI patterns
- Phones: spi:phone/{phone_id} (e.g., spi:phone/apple_apple_iphone_15_pro)
- Brands: spi:brand/{brand_name} (e.g., spi:brand/Apple)
- Users: spi:user/{user_id} (e.g., spi:user/gamer_0001)
- Variants: spi:variant/{variant_id}

## Example Queries

# Find all Samsung phones with battery > 5000mAh
PREFIX sp: <http://example.org/smartphone#>
SELECT ?phoneName ?battery WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName ;
         sp:hasBrand ?brand ;
         sp:batteryCapacityMah ?battery .
  ?brand sp:brandName "Samsung" .
  FILTER(?battery > 5000)
}

# Find phones suitable for gaming
PREFIX sp: <http://example.org/smartphone#>
PREFIX spv: <http://example.org/smartphone/vocab/>
SELECT ?phoneName WHERE {
  ?phone sp:phoneName ?phoneName ;
         sp:suitableFor spv:Gaming .
}

# Get user interests
PREFIX sp: <http://example.org/smartphone#>
SELECT ?userId ?interest WHERE {
  ?user sp:userId ?userId ;
        sp:interestedIn ?uc .
  BIND(STRAFTER(STR(?uc), "vocab/") AS ?interest)
}
"""

SYSTEM_PROMPT = """You are a SPARQL query generator for a smartphone knowledge graph.

Given a natural language question, generate a valid SPARQL query that answers it.

IMPORTANT RULES:
1. ALWAYS start with PREFIX declarations:
   PREFIX sp: <http://example.org/smartphone#>
   PREFIX spv: <http://example.org/smartphone/vocab/>
2. Use ONLY the prefixes, classes, and properties defined in the schema
3. Use proper SPARQL syntax
4. ALWAYS include sp:phoneName when selecting phone information
5. Use SELECT DISTINCT to avoid duplicates
6. For text matching, use case-insensitive FILTER with CONTAINS or LCASE
7. ORDER BY and LIMIT must come AFTER the closing brace of WHERE
8. Return ONLY the SPARQL query, no explanations

Schema:
{schema}
"""

USER_PROMPT = """Question: {question}

Generate a SPARQL query to answer this question. Return ONLY the query."""


@dataclass
class QueryResult:
    """Result of a GraphRAG query."""
    question: str
    sparql_query: str
    results: list[dict]
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
            code_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    code_lines.append(line)
            sparql = "\n".join(code_lines) if code_lines else sparql
        
        return sparql
    
    def execute_sparql(self, sparql: str) -> list[dict]:
        """Execute SPARQL query on the knowledge graph."""
        self.load_graph()
        
        results = []
        query_results = self.graph.query(sparql)
        
        for row in query_results:
            result_dict = {}
            for var in query_results.vars:
                value = getattr(row, str(var), None)
                result_dict[str(var)] = str(value) if value else None
            results.append(result_dict)
        
        return results
    
    def query(self, question: str) -> QueryResult:
        """
        Process a natural language question:
        1. Generate SPARQL query
        2. Execute query
        3. Return results
        """
        try:
            # Generate SPARQL
            sparql = self.generate_sparql(question)
            print(f"\nGenerated SPARQL:\n{sparql}\n")
            
            # Execute query
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
                sparql_query=sparql if 'sparql' in locals() else "",
                results=[],
                success=False,
                error=str(e)
            )
    
    def format_results(self, result: QueryResult) -> str:
        """Format query results for display."""
        output = []
        output.append(f"Question: {result.question}")
        output.append(f"\nGenerated SPARQL:\n{result.sparql_query}")
        
        if not result.success:
            output.append(f"\nError: {result.error}")
            return "\n".join(output)
        
        output.append(f"\nResults ({len(result.results)} rows):")
        
        if not result.results:
            output.append("  No results found.")
        else:
            # Format as table
            if result.results:
                headers = list(result.results[0].keys())
                output.append("  " + " | ".join(headers))
                output.append("  " + "-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)))
                
                for row in result.results[:20]:  # Limit to 20 rows
                    values = [str(row.get(h, ""))[:50] for h in headers]
                    output.append("  " + " | ".join(values))
                
                if len(result.results) > 20:
                    output.append(f"  ... and {len(result.results) - 20} more rows")
        
        return "\n".join(output)


def interactive_mode():
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


def demo():
    """Run demo with sample questions."""
    questions = [
        "What Samsung phones have a battery capacity greater than 5000mAh?",
        "Show me all phones suitable for gaming",
        "What are the top 5 phones with the best camera (highest megapixels)?",
        "Which brands manufacture phones with 5G support?",
        "Find phones released in 2023 with AMOLED display",
        "What phones does user gamer_0001 like?",
        "List all flagship phones (price > 900 EUR)",
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
