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
# Smartphone KG - SPARQL Guide

## Prefixes (ALWAYS include in queries)
PREFIX sp: <http://example.org/smartphone#>
PREFIX spv: <http://example.org/smartphone/vocab/>
PREFIX spi: <http://example.org/smartphone/instance/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

## Classes
- sp:Smartphone - A smartphone device
- sp:Brand - Phone manufacturer (e.g., Apple, Samsung)
- sp:Variant - Storage/RAM configuration of a phone
- sp:User - Synthetic user with preferences

## Defined Classes (inferred automatically)
- sp:LargeBatteryPhone - Smartphone with battery >= 5000mAh
- sp:FiveGPhone - Smartphone with 5G support
- sp:HighResolutionCameraPhone - Smartphone with camera >= 100MP

## Object Properties
- sp:hasBrand (Smartphone -> Brand) - Phone's manufacturer
- sp:manufactures (Brand -> Smartphone) - Inverse of hasBrand
- sp:hasVariant (Phone -> Variant) - Phone's storage configurations
- sp:variantOf (Variant -> Phone) - Inverse of hasVariant
- sp:interestedIn (User -> spv:Concept) - User's use-case interests (e.g., spv:ProGaming)
- sp:likes (User -> Smartphone) - User likes a phone

## Datatype Properties
- sp:phoneName (xsd:string) - Phone model name (ALWAYS select this for phones)
- sp:brandName (xsd:string) - Brand name
- sp:releaseYear (xsd:integer) - Year of release (integer, e.g., 2023)
- sp:batteryCapacityMah (xsd:integer) - Battery in mAh
- sp:mainCameraMP (xsd:integer) - Main camera megapixels
- sp:selfieCameraMP (xsd:integer) - Selfie camera megapixels
- sp:refreshRateHz (xsd:integer) - Screen refresh rate in Hz
- sp:supports5G (xsd:boolean) - Has 5G support (true/false, no quotes)
- sp:supportsNFC (xsd:boolean) - Has NFC support (true/false, no quotes)
- sp:processorName (xsd:string) - Processor/chipset name
- sp:displayType (xsd:string) - Display technology (contains "AMOLED", "LCD", "OLED", etc.)
- sp:storageGB (xsd:integer) - Storage in GB (on Variant, not Phone)
- sp:ramGB (xsd:integer) - RAM in GB (on Variant, not Phone)
- sp:priceEUR (xsd:decimal) - Price in Euros (on Variant, not Phone)
- sp:userId (xsd:string) - User identifier

## SKOS Use-Case Concepts (spv: namespace)
Users can be interested in these use-cases:
- spv:CasualPhotography - Camera >= 50MP + OLED display
- spv:ProPhotography - Camera >= 100MP + OLED + storage >= 512GB
- spv:CasualGaming - Refresh >= 90Hz + battery >= 4000mAh
- spv:ProGaming - Refresh >= 144Hz + screen >= 6.5" + battery >= 5000mAh
- spv:Business - Battery >= 5000mAh + camera >= 48MP
- spv:EverydayUse - Year >= 2020 + battery >= 3500mAh
- spv:Vlogging - Selfie camera >= 48MP
- spv:Minimalist - No NFC, no 5G

## SKOS Price Segment Concepts (spv: namespace)
- spv:Flagship - Price > 900 EUR
- spv:MidRange - Price 400-900 EUR
- spv:Budget - Price < 400 EUR
- spv:AfterMarket - No current retail price

## User ID Format
User IDs follow the pattern: {persona}_{number}
Examples: pro_gamer_0000, casual_photographer_0004, business_0002, minimalist_0001

## IMPORTANT: What's NOT in the KG
- Phones are NOT tagged with sp:suitableFor (no direct phone-usecase links)
- To find phones for a use-case, filter by specs (e.g., gaming = high refresh rate + big battery)

## Example Queries

# Find Samsung phones with battery > 5000mAh
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName ?battery WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName ;
         sp:hasBrand ?brand ;
         sp:batteryCapacityMah ?battery .
  ?brand sp:brandName "Samsung" .
  FILTER(?battery > 5000)
}

# Find phones good for gaming (high refresh rate + big battery)
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName ?refreshRate ?battery WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName ;
         sp:refreshRateHz ?refreshRate ;
         sp:batteryCapacityMah ?battery .
  FILTER(?refreshRate >= 120 && ?battery >= 5000)
}

# Find phones with 5G support (boolean = true without quotes)
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName ;
         sp:supports5G true .
}

# Find which brands have 5G phones
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?brandName WHERE {
  ?phone a sp:Smartphone ;
         sp:hasBrand ?brand ;
         sp:supports5G true .
  ?brand sp:brandName ?brandName .
}

# Find phones with AMOLED display (use CONTAINS for partial match)
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName ;
         sp:displayType ?display .
  FILTER(CONTAINS(LCASE(?display), "amoled"))
}

# Top 5 phones by camera megapixels (ORDER BY after WHERE closing brace)
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName ?camera WHERE {
  ?phone a sp:Smartphone ;
         sp:phoneName ?phoneName ;
         sp:mainCameraMP ?camera .
}
ORDER BY DESC(?camera)
LIMIT 5

# Get user's interests
PREFIX sp: <http://example.org/smartphone#>
PREFIX spv: <http://example.org/smartphone/vocab/>
SELECT ?userId ?interest WHERE {
  ?user a sp:User ;
        sp:userId ?userId ;
        sp:interestedIn ?uc .
  BIND(STRAFTER(STR(?uc), "vocab/") AS ?interest)
}
LIMIT 20

# Get phones liked by a specific user
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName WHERE {
  ?user a sp:User ;
        sp:userId "pro_gamer_0000" ;
        sp:likes ?phone .
  ?phone sp:phoneName ?phoneName .
}

# Find users interested in gaming
PREFIX sp: <http://example.org/smartphone#>
PREFIX spv: <http://example.org/smartphone/vocab/>
SELECT DISTINCT ?userId WHERE {
  ?user a sp:User ;
        sp:userId ?userId ;
        sp:interestedIn spv:ProGaming .
}

# Find phones with price > 900 EUR (price is on Variant)
PREFIX sp: <http://example.org/smartphone#>
SELECT DISTINCT ?phoneName ?price WHERE {
  ?variant sp:variantOf ?phone ;
           sp:priceEUR ?price .
  ?phone sp:phoneName ?phoneName .
  FILTER(?price > 900)
}
ORDER BY DESC(?price)
LIMIT 10
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
7. ORDER BY and LIMIT must come AFTER the closing brace }} of WHERE
8. Price (sp:priceEUR) is on Variant, not Phone - join via sp:variantOf
9. There is NO sp:suitableFor property - filter phones by specs instead
10. Return ONLY the SPARQL query, no explanations

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
        "Find phones good for gaming (high refresh rate and big battery)",
        "What are the top 5 phones with the best camera (highest megapixels)?",
        "Which brands manufacture phones with 5G support?",
        "Find phones released in 2023 with AMOLED display",
        "What phones does user pro_gamer_0000 like?",
        "List flagship phones with price over 900 EUR",
        "Show users interested in ProGaming",
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
