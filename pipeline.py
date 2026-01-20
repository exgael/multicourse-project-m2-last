import json
import uuid
import re
from rdflib import Graph, Namespace, RDF, RDFS, XSD, URIRef, Literal
from agents import DefaultAgent
from config_loader import OntologyConfigLoader
# =============================================================================
# 1. SETUP & MOCK LLM
# =============================================================================

class MockLLM:
    """
    Simulates the LLM for demonstration purposes. 
    Replace the logic inside 'action' with your actual API call.
    """
    def action(self, system_prompt: str, user_prompt: str) -> str:
        # In a real scenario, you send these prompts to GPT-4/Claude
        # Here, we simulate a perfect JSON response based on a hypothetical article
        
        print(f"\n[LLM] Received System Prompt with {len(system_prompt)} chars.")
        print(f"[LLM] Processing User Prompt: '{user_prompt[:50]}...'")
        
        # Simulating the extraction for the article provided in 'main'
        mock_response = {
            "name": "Samsung Galaxy S24 Ultra",
            "model_key": "SM-S928B",
            "brand": "Samsung", # Matches SKOS Brand
            "price_usd": 1299.99,
            "specs": {
                "processor": "Snapdragon 8 Gen 3", # Matches SKOS Concept
                "os": "Android 14",                # Matches SKOS Concept
                "screen_size": "6.8 inches",
                "ram": "12GB"
            }
        }
        return json.dumps(mock_response)

# Initialize the model wrapper
#model = MockLLM()
model = DefaultAgent()
# Namespaces
EX = Namespace("http://example.org/smartphone#")
THES = Namespace("http://example.org/smartphone/thesaurus#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# =============================================================================
# 2. SKOS CONTEXT LOADER
# =============================================================================

def load_thesaurus_context(ttl_file_path: str):
    """
    Parses the SKOS Turtle file.
    Returns:
      1. allowed_terms: Dict for the LLM System Prompt (human-readable lists)
      2. uri_mapper: Dict for the Code (Label string -> URIRef)
    """
    g = Graph()
    with open(ttl_file_path, "r") as file:
        ttl_file_content = file.read()
        
    g.parse(data=ttl_file_content, format="turtle")
    
    allowed_terms = {
        "Brands": [],
        "Processors": [],
        "OS": [],
        "Features": []
    }
    
    uri_mapper = {}

    # Helper to process concepts by scheme
    def process_scheme(scheme_uri, dict_key):
        # Find concepts in this scheme
        for concept in g.subjects(SKOS.inScheme, scheme_uri):
            # Get preferred label
            pref_label = g.value(concept, SKOS.prefLabel)
            if pref_label:
                label_str = str(pref_label)
                allowed_terms[dict_key].append(label_str)
                uri_mapper[label_str] = concept
                
                # Also map altLabels to the SAME concept URI for robustness
                for alt in g.objects(concept, SKOS.altLabel):
                    uri_mapper[str(alt)] = concept

    # You need to know your Scheme URIs from the ontology
    process_scheme(THES.ProcessorBrandScheme, "Processors")
    process_scheme(THES.OperatingSystemScheme, "OS")
    # Note: Brands were defined as Classes in your first file, but assumed as concepts here. 
    # If Brands are Classes in ontology, we map them manually or query them similarly.
    # For this example, let's assume we extract Brands from the ontology classes or SKOS.
    # We will manually populate Brands from the Class definitions for the prompt context.
    
    # Manually adding Brands based on typical knowledge or a specific Brand scheme
    # If Brands are owl:Class instances, we'd query ?s a :Brand.
    # Let's populate the prompt list for Brands manually or via query if they exist in graph
    allowed_terms["Brands"] = ["Apple", "Samsung", "Google", "Xiaomi", "OnePlus"]
    # Map them to URIs (assuming pattern :Brand_Name)
    for b in allowed_terms["Brands"]:
        uri_mapper[b] = EX[f"Brand_{b}"]

    return allowed_terms, uri_mapper

# =============================================================================
# 3. PROMPT ENGINEERING
# =============================================================================

def generate_system_prompt(target_schema: str, allowed_vocabularies: dict):
    """
    Generates a prompt independent of the specific ontology version.
    
    Args:
        target_schema (str): A string representation of the desired JSON structure.
        allowed_vocabularies (dict): A dictionary where Key = Category Name, Value = List of allowed strings.
    """
    
    # Dynamically build the "Allowed Options" section based on input dict
    vocabulary_section = ""
    for category, terms in allowed_vocabularies.items():
        # strict=False allows the JSON dump to fit into the f-string without errors
        vocabulary_section += f"- **{category}:** {json.dumps(terms)}\n"

    return f"""
You are an expert Knowledge Graph Engineer.
Your goal is to extract structured data from technical articles to populate a Knowledge Graph.

### TASK
Extract data from the user text strictly following the **OUTPUT SCHEMA** provided below.

### CRITICAL CONSTRAINTS
1. **Strict Vocabulary:** For any field in the schema that corresponds to a category in the "ALLOWED OPTIONS" below, you MUST select the value EXACTLY from the provided lists.
2. **No Invention:** If the text mentions an entity (e.g., a specific brand or component) that is NOT in the allowed list, set that field to `null`. Do not invent new terms.
3. **Format:** Output ONLY raw JSON. No markdown formatting.

### ALLOWED OPTIONS (Controlled Vocabulary)
{vocabulary_section}

### OUTPUT SCHEMA
{target_schema}
"""
# =============================================================================
# 4. RDF BUILDER
# =============================================================================

def create_turtle_from_json(data: dict, uri_mapper: dict) -> str:
    """
    Converts the LLM-extracted JSON into a Valid Turtle String 
    adhering to the Smartphone Ontology.
    """
    g = Graph()
    g.bind(":", EX)
    g.bind("thes", THES)
    g.bind("xsd", XSD)

    # 1. Mint URI for the Smartphone (Slugify model key)
    # Sanitizing the model key for URI safety
    safe_key = re.sub(r'[^a-zA-Z0-9]', '_', data['model_key'])
    phone_uri = EX[f"phone_{safe_key}"]
    
    # 2. Add Type and Base Properties (Mandatory per SHACL)
    g.add((phone_uri, RDF.type, EX.Smartphone))
    g.add((phone_uri, EX.hasName, Literal(data['name'], datatype=XSD.string)))
    g.add((phone_uri, EX.hasModelKey, Literal(data['model_key'], datatype=XSD.string)))
    
    if data.get('price_usd'):
        g.add((phone_uri, EX.hasPrice, Literal(data['price_usd'], datatype=XSD.float)))

    # 3. Link Brand (Object Property)
    # We use the Mapper to find the correct URI, or fallback to a string if something went wrong
    brand_str = data.get('brand')
    if brand_str and brand_str in uri_mapper:
        g.add((phone_uri, EX.hasBrand, uri_mapper[brand_str]))
    elif brand_str:
        # Fallback: If LLM hallucinated a brand not in our mapper, we log it or skip.
        # For now, we assume strict adherence or fallback to a generated URI
        fallback_uri = EX[f"Brand_{brand_str.replace(' ', '')}"]
        g.add((phone_uri, EX.hasBrand, fallback_uri))

    # 4. Link Specs (Object Properties vs Datatypes)
    # Note: In your original ontology, :hasProcessor was xsd:string. 
    # To fully utilize SKOS, we should ideally use ObjectProperties.
    # Here, I will assume we updated the ontology to support URIs, 
    # OR we will store the URI as a string literal if strict schema compliance is required.
    
    specs = data.get('specs', {})
    
    # Processor
    proc_str = specs.get('processor')
    if proc_str and proc_str in uri_mapper:
        # Linking to the Thesaurus Concept
        g.add((phone_uri, EX.hasProcessor, uri_mapper[proc_str]))
    
    # OS (Assuming you add :hasOS to your ontology)
    os_str = specs.get('os')
    if os_str and os_str in uri_mapper:
        g.add((phone_uri, EX.hasOS, uri_mapper[os_str]))

    # Datatype Specs
    if specs.get('ram'):
        g.add((phone_uri, EX.hasRAM, Literal(specs['ram'], datatype=XSD.string)))
    if specs.get('screen_size'):
        g.add((phone_uri, EX.hasScreenSize, Literal(specs['screen_size'], datatype=XSD.string)))

    return g.serialize(format="turtle")

# =============================================================================
# 5. MAIN PIPELINE EXECUTION
# =============================================================================

# This string represents the SKOS file you provided earlier
SKOS_FILE_CONTENT = """
@prefix : <http://example.org/smartphone/thesaurus#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

:ProcessorBrandScheme a skos:ConceptScheme .
:OperatingSystemScheme a skos:ConceptScheme .

:Snapdragon8Gen3 a skos:Concept ;
    skos:prefLabel "Snapdragon 8 Gen 3"@en ;
    skos:inScheme :ProcessorBrandScheme .

:Android14 a skos:Concept ;
    skos:prefLabel "Android 14"@en ;
    skos:inScheme :OperatingSystemScheme .

# ... (Imagine the rest of your SKOS file is here)
"""

def run_pipeline(article_text):
    loader = OntologyConfigLoader(r"ontology\shacl.ttl", r"ontology\skos.ttl")

    # 2. Get the Vocabularies
    allowed_vocabs = loader.get_allowed_vocabularies()
    # Result: {'ProcessorBrand': ['Snapdragon 8 Gen 3', ...], 'OperatingSystem': ['Android 14'...]}

    # 3. Get the Schema
    # It matches :hasProcessor -> ProcessorBrand list automatically
    dynamic_schema = loader.generate_llm_schema("SmartphoneShape")

    # 4. Generate the Prompt
    # Now you use the function from the previous step
    sys_prompt = generate_system_prompt(dynamic_schema, allowed_vocabs)

    print("\n--- 3. Calling LLM ---")
    json_str = model.action(sys_prompt, article_text)
    
    try:
        extracted_data = json.loads(json_str)
        print("Extraction Successful.")
    except json.JSONDecodeError:
        print("Error: LLM returned invalid JSON.")
        return

    print("\n--- 4. Building RDF ---")
    turtle_output = create_turtle_from_json(extracted_data, uri_mapper)
    
    print("\n--- FINAL OUTPUT (Turtle) ---")
    print(turtle_output)

# =============================================================================
# RUN
# =============================================================================

article = """
Review: The new Samsung Galaxy S24 Ultra is a beast. 
It features the latest Snapdragon 8 Gen 3 chipset and runs on Android 14 out of the box.
With a massive 6.8 inch screen and 12GB of RAM, it handles everything.
The price is steep at $1299.99, but for the model SM-S928B, it's worth it.
"""

if __name__ == "__main__":
    run_pipeline(article)