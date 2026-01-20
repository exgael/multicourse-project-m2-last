import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import sys
import os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# RDFLib is required to read the existing KG
try:
    from rdflib import Graph, Namespace, RDF, URIRef
except ImportError:
    raise ImportError("Please install rdflib: pip install rdflib")

# Try importing your agent, fallback to Mock
try:
    from agents import DefaultAgent
    model = DefaultAgent()
except ImportError:
    class MockLLM:
        def action(self, sys, user):
            # Simulating a case: S24 Ultra exists in KG but lacks price. 
            # Pixel 9 is totally new.
            return json.dumps([
                {
                    "phone_id": "samsung_galaxy_s24_ultra_256gb_12gb",
                    "base_phone_id": "samsung_galaxy_s24_ultra",
                    "brand": "Samsung",
                    "phone_name": "Galaxy S24 Ultra",
                    "year": 2024,
                    "price_eur": 1449.00, # KG has this phone but might miss price
                    "battery_mah": 5000   # KG might already have this
                },
                {
                    "phone_id": "google_pixel_9_128gb_8gb",
                    "base_phone_id": "google_pixel_9",
                    "brand": "Google",
                    "phone_name": "Pixel 9",
                    "year": 2024,
                    "price_eur": 899.00
                }
            ])
    model = MockLLM()

# =============================================================================
# 1. SCHEMA & MAPPING
# =============================================================================

class PhoneSchema(BaseModel):
    # Identity (Always Required)
    phone_id: str = Field(description="brand_model_storage_ram (lowercase, underscores)")
    base_phone_id: str
    brand: str
    phone_name: str
    
    # Specs (Optional / Nullable)
    year: Optional[int] = None
    display_type: Optional[str] = None
    screen_size_inches: Optional[float] = None
    refresh_rate_hz: Optional[int] = None
    chipset: Optional[str] = None
    main_camera_mp: Optional[float] = None
    selfie_camera_mp: Optional[float] = None
    battery_mah: Optional[int] = None
    supports_5g: Optional[bool] = None
    nfc: Optional[bool] = None
    storage_gb: Optional[int] = None
    ram_gb: Optional[int] = None
    price_eur: Optional[float] = None

# Mapping JSON fields to your Ontology Predicates
# Update these URIs to match your smartphone.ttl exactly
SP = Namespace("http://example.org/smartphone#")
FIELD_TO_PREDICATE = {
    "year": SP.releaseYear,
    "display_type": SP.displayType,
    "screen_size_inches": SP.screenSize, # Check your ontology for exact name
    "refresh_rate_hz": SP.refreshRateHz,
    "chipset": SP.processorName,
    "main_camera_mp": SP.mainCameraMP,
    "selfie_camera_mp": SP.selfieCameraMP,
    "battery_mah": SP.batteryCapacityMah,
    "supports_5g": SP.supports5G,
    "nfc": SP.supportsNFC,
    "storage_gb": SP.storageGB,
    "ram_gb": SP.ramGB,
    "price_eur": SP.priceEUR
}

# =============================================================================
# 2. PROMPT
# =============================================================================

def generate_system_prompt():
    schema_json = PhoneSchema.model_json_schema()
    return f"""
    You are an expert Data Extractor.
    Extract smartphone specifications into a strict JSON list.
    
    ### TARGET OUTPUT
    {json.dumps(schema_json, indent=2)}

    ### RULES
    1. **Nulls:** If a spec is not mentioned, set to `null`.
    2. **IDs:** - phone_id: 'brand_model_storage_ram' (lowercase, snake_case).
       - base_phone_id: 'brand_model'.
    """

# =============================================================================
# 3. KNOWLEDGE GRAPH MERGE LOGIC
# =============================================================================

def merge_with_kg(extracted_items: List[Dict[str, Any]], kg_path: Path) -> List[Dict[str, Any]]:
    """
    Filters the extracted data against the existing Knowledge Graph.
    - If phone is new: Keep it.
    - If phone exists: Only keep fields that are MISSING in the KG.
    """
    if not kg_path.exists():
        print("KG file not found. Treating all items as new.")
        return extracted_items

    print(f"Loading Knowledge Graph from {kg_path}...")
    g = Graph()
    g.parse(str(kg_path), format="turtle")
    
    merged_items = []
    
    for item in extracted_items:
        # Construct the URI exactly as your pipeline does (sp:phone_{id})
        phone_uri = SP[f"phone_{item['phone_id']}"]
        
        # Check if phone exists
        # 
        exists = (phone_uri, RDF.type, SP.Smartphone) in g
        
        if not exists:
            # Case 1: New Phone -> Keep everything
            merged_items.append(item)
        else:
            # Case 2: Existing Phone -> Fill gaps
            print(f"Phone exists: {item['phone_name']}. Checking for missing values...")
            item_to_update = item.copy()
            
            # Iterate over spec fields
            for field, predicate in FIELD_TO_PREDICATE.items():
                extracted_val = item.get(field)
                
                # If we extracted nothing, skip
                if extracted_val is None:
                    continue
                
                # Check if KG already has this property
                # (uri, predicate, None) means "does any value exist?"
                if (phone_uri, predicate, None) in g:
                    # KG has value -> Ignore extracted (prefer KG authority)
                    # We set it to None so the downstream writer doesn't duplicate/overwrite
                    item_to_update[field] = None
                else:
                    # KG missing value -> Keep extracted (Fill the gap)
                    print(f"  -> Filling missing {field}: {extracted_val}")
            
            merged_items.append(item_to_update)

    return merged_items

# =============================================================================
# 4. MAIN FLOW
# =============================================================================

def run_extraction_batch(input_dir: Path, output_file: Path, kg_file: Path):
    print(f"--- [Extraction] Batch Processing ---")
    
    # 1. Extract from Text (LLM)
    extracted_phones = []
    if input_dir.exists():
        prompt = generate_system_prompt()
        for txt in input_dir.glob("*.txt"):
            print(f"Reading {txt.name}...")
            raw = model.action(prompt, txt.read_text(encoding="utf-8"))
            try:
                # Clean and Parse
                clean_json = raw.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_json)
                if isinstance(data, dict): data = [data]
                
                # Pydantic Validation
                for d in data:
                    obj = PhoneSchema(**d)
                    extracted_phones.append(obj.model_dump(exclude_none=False))
            except Exception as e:
                print(f"Error in {txt.name}: {e}")

    print(f"Extracted {len(extracted_phones)} candidates.")

    # 2. Merge/Filter against KG
    final_list = merge_with_kg(extracted_phones, kg_file)
    
    # 3. Save JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2)
    
    print(f"--- Saved {len(final_list)} processed items to {output_file} ---")