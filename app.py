"""
Smartphone Knowledge Graph - Landing Page
Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
from rdflib import Graph

ROOT_DIR = Path(__file__).parent
KG_PATH = ROOT_DIR / "data" / "rdf" / "knowledge_graph_full.ttl"

st.set_page_config(
    page_title="Amafone",
    page_icon="phone",
    layout="wide"
)


@st.cache_resource
def load_knowledge_graph():
    """Load the knowledge graph (cached)."""
    g = Graph()
    g.parse(KG_PATH, format="turtle")
    return g


def run_sparql(graph: Graph, query: str) -> list[dict]:
    """Execute SPARQL query and return results."""
    try:
        results = graph.query(query)
        rows = []
        for row in results:
            row_dict = {}
            if results.vars:
                for var in results.vars:
                    value = getattr(row, str(var), None)
                    row_dict[str(var)] = str(value) if value else None
            rows.append(row_dict)
        return rows
    except Exception:
        return []


# Load KG
with st.spinner("Loading Knowledge Graph..."):
    kg = load_knowledge_graph()

st.title("Amafone")
st.caption("Find the best smartphones based on real user reviews")

# Search bar
search_query = st.text_input("Search phones", placeholder="e.g. Samsung Galaxy, iPhone, Xiaomi...")

# Build query based on search
if search_query:
    search_terms = search_query.strip().lower()
    filter_clause = f'FILTER(CONTAINS(LCASE(?phoneName), "{search_terms}") || CONTAINS(LCASE(?brandName), "{search_terms}"))'
    title = f'Search results for "{search_query}"'
else:
    filter_clause = ""
    title = "Top Rated Phones"

# Query: phones with sentiment + prices, sorted by net sentiment
main_query = f"""
PREFIX sp: <http://example.org/smartphone#>

SELECT ?phoneName ?brandName ?netSentiment ?positivePercent
       ?storage ?ram ?price ?storeName
       ?year ?battery ?camera ?selfie ?refresh ?display ?processor ?has5g ?hasNfc
WHERE {{
    # Get sentiment aggregation
    {{
        SELECT ?phone ?phoneName ?brandName
               (SUM(?pos) - SUM(?neg) AS ?netSentiment)
               (ROUND(SUM(?pos) * 100.0 / (SUM(?pos) + SUM(?neg) + 0.001)) AS ?positivePercent)
        WHERE {{
            ?phone a sp:BasePhone ;
                   sp:phoneName ?phoneName ;
                   sp:hasBrand/sp:brandName ?brandName .
            OPTIONAL {{
                ?phone sp:hasSentiment ?sentiment .
                ?sentiment sp:positiveCount ?pos ;
                           sp:negativeCount ?neg .
            }}
            {filter_clause}
        }}
        GROUP BY ?phone ?phoneName ?brandName
        ORDER BY DESC(?netSentiment)
        LIMIT 30
    }}

    # Join with config and prices
    OPTIONAL {{
        ?config a sp:PhoneConfiguration ;
                sp:hasBasePhone ?phone ;
                sp:storageGB ?storage ;
                sp:ramGB ?ram ;
                sp:hasPriceOffering ?offering .
        ?offering sp:priceValue ?price ;
                  sp:offeredBy/sp:storeName ?storeName .
    }}

    # Get specs
    OPTIONAL {{ ?phone sp:releaseYear ?year }}
    OPTIONAL {{ ?phone sp:batteryCapacityMah ?battery }}
    OPTIONAL {{ ?phone sp:mainCameraMP ?camera }}
    OPTIONAL {{ ?phone sp:selfieCameraMP ?selfie }}
    OPTIONAL {{ ?phone sp:refreshRateHz ?refresh }}
    OPTIONAL {{ ?phone sp:displayType ?display }}
    OPTIONAL {{ ?phone sp:processorName ?processor }}
    OPTIONAL {{ ?phone sp:supports5G ?has5g }}
    OPTIONAL {{ ?phone sp:supportsNFC ?hasNfc }}
}}
ORDER BY DESC(?netSentiment) ?phoneName ?price
LIMIT 100
"""

results = run_sparql(kg, main_query)

st.subheader(title)

if results:
    st.success(f"Found {len(results)} results")

    display_data = []
    for r in results:
        # Format price
        price_val = r.get("price")
        price_str = f"{float(price_val):.0f}€" if price_val else "-"

        # Format config
        storage = r.get("storage")
        ram = r.get("ram")
        config_str = f"{storage}GB/{ram}GB" if storage and ram else "-"

        # Format sentiment
        net = r.get("netSentiment")
        pct = r.get("positivePercent")
        if net and pct:
            sentiment_str = f"{pct}% positive"
        else:
            sentiment_str = "-"

        # Format display (truncate if too long)
        display_val = r.get("display") or "-"
        if len(display_val) > 20:
            display_val = display_val[:20] + "..."

        # Format processor (truncate if too long)
        processor_val = r.get("processor") or "-"
        if len(processor_val) > 25:
            processor_val = processor_val[:25] + "..."

        row = {
            "Phone": r.get("phoneName") or "-",
            "Brand": r.get("brandName") or "-",
            "Year": r.get("year") or "-",
            "Rating": sentiment_str,
            "Config": config_str,
            "Price": price_str,
            "Store": r.get("storeName") or "-",
            "Battery": f"{r.get('battery')}mAh" if r.get("battery") else "-",
            "Camera": f"{r.get('camera')}MP" if r.get("camera") else "-",
            "Selfie": f"{r.get('selfie')}MP" if r.get("selfie") else "-",
            "Refresh": f"{r.get('refresh')}Hz" if r.get("refresh") else "-",
            "Display": display_val,
            "Processor": processor_val,
            "5G": "Yes" if r.get("has5g") == "true" else "No",
            "NFC": "Yes" if r.get("hasNfc") == "true" else "No",
        }
        display_data.append(row)

    st.dataframe(display_data, use_container_width=True, hide_index=True)
else:
    st.info("No phones found.")

st.divider()

# Stats
st.subheader("Knowledge Graph Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Triples", f"{len(kg):,}")

# Count entities
phones_query = "SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE { ?p a <http://example.org/smartphone#BasePhone> }"
brands_query = "SELECT (COUNT(DISTINCT ?b) AS ?c) WHERE { ?b a <http://example.org/smartphone#Brand> }"
configs_query = "SELECT (COUNT(DISTINCT ?c) AS ?c) WHERE { ?c a <http://example.org/smartphone#PhoneConfiguration> }"
users_query = "SELECT (COUNT(DISTINCT ?u) AS ?c) WHERE { ?u a <http://example.org/smartphone#User> }"

phones = run_sparql(kg, phones_query)
brands = run_sparql(kg, brands_query)
configs = run_sparql(kg, configs_query)
users = run_sparql(kg, users_query)

col2.metric("Base Phones", phones[0]['c'] if phones else "0")
col3.metric("Configurations", configs[0]['c'] if configs else "0")
col4.metric("Brands", brands[0]['c'] if brands else "0")

st.divider()

# Ontology overview
st.subheader("Ontology Structure")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Classes**")
    st.markdown("""
    - `sp:BasePhone` - Base phone model with specifications
    - `sp:PhoneConfiguration` - Storage/RAM variant
    - `sp:Brand` - Manufacturer
    - `sp:Store` - Retail store
    - `sp:PriceOffering` - Price at a store
    - `sp:TagSentiment` - Review sentiment data
    - `sp:User` - User with preferences
    """)

with col2:
    st.markdown("**Key Properties**")
    st.markdown("""
    - `sp:phoneName`, `sp:brandName`
    - `sp:batteryCapacityMah`, `sp:mainCameraMP`
    - `sp:refreshRateHz`, `sp:displayType`
    - `sp:supports5G`, `sp:supportsNFC`
    - `sp:storageGB`, `sp:ramGB`
    - `sp:priceValue`, `sp:hasPriceSegment`
    """)

st.divider()

# Sample data
st.subheader("Sample Data")

sample_query = """
PREFIX sp: <http://example.org/smartphone#>

SELECT ?phoneName ?brandName ?battery ?camera ?display
WHERE {
    ?phone a sp:BasePhone ;
           sp:phoneName ?phoneName ;
           sp:hasBrand/sp:brandName ?brandName .
    OPTIONAL { ?phone sp:batteryCapacityMah ?battery }
    OPTIONAL { ?phone sp:mainCameraMP ?camera }
    OPTIONAL { ?phone sp:displayType ?display }
}
LIMIT 10
"""

results = run_sparql(kg, sample_query)

if results:
    display_data = []
    for r in results:
        display_data.append({
            "Phone": r.get("phoneName", ""),
            "Brand": r.get("brandName", ""),
            "Battery (mAh)": r.get("battery", "N/A"),
            "Camera (MP)": r.get("camera", "N/A"),
            "Display": (r.get("display", "")[:50] + "...") if r.get("display") and len(r.get("display", "")) > 50 else r.get("display", "N/A"),
        })
    st.dataframe(display_data, use_container_width=True, hide_index=True)

st.divider()

# Namespaces
st.subheader("RDF Namespaces")

st.code("""
PREFIX sp: <http://example.org/smartphone#>
PREFIX spv: <http://example.org/smartphone/vocab/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
""", language="sparql")
