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


st.title("Amafone")

st.markdown("""
This application demonstrates a **Knowledge Graph** for smartphone data,
built using **RDF** (Resource Description Framework) and queryable via **SPARQL**.
""")

# Load KG
with st.spinner("Loading Knowledge Graph..."):
    kg = load_knowledge_graph()

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
