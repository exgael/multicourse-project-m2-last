"""
Smartphone Knowledge Graph - Demo Interface
A simple Streamlit app to demonstrate the KG capabilities.

Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
from rdflib import Graph

# Page config
st.set_page_config(
    page_title="Smartphone KG Demo",
    page_icon="📱",
    layout="wide"
)

# Paths
ROOT_DIR = Path(__file__).parent
KG_PATH = ROOT_DIR / "data" / "rdf" / "knowledge_graph_full.ttl"


@st.cache_resource
def load_knowledge_graph():
    """Load the knowledge graph (cached)."""
    g = Graph()
    g.parse(KG_PATH, format="turtle")
    return g


def run_sparql(graph: Graph, query: str) -> list[dict]:
    """Execute SPARQL query and return results as list of dicts."""
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
    except Exception as e:
        st.error(f"Query error: {e}")
        return []


# Load KG
with st.spinner("Loading Knowledge Graph..."):
    kg = load_knowledge_graph()

# Title
st.title("📱 Smartphone Knowledge Graph")
st.markdown("*Demo interface for querying smartphone data*")

# Sidebar with stats
with st.sidebar:
    st.header("Knowledge Graph Stats")
    st.metric("Total Triples", f"{len(kg):,}")

    # Count entities
    phones_query = "SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE { ?p a <http://example.org/smartphone#BasePhone> }"
    brands_query = "SELECT (COUNT(DISTINCT ?b) AS ?c) WHERE { ?b a <http://example.org/smartphone#Brand> }"
    configs_query = "SELECT (COUNT(DISTINCT ?c) AS ?c) WHERE { ?c a <http://example.org/smartphone#PhoneConfiguration> }"

    phones = run_sparql(kg, phones_query)
    brands = run_sparql(kg, brands_query)
    configs = run_sparql(kg, configs_query)

    col1, col2 = st.columns(2)
    col1.metric("Phones", phones[0]['c'] if phones else "?")
    col2.metric("Brands", brands[0]['c'] if brands else "?")
    st.metric("Configurations", configs[0]['c'] if configs else "?")

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔍 Query", "📊 Explore", "💡 Examples"])

# Tab 1: SPARQL Query
with tab1:
    st.subheader("SPARQL Query")

    default_query = """PREFIX sp: <http://example.org/smartphone#>

SELECT ?phoneName ?brandName ?battery ?camera
WHERE {
    ?phone a sp:BasePhone ;
           sp:phoneName ?phoneName ;
           sp:hasBrand/sp:brandName ?brandName ;
           sp:batteryCapacityMah ?battery ;
           sp:mainCameraMP ?camera .
    FILTER(?battery > 5000)
}
ORDER BY DESC(?battery)
LIMIT 10"""

    query = st.text_area("Enter SPARQL query:", value=default_query, height=250)

    if st.button("Run Query", type="primary"):
        with st.spinner("Executing query..."):
            results = run_sparql(kg, query)

        if results:
            st.success(f"Found {len(results)} results")
            st.dataframe(results, use_container_width=True)
        else:
            st.warning("No results found")

# Tab 2: Explore
with tab2:
    st.subheader("Explore Phones")

    # Get all brands
    brands_query = """
    PREFIX sp: <http://example.org/smartphone#>
    SELECT DISTINCT ?brandName WHERE {
        ?brand a sp:Brand ; sp:brandName ?brandName .
    } ORDER BY ?brandName
    """
    brands_list = [r['brandName'] for r in run_sparql(kg, brands_query)]

    col1, col2 = st.columns(2)

    with col1:
        selected_brand = st.selectbox("Select Brand", ["All"] + brands_list)

    with col2:
        min_battery = st.slider("Min Battery (mAh)", 3000, 7000, 4000, 500)

    # Build query based on filters
    filter_clause = f"FILTER(?battery >= {min_battery})"
    if selected_brand != "All":
        filter_clause += f'\n    FILTER(?brandName = "{selected_brand}")'

    explore_query = f"""
    PREFIX sp: <http://example.org/smartphone#>
    SELECT ?phoneName ?brandName ?battery ?camera ?refresh ?display
    WHERE {{
        ?phone a sp:BasePhone ;
               sp:phoneName ?phoneName ;
               sp:hasBrand/sp:brandName ?brandName .
        OPTIONAL {{ ?phone sp:batteryCapacityMah ?battery }}
        OPTIONAL {{ ?phone sp:mainCameraMP ?camera }}
        OPTIONAL {{ ?phone sp:refreshRateHz ?refresh }}
        OPTIONAL {{ ?phone sp:displayType ?display }}
        {filter_clause}
    }}
    ORDER BY DESC(?battery)
    LIMIT 20
    """

    results = run_sparql(kg, explore_query)

    if results:
        st.dataframe(results, use_container_width=True)
    else:
        st.info("No phones match the filters")

# Tab 3: Example Queries
with tab3:
    st.subheader("Example Queries")

    examples = {
        "Gaming Phones (120Hz+, 5000mAh+)": """PREFIX sp: <http://example.org/smartphone#>
SELECT ?phoneName ?brandName ?refresh ?battery
WHERE {
    ?phone a sp:BasePhone ;
           sp:phoneName ?phoneName ;
           sp:hasBrand/sp:brandName ?brandName ;
           sp:refreshRateHz ?refresh ;
           sp:batteryCapacityMah ?battery .
    FILTER(?refresh >= 120 && ?battery >= 5000)
}
ORDER BY DESC(?refresh)
LIMIT 15""",

        "Best Camera Phones (100MP+)": """PREFIX sp: <http://example.org/smartphone#>
SELECT ?phoneName ?brandName ?camera
WHERE {
    ?phone a sp:BasePhone ;
           sp:phoneName ?phoneName ;
           sp:hasBrand/sp:brandName ?brandName ;
           sp:mainCameraMP ?camera .
    FILTER(?camera >= 100)
}
ORDER BY DESC(?camera)
LIMIT 15""",

        "Phones with Prices": """PREFIX sp: <http://example.org/smartphone#>
SELECT ?phoneName ?storage ?price ?storeName
WHERE {
    ?phone a sp:BasePhone ;
           sp:phoneName ?phoneName ;
           sp:hasConfiguration ?config .
    ?config sp:storageGB ?storage .
    ?offering sp:forConfiguration ?config ;
              sp:priceValue ?price ;
              sp:offeredBy/sp:storeName ?storeName .
}
ORDER BY ?phoneName ?price
LIMIT 20""",

        "Sentiment Analysis": """PREFIX sp: <http://example.org/smartphone#>
SELECT ?phoneName ?tag ?positive ?negative
WHERE {
    ?phone a sp:BasePhone ;
           sp:phoneName ?phoneName ;
           sp:hasSentiment ?sentiment .
    ?sentiment sp:forTag ?tag ;
               sp:positiveCount ?positive ;
               sp:negativeCount ?negative .
}
ORDER BY ?phoneName ?tag
LIMIT 20""",

        "Brand Statistics": """PREFIX sp: <http://example.org/smartphone#>
SELECT ?brandName (COUNT(?phone) AS ?phoneCount)
WHERE {
    ?phone a sp:BasePhone ;
           sp:hasBrand/sp:brandName ?brandName .
}
GROUP BY ?brandName
ORDER BY DESC(?phoneCount)"""
    }

    selected_example = st.selectbox("Choose an example:", list(examples.keys()))

    st.code(examples[selected_example], language="sparql")

    if st.button("Run Example", type="primary"):
        with st.spinner("Executing..."):
            results = run_sparql(kg, examples[selected_example])

        if results:
            st.success(f"Found {len(results)} results")
            st.dataframe(results, use_container_width=True)
        else:
            st.warning("No results found")

# Footer
st.divider()
st.caption("Smartphone Knowledge Graph - University Project Demo")
