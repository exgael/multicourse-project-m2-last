"""
Explore Phones - Browse and filter smartphones by specifications.
"""

import streamlit as st
from pathlib import Path
from rdflib import Graph

st.set_page_config(
    page_title="Explore Phones",
    page_icon="phone",
    layout="wide"
)

ROOT_DIR = Path(__file__).parent.parent.parent
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


with st.spinner("Loading Knowledge Graph..."):
    kg = load_knowledge_graph()

st.title("Explore Phones")

# Sidebar with filters
with st.sidebar:
    st.header("Filters")

    brands_query = """
    PREFIX sp: <http://example.org/smartphone#>
    SELECT DISTINCT ?brandName WHERE {
        ?brand a sp:Brand ; sp:brandName ?brandName .
    } ORDER BY ?brandName
    """
    brands_list = [r["brandName"] for r in run_sparql(kg, brands_query)]

    selected_brands = st.multiselect("Brands", options=brands_list, default=[])

    st.divider()

    min_battery = st.slider("Min Battery (mAh)", 2000, 7000, 3000, 500)
    min_camera = st.slider("Min Camera (MP)", 8, 200, 12, 4)
    min_refresh = st.select_slider("Min Refresh Rate (Hz)", options=[60, 90, 120, 144, 165], value=60)

    st.divider()

    only_5g = st.checkbox("5G Support", value=False)
    only_nfc = st.checkbox("NFC Support", value=False)

    display_types = ["Any", "AMOLED", "LCD", "OLED", "IPS"]
    selected_display = st.selectbox("Display Type", display_types)

    st.divider()

    show_prices = st.checkbox("Show configurations & prices", value=True)
    result_limit = st.slider("Max Results", 10, 100, 25, 5)

# Build filters
filters = [f"FILTER(?battery >= {min_battery})"]
filters.append(f"FILTER(?camera >= {min_camera})")

if min_refresh > 60:
    filters.append(f"FILTER(?refresh >= {min_refresh})")

if selected_brands:
    brand_filter = " || ".join([f'?brandName = "{b}"' for b in selected_brands])
    filters.append(f"FILTER({brand_filter})")

if only_5g:
    filters.append("FILTER(?has5g = true)")

if only_nfc:
    filters.append("FILTER(?hasNfc = true)")

if selected_display != "Any":
    filters.append(f'FILTER(CONTAINS(LCASE(?display), "{selected_display.lower()}"))')

filter_clause = "\n    ".join(filters)

if show_prices:
    # Query with prices - subquery pattern: filter phones FIRST, then join prices
    explore_query = f"""
PREFIX sp: <http://example.org/smartphone#>

SELECT ?phoneName ?brandName ?year ?battery ?camera ?selfie ?refresh ?display ?processor ?has5g ?hasNfc ?storage ?ram ?price ?storeName
WHERE {{
    # Subquery: get filtered phones FIRST with LIMIT
    {{
        SELECT ?phone ?phoneName ?brandName ?year ?battery ?camera ?selfie ?refresh ?display ?processor ?has5g ?hasNfc
        WHERE {{
            ?phone a sp:BasePhone ;
                   sp:phoneName ?phoneName ;
                   sp:hasBrand/sp:brandName ?brandName .

            OPTIONAL {{ ?phone sp:releaseYear ?year }}
            OPTIONAL {{ ?phone sp:batteryCapacityMah ?battery }}
            OPTIONAL {{ ?phone sp:mainCameraMP ?camera }}
            OPTIONAL {{ ?phone sp:selfieCameraMP ?selfie }}
            OPTIONAL {{ ?phone sp:refreshRateHz ?refresh }}
            OPTIONAL {{ ?phone sp:displayType ?display }}
            OPTIONAL {{ ?phone sp:processorName ?processor }}
            OPTIONAL {{ ?phone sp:supports5G ?has5g }}
            OPTIONAL {{ ?phone sp:supportsNFC ?hasNfc }}

            {filter_clause}
        }}
        ORDER BY DESC(?battery) DESC(?camera)
        LIMIT {result_limit}
    }}

    # Then join with prices for only those phones
    OPTIONAL {{
        ?config a sp:PhoneConfiguration ;
                sp:hasBasePhone ?phone ;
                sp:storageGB ?storage ;
                sp:ramGB ?ram ;
                sp:hasPriceOffering ?offering .
        ?offering sp:priceValue ?price ;
                  sp:offeredBy/sp:storeName ?storeName .
    }}
}}
ORDER BY ?phoneName ?storage ?price
"""
else:
    explore_query = f"""
PREFIX sp: <http://example.org/smartphone#>

SELECT DISTINCT ?phoneName ?brandName ?year ?screen ?battery ?camera ?selfie ?refresh ?processor ?display ?has5g ?hasNfc
WHERE {{
    ?phone a sp:BasePhone ;
           sp:phoneName ?phoneName ;
           sp:hasBrand/sp:brandName ?brandName .

    OPTIONAL {{ ?phone sp:releaseYear ?year }}
    OPTIONAL {{ ?phone sp:screenSizeInches ?screen }}
    OPTIONAL {{ ?phone sp:batteryCapacityMah ?battery }}
    OPTIONAL {{ ?phone sp:mainCameraMP ?camera }}
    OPTIONAL {{ ?phone sp:selfieCameraMP ?selfie }}
    OPTIONAL {{ ?phone sp:refreshRateHz ?refresh }}
    OPTIONAL {{ ?phone sp:processorName ?processor }}
    OPTIONAL {{ ?phone sp:displayType ?display }}
    OPTIONAL {{ ?phone sp:supports5G ?has5g }}
    OPTIONAL {{ ?phone sp:supportsNFC ?hasNfc }}

    {filter_clause}
}}
ORDER BY DESC(?battery) DESC(?camera)
LIMIT {result_limit}
"""

st.subheader("Results")

results = run_sparql(kg, explore_query)

if results:
    st.success(f"Found {len(results)} results")

    display_data = []
    for r in results:
        if show_prices:
            # Format price with EUR
            price_val = r.get("price")
            if price_val:
                try:
                    price_str = f"{float(price_val):.0f}€"
                except (ValueError, TypeError):
                    price_str = "-"
            else:
                price_str = "-"

            # Format storage/RAM
            storage_val = r.get("storage")
            ram_val = r.get("ram")
            config_str = f"{storage_val}GB/{ram_val}GB" if storage_val and ram_val else "-"

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
        else:
            row = {
                "Phone": r.get("phoneName") or "-",
                "Brand": r.get("brandName") or "-",
                "Year": r.get("year") or "-",
                "Screen": r.get("screen") or "-",
                "Battery": r.get("battery") or "-",
                "Camera": r.get("camera") or "-",
                "Selfie": r.get("selfie") or "-",
                "Refresh": r.get("refresh") or "-",
                "Chipset": r.get("processor") or "-",
                "Display": r.get("display") or "-",
                "5G": "Yes" if r.get("has5g") == "true" else "No",
                "NFC": "Yes" if r.get("hasNfc") == "true" else "No",
            }
        display_data.append(row)

    st.dataframe(display_data, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Stats")

    col1, col2, col3, col4 = st.columns(4)

    batteries = [int(r.get("battery", 0) or 0) for r in results if r.get("battery")]
    brands_count = len(set(r.get("brandName") for r in results))
    fiveg_count = sum(1 for r in results if r.get("has5g") == "true")

    col1.metric("Results", len(results))
    col2.metric("Brands", brands_count)
    col3.metric("Avg Battery", f"{sum(batteries) // len(batteries) if batteries else 0} mAh")
    col4.metric("5G Phones", fiveg_count)

    with st.expander("Show SPARQL"):
        st.code(explore_query, language="sparql")

else:
    st.warning("No phones match your filters.")
