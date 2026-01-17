from pathlib import Path
import subprocess
from functools import wraps
from typing import Callable, Any

# ROOT DICRECTORY
ROOT_DIR: Path = Path(__file__).parent.parent


def step(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to print step name before execution."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"\n[{func.__name__}]")
        return func(*args, **kwargs)
    return wrapper

# DATA FILES
DATA_DIR: Path = ROOT_DIR / "data"
PHONES_JSON: Path = DATA_DIR / "phones.json"
PRICES_JSON: Path = DATA_DIR / "prices.json"
REVIEWS_JSON: Path = DATA_DIR / "reviews.json"
USERS_JSON: Path = DATA_DIR / "users.json"
VARIANTS_JSON: Path = DATA_DIR / "variants.json"
REVIEW_TAGS_JSON: Path = DATA_DIR / "review_tags.json"

# RML MAPPER
RML_MAPPER_JAR: Path = ROOT_DIR / "rmlmapper.jar"
RML_MAPPING: Path = ROOT_DIR / "mapper" / "data.ttl"

# KNOWLEDGE GRAPH FILES
KG_DIR: Path = ROOT_DIR / "knowledge_graph"
FINAL_KG_TTL: Path = KG_DIR / "final_knowledge_graph.ttl"
## SCHEMA
KG_SCHEMA_DIR: Path = KG_DIR / "schema"
KG_SKOS_TTL: Path = KG_SCHEMA_DIR / "skos.ttl"
KG_BASE_TTL: Path = KG_SCHEMA_DIR / "smartphone.ttl"
KG_SHACL_TTL: Path = KG_SCHEMA_DIR / "shapes.ttl"
## DATA
KG_DATA_DIR: Path = KG_DIR / "data"
FACTS_TTL: Path = KG_DATA_DIR / "facts.ttl"
CONSTRUCTED_TTL: Path = KG_DATA_DIR / "constructed.ttl"
INFERRED_TTL: Path = KG_DATA_DIR / "inferred.ttl"
LINKAGE_TTL: Path = KG_DATA_DIR / "linkage.ttl"
ALIGNMENT_TTL: Path = KG_DATA_DIR / "alignment.ttl"

class Pipeline:
    def __init__(self, skip_preprocess: bool = False, only_facts: bool = False):
        self.skip_preprocess = skip_preprocess
        self.only_facts: bool = only_facts
        (KG_DIR / "schema").mkdir(parents=True, exist_ok=True)
        KG_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _run(self, name: str, cmd: list[str]) -> None:
        print(f"\n[{name}]")
        subprocess.run(cmd, check=True, cwd=ROOT_DIR)

    @step
    def gen_review_tags(self) -> None:
        from preprocess.analyse_reviews import analyse_reviews
        analyse_reviews(
            input_path=REVIEWS_JSON,
            output_path=REVIEW_TAGS_JSON,
            model="qwen2.5:latest"
        )

    @step
    def gen_facts(self) -> None:
        from facts_mapper.generate_facts import generate_facts
        generate_facts(
            mapping=RML_MAPPING, 
            output=FACTS_TTL, 
            rml_mapper_jar=RML_MAPPER_JAR
        )

    @step
    def materialize_by_construct_and_inference(self) -> None:
        from rdf_enrichment.materialize import materialize

        sparql_constructs: list[tuple[str, str]] = [
            ("HighResolutionCameraPhone", """
                PREFIX sp: <http://example.org/smartphone#>
                CONSTRUCT { ?phone a sp:HighResolutionCameraPhone }
                WHERE {
                    ?phone a sp:Smartphone ;
                        sp:mainCameraMP ?mp .
                    FILTER(?mp >= 100)
                }
            """),

            ("LargeBatteryPhone", """
                PREFIX sp: <http://example.org/smartphone#>
                CONSTRUCT { ?phone a sp:LargeBatteryPhone }
                WHERE {
                    ?phone a sp:Smartphone ;
                        sp:batteryCapacityMah ?mah .
                    FILTER(?mah >= 5000)
                }
            """),

            ("InMarketPhone", """
                PREFIX sp: <http://example.org/smartphone#>
                CONSTRUCT { ?phone a sp:InMarketPhone }
                WHERE {
                    ?phone a sp:Smartphone .
                    ?variant sp:variantOf ?phone ;
                            sp:hasPrice ?price .
                }
            """),
        ]

        materialize(
            input_files=[KG_BASE_TTL, KG_SKOS_TTL, FACTS_TTL],
            sparql_constructs=sparql_constructs,
            output_materialized=CONSTRUCTED_TTL,
            output_inferred=INFERRED_TTL
        )

    @step
    def link(self) -> None:
        from rdf_enrichment.linkage import perform_linkage

        perform_linkage(
            input_file=FACTS_TTL,
            output_file=LINKAGE_TTL
        )

    @step
    def finalize(self) -> None:
        # write both schema and data to final KG
        with open(FINAL_KG_TTL, "w", encoding="utf-8") as final_file:
            for ttl_file in [
                KG_BASE_TTL,
                KG_SKOS_TTL,
                KG_SHACL_TTL,
                FACTS_TTL,
                CONSTRUCTED_TTL,
                INFERRED_TTL,
                LINKAGE_TTL,
                ALIGNMENT_TTL
            ]:
                if ttl_file.exists():
                    with open(ttl_file, "r", encoding="utf-8") as f:
                        final_file.write(f.read())
                        final_file.write("\n\n")

    def run(self) -> None:
        # Preprocessing
        self.gen_review_tags()

        # Facts generation
        self.gen_facts()

        # RDF enrichment
        self.materialize_by_construct_and_inference()
        self.link()

        # Finalize KG
        self.finalize()

if __name__ == "__main__":
    Pipeline().run()