from pathlib import Path
import subprocess
from functools import wraps
from collections.abc import Callable
from typing import Any

ROOT_DIR: Path = Path(__file__).parent.parent
OUTPUT: Path = ROOT_DIR / "output"
TEMP: Path = ROOT_DIR / OUTPUT / "temp"
KG_DIR: Path = ROOT_DIR / "kg"

DATA_DIR: Path = ROOT_DIR / "data"
ARTICLES_DIR: Path = DATA_DIR / "articles"
EXTRACTED_JSON: Path = OUTPUT / "data" / "extracted_phones.json"
PHONES_JSON: Path = DATA_DIR / "phones.json"
PRICES_JSON: Path = DATA_DIR / "prices.json"
VARIANTS_JSON: Path = DATA_DIR / "variants.json"
REVIEW_TAGS_JSON: Path = DATA_DIR / "review_tags.json"
KG_DATA_DIR: Path = TEMP / "kg_data"
EXTRACTED_TTL: Path = KG_DATA_DIR / "extracted.ttl"

STORE_PRICES_FILE: Path = OUTPUT / "data" / "store_prices.json"
CONFIGURATIONS_FILE: Path = OUTPUT / "data" / "phones_configuration.json"
REVIEW_SENTIMENTS_FILE: Path = OUTPUT / "data" / "review_sentiments.json"
USER_DATA_DIR: Path = OUTPUT / "data" / "users"

RML_MAPPER_JAR: Path = ROOT_DIR / "rmlmapper.jar"
RML_MAPPING: Path = KG_DIR / "rml" / "mapper.ttl"

FINAL_KG_TTL: Path = OUTPUT / "final_knowledge_graph.ttl"

KG_SCHEMA_DIR: Path = KG_DIR / "schema"
KG_SKOS_TTL: Path = KG_SCHEMA_DIR / "skos.ttl"
KG_BASE_TTL: Path = KG_SCHEMA_DIR / "smartphone.ttl"
KG_SHACL_TTL: Path = KG_SCHEMA_DIR / "shapes.ttl"

KG_DATA_DIR: Path = TEMP / "kg_data"
FACTS_TTL: Path = KG_DATA_DIR / "facts.ttl"
CONSTRUCTED_TTL: Path = KG_DATA_DIR / "constructed.ttl"
INFERRED_TTL: Path = KG_DATA_DIR / "inferred.ttl"
LINKAGE_TTL: Path = KG_DATA_DIR / "linkage.ttl"
ALIGNMENT_TTL: Path = KG_DATA_DIR / "alignment.ttl"


def step(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"\n[{func.__name__}]")
        return func(*args, **kwargs)
    return wrapper


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
    def process_prices(self) -> None:
        from preprocess.process_prices import process_prices
        process_prices(
            prices_file=PRICES_JSON,
            store_prices_file=STORE_PRICES_FILE,
        )

    @step
    def generate_configurations(self) -> None:
        """Generate phone configurations from phones, variants, and store prices.

        Dependency chain:
        phones.json ─────────────────────────┐
        variants.json ───────────────────────┼──► phones_merged.json
        store_prices.json (variant filter) ──┘
        """
        from preprocess.generate_configurations import generate_configurations
        generate_configurations(
            phones_file=PHONES_JSON,
            variants_file=VARIANTS_JSON,
            store_prices_file=STORE_PRICES_FILE,
            output_file=CONFIGURATIONS_FILE,
        )

    @step
    def aggregate_review_sentiments(self) -> None:
        """Aggregate review tags into per-phone sentiment counts."""
        from preprocess.aggregate_review_sentiments import aggregate_review_sentiments
        aggregate_review_sentiments(
            review_tags_file=REVIEW_TAGS_JSON,
            output_file=REVIEW_SENTIMENTS_FILE,
        )

    @step
    def gen_user_data(self) -> None:
        from preprocess.generate_users import generate_users
        generate_users(
            phone_configurations_file=CONFIGURATIONS_FILE,
            output_dir=USER_DATA_DIR,
        )
    
    @step
    def extract_from_text(self) -> None:
        from extract_unstructured.pipeline import run_extraction_batch
        
        # Ensure directories exist
        ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "data").mkdir(parents=True, exist_ok=True)
        
        run_extraction_batch(
            input_dir=ARTICLES_DIR,
            output_file=EXTRACTED_JSON,
            kg_file=FINAL_KG_TTL 
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
                    ?phone a sp:BasePhone ;
                        sp:mainCameraMP ?mp .
                    FILTER(?mp >= 100)
                }
            """),

            ("LargeBatteryPhone", """
                PREFIX sp: <http://example.org/smartphone#>
                CONSTRUCT { ?phone a sp:LargeBatteryPhone }
                WHERE {
                    ?phone a sp:BasePhone ;
                        sp:batteryCapacityMah ?mah .
                    FILTER(?mah >= 5000)
                }
            """),

            ("InMarketPhone", """
                PREFIX sp: <http://example.org/smartphone#>
                CONSTRUCT { ?phone a sp:InMarketPhone }
                WHERE {
                    ?phone a sp:BasePhone .
                    ?config sp:hasBasePhone ?phone .
                    ?offering sp:forConfiguration ?config ;
                              sp:priceValue ?price .
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
    def train_recommendation_model(self) -> None:
        from recommandation.train import train_model
        train_model()

    @step
    def combine_files(self) -> None:
        OUTPUT.mkdir(parents=True, exist_ok=True)
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
        self.process_prices()
        self.generate_configurations()
        self.aggregate_review_sentiments()
        self.gen_user_data()
        self.gen_facts()
        self.extract_from_text()
        self.materialize_by_construct_and_inference()
        self.link()
        self.combine_files()
        #self.train_recommendation_model()


if __name__ == "__main__":
    Pipeline().run()
