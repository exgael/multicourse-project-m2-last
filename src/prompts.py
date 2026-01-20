graphspec_template = """

```yaml
meta:
  title: <string> # title of the generation task.
  description: <string> # detailed description of the generation task. 
  outputs: <structured|unstructured|mixed>
  
nodes:
  - kind: <Class|Attribute>
    name: <QualifiedName>   # e.g., Person or Person.age
    description: <string>
    datatype:
      base: <string>        # string | integer | float | enum | boolean | text | date
      unit: <string>        # optional
    value_space:            # (required if enum; else: None)
      values: [ ... ]
    
edges:
  - type: <hasAttribute|subclassOf|relatesTo>
    source: <node name>
    target: <node name>
    name: <string>                     # optional, for relatesTo
    multiplicity: { source_to_target: "<..>", target_to_source: "<..>" }  # optional
```
"""

graphspec_example = """
```yaml
meta:
  title: "Hiring"
  description: "Classes & attributes as nodes; structural edges only"
  outputs: mixed

nodes:
  - kind: Class
    name: Person
    description: "An individual"

  - kind: Attribute
    name: Person.age
    description: "Age in years"
    datatype: { base: integer, unit: years }
    
  - kind: Class
    name: Company
    description: "An organization"

  - kind: Attribute
    name: Company.industry
    description: "Industry sector"
    datatype: { base: string }
    
  - kind: Class
    name: Employment
    description: "Link Person and Company"

  - kind: Attribute
    name: Employment.salary
    description: "Annual salary (EUR)"
    datatype: { base: integer, unit: EUR/year }
    
edges:
  - { type: hasAttribute, source: Person, target: Person.age }
  - { type: hasAttribute, source: Company, target: Company.industry }
  - { type: hasAttribute, source: Employment, target: Employment.salary }
  - { type: relatesTo, name: employs, source: Company, target: Person }
```
"""

graphspec_example_2 = """
meta:
  title: "E-commerce"
  description: "Ontology for synthetic e-commerce data generation"
  outputs: mixed

nodes:
  # --- Customer ---
  - kind: Class
    name: Customer
    description: "A customer who makes purchases"

  - kind: Attribute
    name: Customer.id
    description: "Unique identifier of the customer"
    datatype: { base: string }

  - kind: Attribute
    name: Customer.name
    description: "Name of the customer"
    datatype: { base: string }

  - kind: Attribute
    name: Customer.age
    description: "Age of the customer"
    datatype: { base: integer, unit: years }

  - kind: Attribute
    name: Customer.country
    description: "Country of the customer"
    datatype: { base: string }

  # --- Product ---
  - kind: Class
    name: Product
    description: "A product available in the catalog"

  - kind: Attribute
    name: Product.id
    description: "Unique identifier of the product"
    datatype: { base: string }

  - kind: Attribute
    name: Product.name
    description: "Name of the product"
    datatype: { base: string }

  - kind: Attribute
    name: Product.category
    description: "Category of the product"
    datatype: { base: string }

  - kind: Attribute
    name: Product.price
    description: "Price of the product in USD"
    datatype: { base: float, unit: USD }

  - kind: Attribute
    name: Product.release_year
    description: "Year the product was released"
    datatype: { base: integer }

  # --- Order ---
  - kind: Class
    name: Order
    description: "A purchase made by a customer"

  - kind: Attribute
    name: Order.id
    description: "Unique identifier of the order"
    datatype: { base: string }

  - kind: Attribute
    name: Order.quantity
    description: "Quantity of products in the order"
    datatype: { base: integer }

  - kind: Attribute
    name: Order.total_amount
    description: "Total order value in USD"
    datatype: { base: float, unit: USD }

  # --- Review ---
  - kind: Class
    name: Review
    description: "A review written by a customer about a product"

  - kind: Attribute
    name: Review.rating
    description: "Numeric rating between 1 and 5"
    datatype: { base: integer }

  - kind: Attribute
    name: Review.text
    description: "Short text review"
    datatype: { base: text }

edges:
  # Customer edges
  - { type: hasAttribute, source: Customer, target: Customer.id }
  - { type: hasAttribute, source: Customer, target: Customer.name }
  - { type: hasAttribute, source: Customer, target: Customer.age }
  - { type: hasAttribute, source: Customer, target: Customer.country }

  # Product edges
  - { type: hasAttribute, source: Product, target: Product.id }
  - { type: hasAttribute, source: Product, target: Product.name }
  - { type: hasAttribute, source: Product, target: Product.category }
  - { type: hasAttribute, source: Product, target: Product.price }
  - { type: hasAttribute, source: Product, target: Product.release_year }

  # Order edges
  - { type: hasAttribute, source: Order, target: Order.id }
  - { type: hasAttribute, source: Order, target: Order.quantity }
  - { type: hasAttribute, source: Order, target: Order.total_amount }

  # Review edges
  - { type: hasAttribute, source: Review, target: Review.rating }
  - { type: hasAttribute, source: Review, target: Review.text }

  # Relational links
  - { type: relatesTo, name: places,  source: Customer, target: Order }
  - { type: relatesTo, name: contains, source: Order, target: Product }
  - { type: relatesTo, name: reviews, source: Customer, target: Review }
  - { type: relatesTo, name: about,   source: Review, target: Product }

"""

schema_gen_instruct = f"""
You are an ontology schema graph generator. 
Your single job is to convert a short domain specification (either a DB-like schema or a textual brief) into a valid GraphSpec YAML document that will be consumed by an automated pipeline.

Hard rules (must follow exactly):

- Output ONLY valid YAML that conforms to GraphSpec (structure provided below). Do not add any prose, explanation, or metadata outside the YAML. No JSON, no markdown, no comments.

- Node names must be unique and use qualified attribute names for attributes: ClassName.attrName (e.g., Person.age). Class nodes use ClassName (e.g., Person).

- Use kind: Class or kind: Attribute. Every Attribute node must include an of_class property matching an existing Class node.

- If an attribute domain is open-ended, its datatype must be string (to allow multiple instantiations of the values in later steps). Choose enum as a datatype only if the attribute domain is small (2 or 3) and close ended (gender, marital status, ...).

- For attributes of enumerated nature, include value_space with kind: enum and values: [...].

- Even if it is mentionned in the domain specification, do not create any attribute for the id. 

- If not specifically mentionned, prefer string types over enum, unless enum is the best for a specific category. 

- meta must include title, description, and outputs (one of structured, unstructured, mixed).

- Validate that all hasAttribute edges point from a Class to an Attribute and that Attribute.of_class equals the edge source when present.

- Do not invent external ontologies or URIs. Keep entries self-contained.

- Keep the YAML compact (no long narrative text). Descriptions should be 1–2 short sentences.

- Output format: follow the GraphSpec example block shown in the User message below (the exact key names and structure). Your YAML must parse to that schema.

- Attribute attachment rule: Every Attribute node must have exactly one incoming hasAttribute edge from a Class.
→ That edge is the single source of truth for “which class owns this attribute”.

- Enum rule: If datatype.base: enum, then value_space.values must be a non-empty list. Otherwise, value_space must be omitted (not an empty object).

- Multiplicity defaults: If a relatesTo edge has no multiplicity, default to: source_to_target: "0..n", target_to_source: "0..1".

- Name uniqueness: nodes[*].name must be unique across the graph.

- Type whitelist: datatype.base ∈ {{string, integer, float, enum, boolean, text, date}}.

Generate the ontology schema graph based given this template: 
{graphspec_template}

Here is an example of an ontology graph schema:
{graphspec_example}

"""

dependency_gen = f"""
You are DependencyEdgeFinder.

GOAL
Given a GraphSpec YAML with Class and Attribute nodes and only structural edges
(hasAttribute | subclassOf | relatesTo), infer a concise set of directed **Attribute→Attribute**
dependencies that will guide conditional generation.

OUTPUT FORMAT
Return ONLY a valid JSON array (no wrapper object, no comments, no prose):
```json
[
  {{ "source": "<AttributeName>", "target": "<AttributeName>" }},
  ...
]
```

DEFINITION (what a dependency means)
A directed edge S → T between Attribute nodes means:
1) ORDER: T must be generated AFTER S.
2) CONDITIONING: When generating values for T, the model must CONDITION ON the realized value of S.
3) SCOPE: Conditioning may be intra-class (same instance) or inter-class (across instances that are joinable via existing structural edges such as relatesTo or relation classes).
4) DAG: The dependency graph MUST be acyclic. If you detect a cycle, remove the weakest/least-informative edge.

STRICT RULES
- Only Attribute names from the input graph are allowed in source/target (never Classes).
- No duplicates. No self-loops. Keep edges minimal but sufficient (avoid redundant chains).
- Prefer fine-grained, high-signal dependencies (attribute→attribute) over coarse proxies.
- Add an edge ONLY if changing S would materially change the distribution or content of T.

DECISION GUIDELINES
A) LATENT → SURFACE (tie-breaker & cycle-avoidance):
   If one attribute is a latent/category/identity and the other is a realized text/numeric/derived field,
   direct the edge latent → surface. Examples: category → name; role/status → salary/price/risk; country → currency/language.
B) PROBABILISTIC conditioning:
   If T’s distribution depends on S (e.g., price | category, rating | user_age, region), add S → T.
C) DETERMINISTIC/DERIVED:
   If T is computed/derived from S (e.g., total_amount = price × quantity), add parent attributes → T.
D) TEXT SLOTS:
   If a text attribute T incorporates or paraphrases S (e.g., short_description uses product.name and category), add S → T.
E) JOIN-AWARE:
   Inter-class dependencies are allowed only when S can be routed to T via existing structural links.

QUALITY FILTER
- Favor edges that prevent incoherent samples (e.g., category↔name mismatches; totals not matching price×quantity).
- Omit decorative or weak signals (e.g., username → review_text).

OUTPUT POLICY
- Sort edges by target, then source for stable diffs.
- Return ONLY the JSON array; nothing else.

Here is an example of an ontology schema:
{graphspec_example_2}

Here is the expected output in JSON: 
```json
[
  {{ "source": "Product.category",     "target": "Product.name" }},
  {{ "source": "Product.category",     "target": "Product.price" }},
  {{ "source": "Product.release_year", "target": "Product.price" }},

  {{ "source": "Product.price",        "target": "Order.total_amount" }},
  {{ "source": "Order.quantity",       "target": "Order.total_amount" }},

  {{ "source": "Customer.age",         "target": "Review.rating" }},
  {{ "source": "Product.category",     "target": "Review.rating" }},

  {{ "source": "Review.rating",        "target": "Review.text" }},
  {{ "source": "Product.name",         "target": "Review.text" }},
  {{ "source": "Product.category",     "target": "Review.text" }}
]

```

After the outputted json, explain how each attribute depends on the other. 
"""

dist_spec_gen = """
You are DistSpecGenerator-MVP.

GOAL
Given:
- a target attribute (numeric, enum, or date),
- a small set of conditioning attributes (its predecessors from a dependency graph),
- and one or more rows of realized conditioning values ("contexts"),

produce a parsable distribution specification ("DistSpec") that describes how to sample values for the target, conditioned on those contexts.

DISTSPEC FORMAT
Output ONLY valid JSON that conforms to these shapes (no prose, no comments):

Numeric:
  {"$dist":"uniform","low":a,"high":b}
  {"$dist":"normal","mu":m,"sigma":s,"truncate":[a,b]}
  {"$dist":"lognormal","mu":m,"sigma":s,"truncate":[a,b]}
  {"$dist":"triangular","low":a,"mode":c,"high":b}
  {"$dist":"poisson","lam":λ}
  {"$dist":"histogram","bins":[...],"probs":[...]} // list of bins must have length k+1 meanwhile list of probabilities, must have length k
  {"$dist":"mixture","components":[{"w":w1,"dist":{...}}, {"w":w2,"dist":{...}}]}

Categorical (enums):
  {"$dist":"categorical","values":[...],"probs":[...]}

Dates (ISO YYYY-MM-DD):
  {"$dist":"uniform_date","start":"YYYY-MM-DD","end":"YYYY-MM-DD"}
  {"$dist":"normal_date","mean":"YYYY-MM-DD","sigma_days":N,"bounds":["YYYY-MM-DD","YYYY-MM-DD"]}

REQUIREMENTS
1) Your output must be **JSON only** and must match one of the DistSpec shapes above.
2) **Boundaries**: include realistic truncation/bounds when applicable (e.g., ages 18–80, salary 25k–220k).
3) **Conditioning**: the parameters should reflect the provided conditioning context(s). Do not ignore conditioning attributes.
4) **Stability**: avoid degenerate or near-delta distributions; prefer mixtures or wider sigmas over collapsing to a constant.
5) **Categorical probs** must be ≥0 and sum to >0 (they may be normalized by the sampler).
6) Keep numeric parameters in reasonable units (e.g., salary in EUR/year).
7) Do NOT emit raw samples; emit the **distribution** only.

DECISION HEURISTICS
- NUMERIC: use `lognormal` for right-skewed money-like amounts (salary, price), `normal` or `mixture` for symmetric quantities, `triangular` for bounded expert ranges, `histogram` when a piecewise shape is natural.
- ENUM: use `categorical` with probabilities conditioned on the context.
- DATE: use `uniform_date` for flat ranges; `normal_date` with bounds when events cluster around a mean date.
- When uncertain between parametric families, prefer a **mixture** of simple components over a single brittle one.

### Example:

#### Input: 
```json
{
  "task": "dist_for_target_attribute",
  "target": {
    "name": "Employment.salary",
    "datatype": "integer",
    "unit": "EUR/year",
  },
  "conditioning_attributes": [
    "Employment.title",
    "Company.industry",
    "Person.country"
  ],
  "context": {
    "Employment.title": "Engineer",
    "Company.industry": "SaaS",
    "Person.country": "FR"
  },
  "output": "DistSpecJSON"
}
```

#### Expected Output:
```json
{ "$dist": "lognormal", "mu": 10.9, "sigma": 0.35, "truncate": [28000, 160000] }
``` 
"""

string_list_gen = """ 
You are StringListGenerator.

GOAL:
Generate up to N string values for a target attribute, conditioned on given predecessor attributes (context).
Values must be coherent with the context. If the context is incoherent or impossible, return an EMPTY LIST and mention the incoherence.
If the context is coherent, the values you should generated must be very customized to the fiven context. (Do not give general values).
OUTPUT
Return ONLY a valid JSON array of strings. No wrapper object, no prose, no comments.

CONSTRAINTS
1) Length: return many items if multiplicity is set to "many" and return only one element if multiplicity is set to "one".
2) Uniqueness: deduplicate; no near-duplicates differing only by casing/punctuation.
3) Coherence: use the provided context (conditioning attributes) to determine what is valid.
   - If the target cannot reasonably exist under the given context (e.g., job title irrelevant to industry), output [].
4) Style & validity:
   - Respect any provided style hints (case, max_length, allowed charset).
   - No placeholders like "Lorem ipsum", no personal data, no offensive content.
5) Diversity: cover plausible subtypes/variants where applicable (avoid mode collapse).
6) Formatting: each item must be a single-line string (no newlines, no trailing/leading spaces).
7) Output format: the output must be formatted between ```json``` tags.
8) Try to produce as much as relevant data as you can when multiplicity is set to "many".
9) VERY IMPORTANT: the generated values should be very specific to the conditionning attributes and values (in context). Do not output generic data that are not influenced by the conditionning attributes and values.
DECISION TEST (before output):
- If any required conditioning value is missing or clearly out-of-domain for the ontology hints, output [].
- Else, produce a coherent, diverse set (≤ max_items = 50) following the style rules.

EXAMPLES OF TARGETS
- Employment.title (enum-like textual labels)
- Product.category_name
- Person.first_name (if locale is fixed)
- Short tags/keywords for a given topic

Remember: OUTPUT ONLY A JSON ARRAY OF STRINGS.

### Example

#### Input Example: 
```json
{
  "task": "string_values_for_target",
  "target": {
    "name": "Employment.title",
    "description": "Job titles used on contracts",
  },
  "conditioning_attributes": [
    "Company.industry",
    "Person.country"
  ],
  "context": {
    "Company.industry": "SaaS",
    "Person.country": "FR"
  },
  "multiplicity": "many"
}
``` 
#### Expected Output: 
```json
[
  "Software Engineer",
  "Backend Engineer",
  "Frontend Engineer",
  "Full-Stack Developer",
  "Site Reliability Engineer",
  "Data Engineer",
  "ML Engineer",
  "DevOps Engineer",
  "Product Manager",
  "Technical Program Manager",
  "QA Engineer",
  "Security Engineer",
  "UX Designer",
  "UI Designer",
  "Data Analyst",
  "Business Analyst",
  "Solutions Architect",
  "Cloud Architect",
  "Support Engineer",
  "Implementation Consultant"
]
``` 

If the context were incoherent (e.g., {"Company.industry":"Dairy Farming","Person.country":"FR"} but your ontology/business rules say the target Employment.title for a software product company is required), the model should return:
```json
[]
```

"""

context_cardinality_gen = """
You are CardinalityPlanner-MVP.

TASK
Given:
1) A GraphSpec MVP (nodes + structural edges only), and
2) A list of attributes with their proposed context_key (the attributes whose realized values are required to generate the target),

produce, for each Attribute node, a generation plan that specifies ONLY the multiplicity per context:
```json
{
  "attribute": "<AttributeName>",
  "generation": {
    "context_key": [ <AttributeName>, ... ],
    "multiplicity": { "mode": "<one|many|fixed_k>", "k": <int?> }
  }
}
```
OUTPUT
Return ONLY a JSON array of such objects (no wrapper, no prose).

SEMANTICS
- context_key (given): the attributes whose values will be available and define the conditioning bucket for generating the target.
- multiplicity.mode:
  - one     → exactly one value per distinct context_key combination.
  - many    → an open-ended list (≥1) per context (the executor decides how many).
  - fixed_k → exactly k values per context (you MUST provide a positive integer k).

QUALITY RULES
1) Attribute-only: Refer ONLY to Attribute node names from the provided graph.
2) Respect the given context_key verbatim; do not add/remove keys. (Your job is to choose multiplicity given that context.)
3) Determinism vs catalogs:
   - Use mode=one for attributes that are conceptually single-valued under the provided context (e.g., a canonical description given a unique title).
   - Use mode=many for catalog/label/alias lists where multiple distinct values naturally exist per context (e.g., product titles per category, names per country).
   - Use mode=fixed_k when domain practice fixes the count (e.g., exactly 5 tags per category, exactly 3 images per product).
4) Be specific, not maximalist: prefer one when the attribute represents a canonical field; avoid many unless multiplicity is meaningful for downstream sampling.
5) Cross-entity contexts: if the context_key includes attributes from different classes (e.g., Customer.country + Product.category), treat the tuple as a join bucket; multiplicity still follows the same rules.
6) No circular logic: the target attribute must not appear in its own context_key.
7) Prefer "many" multiplicity unless it is not coherent with the given attributes. 
8) Output must be valid JSON; no comments, no trailing commas, no extra keys.

FORMAT
```json
[
  { "attribute":"...", "generation":{ "context_key":[...], "multiplicity":{"mode":"one"} } },
  { "attribute":"...", "generation":{ "context_key":[...], "multiplicity":{"mode":"fixed_k","k":5} } },
  ...
]
```
## Example
### Input:
graphspec: 
```yaml
version: 2
meta: { title: "E-commerce", description: "Products, customers, and catalog", outputs: "mixed" }
nodes:
  - { kind: Class, name: Product,  description: "Sellable item" }
  - { kind: Attribute, name: Product.category,    description: "Top-level category", datatype: { base: enum }, value_space: { values: [Electronics, Home & Kitchen, Books, Clothing] } }
  - { kind: Attribute, name: Product.title,       description: "Marketing title",    datatype: { base: string } }
  - { kind: Attribute, name: Product.description, description: "Short description",  datatype: { base: text } }
  - { kind: Attribute, name: Product.tags,        description: "Keyword tags",       datatype: { base: string } }

  - { kind: Class, name: Customer, description: "Buyer" }
  - { kind: Attribute, name: Customer.country, description: "Country code", datatype: { base: enum }, value_space: { values: [US, DE, FR, SG] } }
  - { kind: Attribute, name: Customer.name,    description: "Full name",    datatype: { base: string } }

edges:
  - { type: hasAttribute, source: Product,  target: Product.category }
  - { type: hasAttribute, source: Product,  target: Product.title }
  - { type: hasAttribute, source: Product,  target: Product.description }
  - { type: hasAttribute, source: Product,  target: Product.tags }
  - { type: hasAttribute, source: Customer, target: Customer.country }
  - { type: hasAttribute, source: Customer, target: Customer.name }
```
context keys:
```json
[
    { "attribute": "Product.title",       "context_key": ["Product.category"] },
    { "attribute": "Product.description", "context_key": ["Product.title"] },
    { "attribute": "Product.tags",        "context_key": ["Product.category"] },
    { "attribute": "Customer.name",       "context_key": ["Customer.country"] }
]

```

### Expected Output: 
```json
[
  {
    "attribute": "Product.title",
    "generation": {
      "context_key": ["Product.category"],
      "multiplicity": { "mode": "many" }
    }
  },
  {
    "attribute": "Product.description",
    "generation": {
      "context_key": ["Product.title"],
      "multiplicity": { "mode": "one" }
    }
  },
  {
    "attribute": "Product.tags",
    "generation": {
      "context_key": ["Product.category"],
      "multiplicity": { "mode": "fixed_k", "k": 5 }
    }
  },
  {
    "attribute": "Customer.name",
    "generation": {
      "context_key": ["Customer.country"],
      "multiplicity": { "mode": "many" }
    }
  }
]
```
"""