import json
from rdflib import Graph, Namespace, RDF, SH, SKOS

# Namespaces based on your provided files
SP = Namespace("http://example.org/smartphone#")
SPV = Namespace("http://example.org/smartphone/vocab/")
SPSH = Namespace("http://example.org/smartphone/shapes/")

class OntologyConfigLoader:
    def __init__(self, shacl_path, skos_path):
        self.shacl_graph = Graph().parse(shacl_path, format="turtle")
        self.skos_graph = Graph().parse(skos_path, format="turtle")
        
        self.vocabularies = {}
        self.uri_mapper = {}

    def get_allowed_vocabularies(self):
        """
        Parses SKOS to build the simplified vocabulary lists for the LLM.
        """
        self.vocabularies = {}
        self.uri_mapper = {}

        # 1. Identify Concept Schemes
        main_scheme = SPV.SmartphoneVocabulary
        top_concepts = list(self.skos_graph.objects(main_scheme, SKOS.hasTopConcept))

        for top_concept in top_concepts:
            # Check if prefLabel exists
            label_node = self.skos_graph.value(top_concept, SKOS.prefLabel)
            if not label_node:
                continue
                
            category_name = str(label_node)
            self.vocabularies[category_name] = []

            # Find narrower concepts
            for concept in self.skos_graph.subjects(SKOS.broader, top_concept):
                c_label = self.skos_graph.value(concept, SKOS.prefLabel)
                if c_label:
                    label_str = str(c_label)
                    self.vocabularies[category_name].append(label_str)
                    self.uri_mapper[label_str] = concept
                    
                    # Map AltLabels
                    for alt in self.skos_graph.objects(concept, SKOS.altLabel):
                        self.uri_mapper[str(alt)] = concept

        # 2. Add Brands Manually
        self.vocabularies["Brand"] = ["Samsung", "Apple", "Google", "Xiaomi", "OnePlus", "Sony"]
        # Basic mapping for brands to ensure URIs exist
        for b in self.vocabularies["Brand"]:
             # Assuming standard naming convention for Brands not in SKOS
             self.uri_mapper[b] = SP[f"Brand_{b}"]

        return self.vocabularies, self.uri_mapper

    def generate_llm_schema(self, root_shape_name="SmartphoneShape"):
        """
        Parses SHACL to generate the JSON schema for the LLM.
        """
        root_shape = SPSH[root_shape_name]
        schema_structure = {}

        # Get all sh:property nodes
        properties = self.shacl_graph.objects(root_shape, SH.property)
        
        for prop in properties:
            path = self.shacl_graph.value(prop, SH.path)
            if not path:
                continue
                
            # sp:phoneName -> phoneName
            field_name = str(path).split("#")[-1] 
            
            datatype = self.shacl_graph.value(prop, SH.datatype)
            
            # FIX: Use SH['class'] instead of SH.class_
            sh_class = self.shacl_graph.value(prop, SH['class'])
            
            desc = "string"
            if datatype:
                desc = str(datatype).split("#")[-1] # e.g. integer, boolean
            elif sh_class:
                # If it links to a Class (like sp:Brand), note it
                cls_name = str(sh_class).split("#")[-1]
                desc = f"Link to {cls_name}"

            schema_structure[field_name] = desc

        return json.dumps(schema_structure, indent=4)