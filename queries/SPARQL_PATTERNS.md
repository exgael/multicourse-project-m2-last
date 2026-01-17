# SPARQL Query Patterns

```sparql
# Count positive [FEATURE] sentiment
OPTIONAL {
  SELECT ?phone (COUNT(?review) AS ?posCount)
  WHERE {
    ?review sp:reviewOf ?phone ;
            sp:hasPositiveSentimentTag spv:[FEATURE] .
  }
  GROUP BY ?phone
}

# Count negative [FEATURE] sentiment
OPTIONAL {
  SELECT ?phone (COUNT(?review) AS ?negCount)
  WHERE {
    ?review sp:reviewOf ?phone ;
            sp:hasNegativeSentimentTag spv:[FEATURE] .
  }
  GROUP BY ?phone
}
```

### Usage in SELECT

Always use `COALESCE` to handle phones without reviews:

```sparql
SELECT ?phoneName
       (COALESCE(?posCount, 0) AS ?positive[FEATURE])
       (COALESCE(?negCount, 0) AS ?negative[FEATURE])
       ((COALESCE(?posCount, 0) - COALESCE(?negCount, 0)) AS ?sentimentScore)
WHERE {
  ?phone sp:phoneName ?phoneName .

  # Insert OPTIONAL blocks here

}
```

### Available Features

Replace `[FEATURE]` with any of these SKOS concepts:
- `Camera`
- `Display`
- `Battery`
- `Performance`
- `Storage`
- `Build`
- `OS`
- `Gaming`
- `Value`
- `Overall`