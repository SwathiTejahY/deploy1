import streamlit as st
from rdflib import Graph, Namespace, RDF, OWL, URIRef
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Streamlit App Title
st.title(" CoCoOn Cloud Security Ontology")

# Initialize the graph and define namespace
g = Graph()
COCOON = Namespace("http://example.org/cocoon#")
g.bind("cocoon", COCOON)

# Define Classes
g.add((COCOON.Asset, RDF.type, OWL.Class))
g.add((COCOON.Threat, RDF.type, OWL.Class))
g.add((COCOON.Vulnerability, RDF.type, OWL.Class))
g.add((COCOON.Control, RDF.type, OWL.Class))

# Define Relationships
g.add((COCOON.hasVulnerability, RDF.type, OWL.ObjectProperty))
g.add((COCOON.exploitedBy, RDF.type, OWL.ObjectProperty))
g.add((COCOON.mitigatedBy, RDF.type, OWL.ObjectProperty))

# Add Instances
asset1 = URIRef(COCOON.CloudServer)
vuln1 = URIRef(COCOON.DataExposure)
threat1 = URIRef(COCOON.MalwareAttack)
control1 = URIRef(COCOON.Encryption)

# Define relationships
g.add((asset1, RDF.type, COCOON.Asset))
g.add((asset1, COCOON.hasVulnerability, vuln1))
g.add((vuln1, RDF.type, COCOON.Vulnerability))
g.add((vuln1, COCOON.exploitedBy, threat1))
g.add((threat1, RDF.type, COCOON.Threat))
g.add((vuln1, COCOON.mitigatedBy, control1))
g.add((control1, RDF.type, COCOON.Control))

# Query the Ontology
st.header("Query Results")
results = []
for s, p, o in g.triples((None, COCOON.exploitedBy, None)):
    results.append((s.split("#")[-1], o.split("#")[-1]))

# Convert to pandas DataFrame
df = pd.DataFrame(results, columns=["Vulnerability", "Exploited By"])
st.write("---")
st.write("### Vulnerabilities and the Threats Exploiting Them")
st.dataframe(df)

# Visualize Results as a Graph
st.header("Ontology Graph Visualization")
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['Vulnerability'], row['Exploited By'], label='exploitedBy')

# Draw graph
fig, ax = plt.subplots(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=12, font_weight="bold", ax=ax)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['label'] for u, v, d in G.edges(data=True)}, ax=ax)
st.pyplot(fig)

# Save Ontology to File
st.header("Save Ontology")
if st.button("Save Ontology as Turtle"):
    output_path = "cocoon_cloud_security_ontology.ttl"
    g.serialize(destination=output_path, format="turtle")
    st.success(f"Ontology saved to: {output_path}")


