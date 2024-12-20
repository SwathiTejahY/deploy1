import streamlit as st
from rdflib import Graph, Namespace, RDF, OWL, URIRef
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO

# Initialize Ontology and Namespace
g = Graph()
CYBERSEC = Namespace("http://example.org/cybersecurity#")
g.bind("cybersec", CYBERSEC)

# Define Classes and Property
g.add((CYBERSEC.Asset, RDF.type, OWL.Class))
g.add((CYBERSEC.Threat, RDF.type, OWL.Class))
g.add((CYBERSEC.Vulnerability, RDF.type, OWL.Class))
g.add((CYBERSEC.exploitedBy, RDF.type, OWL.ObjectProperty))

# Add Instances
asset = URIRef(CYBERSEC.Server)
vulnerability = URIRef(CYBERSEC.SQLInjection)
threat = URIRef(CYBERSEC.HackerAttack)

# Add Relationships
g.add((asset, RDF.type, CYBERSEC.Asset))
g.add((vulnerability, RDF.type, CYBERSEC.Vulnerability))
g.add((threat, RDF.type, CYBERSEC.Threat))
g.add((vulnerability, CYBERSEC.exploitedBy, threat))
g.add((asset, CYBERSEC.exploitedBy, vulnerability))

# Metrics
metrics = {
    "Class": [0, 1],
    "Precision": [0.58, 0.97],
    "Recall": [1.00, 0.08],
    "F1-Score": [0.73, 0.14],
    "Support": [2532, 1977],
}
accuracy = 0.595

# Streamlit UI
st.title("Takeshi Takahashi Cloud Security Ontology Visualization")

st.write("### Ontology initialized with instances and relationships.")

# Query Results
results = []
for s, p, o in g.triples((None, CYBERSEC.exploitedBy, None)):
    results.append((s.split("#")[-1], o.split("#")[-1]))

df = pd.DataFrame(results, columns=["Subject", "Exploited By"])
st.write("### Query Results")
st.dataframe(df)

# Display Accuracy
if st.checkbox("Show Model Accuracy"):
    st.write(f"Overall Accuracy: {accuracy:.3f} ({accuracy * 100:.2f}%)")

# Display Detailed Metrics
if st.checkbox("Show Detailed Metrics"):
    metrics_df = pd.DataFrame(metrics)
    st.write("### Detailed Metrics")
    st.dataframe(metrics_df)
    st.write("Macro Avg: Precision: 0.78, Recall: 0.54, F1-Score: 0.44")
    st.write("Weighted Avg: Precision: 0.75, Recall: 0.59, F1-Score: 0.48")

# Visualize the relationships using NetworkX
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row["Subject"], row["Exploited By"], label="exploitedBy")

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=12, font_weight="bold")
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['label'] for u, v, d in G.edges(data=True)})
plt.title("Takeshi Takahashi Cloud Security Ontology Visualization")

# Render the matplotlib figure in Streamlit
buf = BytesIO()
plt.savefig(buf, format="png")
st.image(buf)


     
