# Cell start
!pip install rdflib pandas networkx matplotlib
# Cell end



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



print("Ontology initialized with instances and relationships.")
# Cell end

# Cell start
# Query Results

results = []

for s, p, o in g.triples((None, CYBERSEC.exploitedBy, None)):

    results.append((s.split("#")[-1], o.split("#")[-1]))



# Convert results to pandas DataFrame

df = pd.DataFrame(results, columns=["Subject", "Exploited By"])

print("Query Results:")

print(df)
# Cell end

# Cell start
# Visualize the relationships using NetworkX

G = nx.DiGraph()

for _, row in df.iterrows():

    G.add_edge(row["Subject"], row["Exploited By"], label="exploitedBy")



# Draw the graph

plt.figure(figsize=(8, 6))

pos = nx.spring_layout(G)

nx.draw(G, pos, with_labels=True, node_size=2000, font_size=12, font_weight="bold")

nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['label'] for u, v, d in G.edges(data=True)})

plt.title("Simplified Cybersecurity Ontology Visualization")

plt.show()
# Cell end


