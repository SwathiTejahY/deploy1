{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/SwathiTejahY/deploy1/blob/main/corrected_cocoon_cloud_security_ontology.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "b69d1bac",
      "metadata": {
        "id": "b69d1bac"
      },
      "source": [
        "# **Simplified CoCoOn Cloud Security Ontology**\n",
        "This notebook demonstrates a minimal version of the CoCoOn Cloud Security Ontology for cloud environments using RDFLib in Python. It models relationships between assets, vulnerabilities, threats, and controls."
      ]
    },
    {
      "cell_type": "markdown",
      "id": "c743f11c",
      "metadata": {
        "id": "c743f11c"
      },
      "source": [
        "## **Step 1: Install Required Libraries**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "619d6b8f",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "619d6b8f",
        "outputId": "cbd69ad1-6050-4e3d-d74f-8a61a20eeb2e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: rdflib in /usr/local/lib/python3.10/dist-packages (7.1.1)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (2.2.2)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (3.4.2)\n",
            "Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (3.8.0)\n",
            "Requirement already satisfied: isodate<1.0.0,>=0.7.2 in /usr/local/lib/python3.10/dist-packages (from rdflib) (0.7.2)\n",
            "Requirement already satisfied: pyparsing<4,>=2.1.0 in /usr/local/lib/python3.10/dist-packages (from rdflib) (3.2.0)\n",
            "Requirement already satisfied: numpy>=1.22.4 in /usr/local/lib/python3.10/dist-packages (from pandas) (1.26.4)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.2)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.3.1)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (4.55.3)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.4.7)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (24.2)\n",
            "Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (11.0.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)\n"
          ]
        }
      ],
      "source": [
        "!pip install rdflib pandas networkx matplotlib"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "8814cf4c",
      "metadata": {
        "id": "8814cf4c"
      },
      "source": [
        "## **Step 2: Import Libraries**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "eb906139",
      "metadata": {
        "id": "eb906139"
      },
      "outputs": [],
      "source": [
        "from rdflib import Graph, Namespace, RDF, OWL, URIRef\n",
        "import pandas as pd\n",
        "import networkx as nx\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "9098d304",
      "metadata": {
        "id": "9098d304"
      },
      "source": [
        "## **Step 3: Initialize Ontology and Namespace**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "d4b1e5c2",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "d4b1e5c2",
        "outputId": "b60144ab-ed8e-43b9-c841-0196c09572c0"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Graph identifier=Nf7e546aa63eb463aa6a0fa2a24bce4e3 (<class 'rdflib.graph.Graph'>)>"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ],
      "source": [
        "# Initialize the graph and define namespace\n",
        "g = Graph()\n",
        "COCOON = Namespace(\"http://example.org/cocoon#\")\n",
        "g.bind(\"cocoon\", COCOON)\n",
        "\n",
        "# Define Classes\n",
        "g.add((COCOON.Asset, RDF.type, OWL.Class))\n",
        "g.add((COCOON.Threat, RDF.type, OWL.Class))\n",
        "g.add((COCOON.Vulnerability, RDF.type, OWL.Class))\n",
        "g.add((COCOON.Control, RDF.type, OWL.Class))\n",
        "\n",
        "# Define Relationships\n",
        "g.add((COCOON.hasVulnerability, RDF.type, OWL.ObjectProperty))\n",
        "g.add((COCOON.exploitedBy, RDF.type, OWL.ObjectProperty))\n",
        "g.add((COCOON.mitigatedBy, RDF.type, OWL.ObjectProperty))"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "76577733",
      "metadata": {
        "id": "76577733"
      },
      "source": [
        "## **Step 4: Add Instances and Relationships**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "db7094ee",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "db7094ee",
        "outputId": "2751b310-8a54-431f-e560-bb3e36b8989d"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Graph identifier=Nf7e546aa63eb463aa6a0fa2a24bce4e3 (<class 'rdflib.graph.Graph'>)>"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ],
      "source": [
        "# Add Instances\n",
        "asset1 = URIRef(COCOON.CloudServer)\n",
        "vuln1 = URIRef(COCOON.DataExposure)\n",
        "threat1 = URIRef(COCOON.MalwareAttack)\n",
        "control1 = URIRef(COCOON.Encryption)\n",
        "\n",
        "# Define relationships\n",
        "g.add((asset1, RDF.type, COCOON.Asset))\n",
        "g.add((asset1, COCOON.hasVulnerability, vuln1))\n",
        "g.add((vuln1, RDF.type, COCOON.Vulnerability))\n",
        "g.add((vuln1, COCOON.exploitedBy, threat1))\n",
        "g.add((threat1, RDF.type, COCOON.Threat))\n",
        "g.add((vuln1, COCOON.mitigatedBy, control1))\n",
        "g.add((control1, RDF.type, COCOON.Control))"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "cad6761d",
      "metadata": {
        "id": "cad6761d"
      },
      "source": [
        "## **Step 5: Query the Ontology**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "972a0e09",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "972a0e09",
        "outputId": "cd5cf396-918b-4627-9e6c-a14ca18e6dec"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "--- Query Results ---\n",
            "  Vulnerability   Exploited By\n",
            "0  DataExposure  MalwareAttack\n"
          ]
        }
      ],
      "source": [
        "# Query for vulnerabilities and the threats exploiting them\n",
        "results = []\n",
        "for s, p, o in g.triples((None, COCOON.exploitedBy, None)):\n",
        "    results.append((s.split(\"#\")[-1], o.split(\"#\")[-1]))\n",
        "\n",
        "# Convert to pandas DataFrame\n",
        "df = pd.DataFrame(results, columns=[\"Vulnerability\", \"Exploited By\"])\n",
        "print(\"--- Query Results ---\")\n",
        "print(df)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "bd861c63",
      "metadata": {
        "id": "bd861c63"
      },
      "source": [
        "## **Step 6: Visualize Results as a Graph**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "aa25dd27",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 659
        },
        "id": "aa25dd27",
        "outputId": "c9f1f7ef-a43a-4be9-e4f8-356d96c56c8b"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x600 with 1 Axes>"
            ],
