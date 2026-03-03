import json
from collections import defaultdict
import nltk
from nltk.corpus import treebank
from nltk import Tree
nltk.download("treebank")

def binarize(tree):
    """Convert tree to CNF (binarize + remove unary chains later)."""
    tree.chomsky_normal_form(horzMarkov=2)
    return tree

rule_counts = defaultdict(int)
lhs_counts = defaultdict(int)

# Iterate through Penn Treebank trees
for tree in treebank.parsed_sents():
    tree = binarize(tree)

    for prod in tree.productions():
        lhs = str(prod.lhs())
        rhs = [str(sym) for sym in prod.rhs()]

        rule_counts[(lhs, tuple(rhs))] += 1
        lhs_counts[lhs] += 1

# Convert counts → probabilities
grammar = []
for (lhs, rhs), count in rule_counts.items():
    prob = count / lhs_counts[lhs]
    grammar.append({
        "lhs": lhs,
        "rhs": list(rhs),
        "prob": prob
    })

print(f"Exporting {len(grammar)} grammar rules")

with open("grammar.json", "w") as f:
    json.dump(grammar, f, indent=2)