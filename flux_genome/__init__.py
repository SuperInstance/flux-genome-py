"""
flux-genome — Genetic Expression Engine for Constraint Theory

The biological analogue: Tensor-Penrose structure is DNA (rigid, fixed).
The Genome encodes ALL possible constraint-checking proteins. Environmental
context (domain, hardware, latency) determines WHICH genes are expressed,
producing different constraint procedures from the SAME underlying genome.
"""

from flux_genome.gene import Gene
from flux_genome.genome import Genome
from flux_genome.ribosome import Ribosome
from flux_genome.expression import ExpressionProfile, Protein, Incubator
from flux_genome.builtins import constraint_genome

__all__ = [
    "Gene",
    "Genome",
    "Ribosome",
    "ExpressionProfile",
    "Protein",
    "Incubator",
    "constraint_genome",
]
