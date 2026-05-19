"""
Gene — a unit of genetic information in Tensor-Penrose DNA.

A gene is a locus on the Tensor-Penrose tiling encoding a constraint
checking procedure. Like a biological gene, it has:
  - A fixed structure (the DNA sequence — Eisenstein lattice point)
  - Expression conditions (promoter/enhancer elements)
  - A protein template (what constraint this gene produces)
  - Regulatory elements (activators and silencers)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
from numpy.typing import NDArray


@dataclass
class Gene:
    gene_id: str
    structure: NDArray  # Eisenstein lattice point / cyclotomic element (fixed)
    expression_conditions: Dict[str, object]  # environment features that activate
    protein_template: Callable[[NDArray], NDArray]  # constraint checking function
    promoters: List[str] = field(default_factory=list)   # gene_ids that activate this
    silencers: List[str] = field(default_factory=list)   # gene_ids that suppress this
    domain: str = "general"  # constraint domain tag
    description: str = ""

    def matches_environment(self, environment: Dict[str, object]) -> float:
        """
        Compute expression strength [0, 1] for given environment.
        Returns 0.0 if no conditions match, 1.0 if all match perfectly.
        """
        if not self.expression_conditions:
            return 0.0

        score = 0.0
        matched = 0
        for key, required in self.expression_conditions.items():
            if key not in environment:
                continue
            env_val = environment[key]
            matched += 1
            if isinstance(required, (list, set, tuple)):
                if env_val in required:
                    score += 1.0
            elif isinstance(required, float) and isinstance(env_val, (int, float)):
                sigma = 0.5
                score += np.exp(-0.5 * ((env_val - required) / sigma) ** 2)
            elif env_val == required:
                score += 1.0
            elif isinstance(required, str) and isinstance(env_val, str):
                if required.lower() in env_val.lower() or env_val.lower() in required.lower():
                    score += 0.5

        if matched == 0:
            return 0.0
        return score / max(len(self.expression_conditions), 1)
