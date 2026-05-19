"""
Expression — ExpressionProfile, Protein, and Incubator.

ExpressionProfile: which genes are active in a given environment.
Protein: an assembled constraint-checking protein.
Incubator: the full gene expression pipeline (PLATO).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from flux_genome.genome import Genome
from flux_genome.ribosome import Ribosome


@dataclass
class ExpressionProfile:
    """
    The result of reading the genome in a specific environment.
    Like a cell's transcriptome: which genes are turned on, and how strongly.
    """
    environment: Dict[str, object]
    active_genes: List[str] = field(default_factory=list)
    expression_levels: Dict[str, float] = field(default_factory=dict)
    silenced_genes: List[str] = field(default_factory=list)

    @property
    def strongly_expressed(self) -> List[str]:
        """Genes with expression > 0.7"""
        return [g for g in self.active_genes if self.expression_levels.get(g, 0) > 0.7]

    @property
    def weakly_expressed(self) -> List[str]:
        """Genes with expression 0.3-0.7"""
        return [g for g in self.active_genes
                if 0.3 <= self.expression_levels.get(g, 0) <= 0.7]


@dataclass
class Protein:
    """
    An assembled constraint-checking protein. Like biological proteins,
    these are temporary — they degrade over time unless reinforced.
    """
    protein_id: str
    assembled_from: List[str]  # gene_ids that contributed
    procedure: Callable[[NDArray], NDArray]  # the actual checking function
    lifetime: float = 1.0
    degradation_rate: float = 0.1
    domain: str = "general"

    def execute(self, data: NDArray) -> NDArray:
        """Run the constraint check. Returns error mask (0 = pass)."""
        return self.procedure(data)

    def tick(self) -> None:
        """Advance time — protein degrades."""
        self.lifetime = max(0.0, self.lifetime - self.degradation_rate)

    @property
    def is_alive(self) -> bool:
        return self.lifetime > 0.0


class Incubator:
    """
    The PLATO incubator: the environment where genetic potential becomes
    functional reality. Full gene expression pipeline:

    1. Ribosome reads genome
    2. ExpressionProfile determines active genes
    3. Ribosome translates genes into proteins
    4. Proteins execute constraint checks
    5. Results feed back (mutation/epigenetics)
    """

    def __init__(self, genome: Genome, ribosome: Optional[Ribosome] = None):
        self.genome = genome
        self.ribosome = ribosome or Ribosome()
        self.proteins: List[Protein] = []
        self.history: List[Dict] = []

    def express(self, environment: Dict[str, object],
                data: Optional[NDArray] = None) -> Dict:
        """Full expression pipeline: genome + environment → proteins → results."""
        # 1. Transcription
        profile = self.ribosome.transcript(self.genome, environment)

        # 2. Translation
        proteins = self.ribosome.translate_profile(self.genome, profile)

        # 3. Execution (if data provided)
        results = {}
        if data is not None:
            for protein in proteins:
                error_mask = protein.execute(data)
                results[protein.protein_id] = {
                    "error_mask": error_mask,
                    "violations": int(np.sum(error_mask)),
                    "domain": protein.domain,
                    "alive": protein.is_alive,
                }

        # 4. Store active proteins
        self.proteins = proteins

        # 5. Record history (epigenetics)
        record = {
            "environment": environment,
            "active_genes": profile.active_genes,
            "strongly_expressed": profile.strongly_expressed,
            "silenced": profile.silenced_genes,
            "protein_count": len(proteins),
            "domains": list({p.domain for p in proteins}),
        }
        self.history.append(record)

        return {
            "profile": profile,
            "proteins": proteins,
            "results": results,
        }

    def tick(self) -> None:
        """Advance time — all proteins degrade."""
        for protein in self.proteins:
            protein.tick()
        self.proteins = [p for p in self.proteins if p.is_alive]
