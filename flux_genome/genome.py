"""
Genome — the complete Tensor-Penrose DNA.

FIXED structure — does not change per execution. Contains ALL possible
proteins the system can express. The genome is the rigid Penrose structure
encoded as constraint knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from flux_genome.gene import Gene


@dataclass
class Genome:
    genes: Dict[str, Gene] = field(default_factory=dict)
    regulatory_network: Dict[str, List[str]] = field(default_factory=dict)

    def add_gene(self, gene: Gene) -> None:
        self.genes[gene.gene_id] = gene
        for promoter in gene.promoters:
            if promoter not in self.regulatory_network:
                self.regulatory_network[promoter] = []
            self.regulatory_network[promoter].append(gene.gene_id)

    @property
    def gene_count(self) -> int:
        return len(self.genes)

    @property
    def domains(self) -> Set[str]:
        return {g.domain for g in self.genes.values()}
