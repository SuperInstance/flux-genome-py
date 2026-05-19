"""
Ribosome — reads DNA and assembles proteins.

The ribosome IS the sheaf — it maps local genetic information (tiles/genes)
to global protein function (constraint checks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from flux_genome.gene import Gene
from flux_genome.genome import Genome

if TYPE_CHECKING:
    from flux_genome.expression import ExpressionProfile, Protein


class Ribosome:
    EXPRESSION_THRESHOLD = 0.3

    def transcript(self, genome: Genome, environment: Dict[str, object]) -> "ExpressionProfile":
        """Transcription: scan genome for genes matching the environment."""
        from flux_genome.expression import ExpressionProfile
        profile = ExpressionProfile(environment=environment)

        # Phase 1: Direct matching
        for gene_id, gene in genome.genes.items():
            level = gene.matches_environment(environment)
            if level >= self.EXPRESSION_THRESHOLD:
                profile.active_genes.append(gene_id)
                profile.expression_levels[gene_id] = level

        # Phase 2: Regulatory network — promoters enhance, silencers suppress
        enhanced = []
        silenced = []

        for active_id in list(profile.active_genes):
            gene = genome.genes[active_id]
            for promoted_id in gene.promoters:
                if promoted_id in genome.genes and promoted_id not in profile.active_genes:
                    promoted_gene = genome.genes[promoted_id]
                    level = promoted_gene.matches_environment(environment) * 0.6
                    if level >= self.EXPRESSION_THRESHOLD:
                        enhanced.append((promoted_id, level))

            for silenced_id in gene.silencers:
                if silenced_id in profile.active_genes:
                    silenced.append(silenced_id)

        for gene_id, level in enhanced:
            if gene_id not in profile.active_genes:
                profile.active_genes.append(gene_id)
                profile.expression_levels[gene_id] = level

        for gene_id in silenced:
            if gene_id in profile.active_genes:
                profile.active_genes.remove(gene_id)
                profile.silenced_genes.append(gene_id)
                profile.expression_levels.pop(gene_id, None)

        return profile

    def translate(self, gene: Gene, expression_level: float,
                  environment: Dict[str, object]) -> "Protein":
        """Translation: convert a single gene into an executable protein."""
        from flux_genome.expression import Protein
        lifetime = 1.0 * expression_level
        degradation = 0.1 * (1.1 - expression_level)

        return Protein(
            protein_id=f"protein_{gene.gene_id}",
            assembled_from=[gene.gene_id],
            procedure=gene.protein_template,
            lifetime=lifetime,
            degradation_rate=degradation,
            domain=gene.domain,
        )

    def translate_profile(self, genome: Genome,
                          profile: "ExpressionProfile") -> List["Protein"]:
        """Translate all active genes in a profile into proteins."""
        proteins = []
        for gene_id in profile.active_genes:
            gene = genome.genes[gene_id]
            level = profile.expression_levels.get(gene_id, 0.5)
            protein = self.translate(gene, level, profile.environment)
            proteins.append(protein)
        return proteins
