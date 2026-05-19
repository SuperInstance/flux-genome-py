"""
Tests for flux-genome — Genetic Expression Engine
"""

import numpy as np
import pytest

from flux_genome import (
    Gene, Genome, ExpressionProfile, Protein, Ribosome, Incubator,
    constraint_genome,
)
from flux_genome.builtins import (
    make_range_check, make_variance_check, make_threshold_check,
    make_monotonic_check, make_emission_check,
)


# ---------------------------------------------------------------------------
# Gene tests
# ---------------------------------------------------------------------------

class TestGene:
    def test_exact_match(self):
        gene = Gene(
            gene_id="test",
            structure=np.array([1.0, 0.0]),
            expression_conditions={"domain": "maritime"},
            protein_template=lambda d: np.zeros(d.shape[0]),
        )
        assert gene.matches_environment({"domain": "maritime"}) == 1.0

    def test_no_match(self):
        gene = Gene(
            gene_id="test",
            structure=np.array([1.0]),
            expression_conditions={"domain": "maritime"},
            protein_template=lambda d: np.zeros(d.shape[0]),
        )
        assert gene.matches_environment({"domain": "medical"}) == 0.0

    def test_partial_match(self):
        gene = Gene(
            gene_id="test",
            structure=np.array([1.0]),
            expression_conditions={"domain": "maritime", "realtime": True},
            protein_template=lambda d: np.zeros(d.shape[0]),
        )
        level = gene.matches_environment({"domain": "maritime"})
        assert 0 < level < 1.0

    def test_list_condition(self):
        gene = Gene(
            gene_id="test",
            structure=np.array([1.0]),
            expression_conditions={"domain": ["maritime", "aerospace"]},
            protein_template=lambda d: np.zeros(d.shape[0]),
        )
        assert gene.matches_environment({"domain": "aerospace"}) == 1.0
        assert gene.matches_environment({"domain": "medical"}) == 0.0

    def test_empty_conditions(self):
        gene = Gene(
            gene_id="test",
            structure=np.array([1.0]),
            expression_conditions={},
            protein_template=lambda d: np.zeros(d.shape[0]),
        )
        assert gene.matches_environment({"domain": "anything"}) == 0.0

    def test_missing_key_in_env(self):
        gene = Gene(
            gene_id="test",
            structure=np.array([1.0]),
            expression_conditions={"domain": "maritime", "safety": True},
            protein_template=lambda d: np.zeros(d.shape[0]),
        )
        level = gene.matches_environment({"domain": "maritime"})
        assert 0 < level < 1.0


# ---------------------------------------------------------------------------
# Genome tests
# ---------------------------------------------------------------------------

class TestGenome:
    def test_add_gene(self):
        genome = Genome()
        gene = Gene(
            gene_id="g1",
            structure=np.array([1.0]),
            expression_conditions={"domain": "test"},
            protein_template=lambda d: np.zeros(d.shape[0]),
            promoters=["master"],
        )
        genome.add_gene(gene)
        assert "g1" in genome.genes
        assert genome.gene_count == 1
        assert "master" in genome.regulatory_network
        assert "g1" in genome.regulatory_network["master"]

    def test_domains(self):
        genome = Genome()
        for i, domain in enumerate(["maritime", "medical", "aerospace"]):
            genome.add_gene(Gene(
                gene_id=f"g{i}",
                structure=np.array([float(i)]),
                expression_conditions={"domain": domain},
                protein_template=lambda d: np.zeros(d.shape[0]),
                domain=domain,
            ))
        assert genome.domains == {"maritime", "medical", "aerospace"}

    def test_constraint_genome_has_25_genes(self):
        g = constraint_genome()
        assert g.gene_count == 25

    def test_constraint_genome_five_domains(self):
        g = constraint_genome()
        assert g.domains == {"maritime", "medical", "automotive", "aerospace", "industrial"}


# ---------------------------------------------------------------------------
# Protein tests
# ---------------------------------------------------------------------------

class TestProtein:
    def test_execute(self):
        check = lambda d: (np.mean(d, axis=1) > 0).astype(np.float64)
        protein = Protein(
            protein_id="test_protein",
            assembled_from=["g1"],
            procedure=check,
        )
        data = np.array([[1.0, 2.0], [-1.0, -2.0]])
        result = protein.execute(data)
        assert result[0] == 1.0
        assert result[1] == 0.0

    def test_degradation(self):
        protein = Protein(
            protein_id="test",
            assembled_from=["g1"],
            procedure=lambda d: np.zeros(d.shape[0]),
            lifetime=1.0,
            degradation_rate=0.3,
        )
        assert protein.is_alive
        protein.tick()
        assert protein.lifetime == pytest.approx(0.7)
        assert protein.is_alive
        for _ in range(4):
            protein.tick()
        assert not protein.is_alive


# ---------------------------------------------------------------------------
# Ribosome tests
# ---------------------------------------------------------------------------

class TestRibosome:
    def _make_genome(self):
        genome = Genome()
        genome.add_gene(Gene(
            gene_id="maritime_1",
            structure=np.array([1.0]),
            expression_conditions={"domain": "maritime"},
            protein_template=lambda d: np.zeros(d.shape[0]),
            domain="maritime",
        ))
        genome.add_gene(Gene(
            gene_id="medical_1",
            structure=np.array([2.0]),
            expression_conditions={"domain": "medical"},
            protein_template=lambda d: np.zeros(d.shape[0]),
            domain="medical",
        ))
        return genome

    def test_transcription_selects_correct_genes(self):
        ribosome = Ribosome()
        genome = self._make_genome()
        profile = ribosome.transcript(genome, {"domain": "maritime"})
        assert "maritime_1" in profile.active_genes
        assert "medical_1" not in profile.active_genes

    def test_translation_produces_proteins(self):
        ribosome = Ribosome()
        genome = self._make_genome()
        profile = ribosome.transcript(genome, {"domain": "maritime"})
        proteins = ribosome.translate_profile(genome, profile)
        assert len(proteins) >= 1
        assert proteins[0].domain == "maritime"

    def test_promoter_activation(self):
        genome = Genome()
        genome.add_gene(Gene(
            gene_id="master",
            structure=np.array([1.0]),
            expression_conditions={"domain": "any"},
            protein_template=lambda d: np.zeros(d.shape[0]),
            promoters=["slave"],
        ))
        genome.add_gene(Gene(
            gene_id="slave",
            structure=np.array([2.0]),
            expression_conditions={"domain": "other"},
            protein_template=lambda d: np.zeros(d.shape[0]),
        ))
        ribosome = Ribosome()
        profile = ribosome.transcript(genome, {"domain": "any"})
        assert "master" in profile.active_genes


# ---------------------------------------------------------------------------
# Incubator tests
# ---------------------------------------------------------------------------

class TestIncubator:
    def test_full_pipeline(self):
        genome = constraint_genome()
        incubator = Incubator(genome)
        data = np.random.default_rng(0).normal(0, 1, (5, 4))

        result = incubator.express({"domain": "maritime"}, data)
        assert result["profile"] is not None
        assert len(result["proteins"]) > 0
        assert len(result["results"]) > 0

    def test_tick_degradation(self):
        genome = Genome()
        genome.add_gene(Gene(
            gene_id="g1",
            structure=np.array([1.0]),
            expression_conditions={"domain": "test"},
            protein_template=lambda d: np.zeros(d.shape[0]),
        ))
        incubator = Incubator(genome)
        incubator.express({"domain": "test"})
        for _ in range(100):
            incubator.tick()
        assert len(incubator.proteins) == 0

    def test_history_recorded(self):
        genome = constraint_genome()
        incubator = Incubator(genome)
        incubator.express({"domain": "maritime"})
        incubator.express({"domain": "medical"})
        assert len(incubator.history) == 2


# ---------------------------------------------------------------------------
# Experiment: proof that different environments express different genes
# ---------------------------------------------------------------------------

class TestExperiment:
    def test_five_environments(self):
        genome = constraint_genome()
        incubator = Incubator(genome)
        for domain in ["maritime", "medical", "automotive", "aerospace", "industrial"]:
            incubator.express({"domain": domain})
        assert len(incubator.history) == 5

    def test_maritime_expresses_maritime_genes(self):
        genome = constraint_genome()
        incubator = Incubator(genome)
        result = incubator.express({"domain": "maritime"})
        active = result["profile"].active_genes
        maritime_genes = {g for g in active if g.startswith("nav_") or g == "solas_compliance" or g == "wave_response"}
        assert len(maritime_genes) >= 3

    def test_different_environments_different_genes(self):
        """CORE PROOF: different environments express different gene sets."""
        genome = constraint_genome()
        incubator = Incubator(genome)
        expressed = {}
        for domain in ["maritime", "medical", "automotive", "aerospace", "industrial"]:
            result = incubator.express({"domain": domain})
            expressed[domain] = set(result["profile"].active_genes)

        maritime = expressed["maritime"]
        medical = expressed["medical"]
        assert maritime != medical, "Maritime and medical must express different genes"

        for env_name, genes in expressed.items():
            assert len(genes) < 25, f"{env_name} expressed all genes — expression not selective"

    def test_each_environment_expresses_something(self):
        genome = constraint_genome()
        for domain in ["maritime", "medical", "automotive", "aerospace", "industrial"]:
            incubator = Incubator(genome)
            result = incubator.express({"domain": domain})
            assert len(result["profile"].active_genes) > 0, f"{domain} expressed zero genes"

    def test_genome_is_fixed_across_environments(self):
        genome = constraint_genome()
        gene_ids_before = set(genome.genes.keys())
        incubator = Incubator(genome)
        for domain in ["maritime", "medical", "automotive", "aerospace", "industrial"]:
            incubator.express({"domain": domain})
        gene_ids_after = set(genome.genes.keys())
        assert gene_ids_before == gene_ids_after, "Genome changed during expression!"

    def test_coverage(self):
        genome = constraint_genome()
        incubator = Incubator(genome)
        all_expressed = set()
        for domain in ["maritime", "medical", "automotive", "aerospace", "industrial"]:
            result = incubator.express({"domain": domain})
            all_expressed |= set(result["profile"].active_genes)
        assert len(all_expressed) >= 15, f"Only {len(all_expressed)} genes covered"

    def test_expression_levels_vary(self):
        genome = constraint_genome()
        incubator = Incubator(genome)
        for domain in ["maritime", "medical", "automotive", "aerospace", "industrial"]:
            result = incubator.express({"domain": domain, "regulatory": True})
            levels = list(result["profile"].expression_levels.values())
            if len(levels) > 1:
                assert max(levels) > min(levels), \
                    f"{domain}: all expression levels identical"


# ---------------------------------------------------------------------------
# Checker factory tests
# ---------------------------------------------------------------------------

class TestCheckers:
    def test_range_check(self):
        check = make_range_check(0, 10)
        data = np.array([[5.0, 5.0, 5.0], [15.0, 15.0, 15.0]])
        result = check(data)
        assert result[0] == 0.0
        assert result[1] == 1.0

    def test_variance_check(self):
        check = make_variance_check(1.0)
        data = np.array([[1.0, 1.0, 1.0], [0.0, 5.0, 10.0]])
        result = check(data)
        assert result[0] == 0.0
        assert result[1] == 1.0

    def test_threshold_check(self):
        check = make_threshold_check(0.5)
        data = np.array([[0.8, 0.9], [0.1, 0.2]])
        result = check(data)
        assert result[0] == 0.0
        assert result[1] == 1.0

    def test_monotonic_check(self):
        check = make_monotonic_check()
        data = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        result = check(data)
        assert result[0] == 0.0
        assert result[1] == 1.0

    def test_emission_check(self):
        check = make_emission_check(1.0)
        data = np.array([[0.5, 0.5], [2.0, 2.0]])
        result = check(data)
        assert result[0] == 0.0
        assert result[1] == 1.0
