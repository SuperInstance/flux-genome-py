"""Tests for flux_genome.genome, expression, ribosome, builtins."""

import pytest
import numpy as np
from flux_genome.gene import Gene
from flux_genome.genome import Genome
from flux_genome.expression import ExpressionProfile, Protein, Incubator
from flux_genome.ribosome import Ribosome
from flux_genome.builtins import (
    make_range_check, make_threshold_check, make_variance_check,
    make_monotonic_check, make_symmetry_check, make_bounded_deriv_check,
    make_integral_check, make_orthogonality_check, make_noise_floor_check,
    make_latency_check, constraint_genome,
)


def _make_gene(gene_id="g1", domain="test"):
    """Helper: create a simple gene."""
    return Gene(
        gene_id=gene_id,
        structure=np.array([0.0, 0.0]),
        expression_conditions={"domain": domain},
        protein_template=lambda d: np.zeros(d.shape[0]),
        domain=domain,
    )


class TestGene:
    def test_matches_exact(self):
        gene = _make_gene()
        score = gene.matches_environment({"domain": "test"})
        assert score > 0.5

    def test_no_match(self):
        gene = _make_gene()
        score = gene.matches_environment({"domain": "other"})
        assert score == 0.0

    def test_empty_conditions(self):
        gene = Gene(
            gene_id="g", structure=np.zeros(2),
            expression_conditions={},
            protein_template=lambda d: np.zeros(d.shape[0]),
        )
        assert gene.matches_environment({"anything": 1}) == 0.0

    def test_numeric_condition(self):
        gene = Gene(
            gene_id="g", structure=np.zeros(2),
            expression_conditions={"temp": 100.0},
            protein_template=lambda d: np.zeros(d.shape[0]),
        )
        score = gene.matches_environment({"temp": 100.0})
        assert score > 0.9

    def test_list_condition(self):
        gene = Gene(
            gene_id="g", structure=np.zeros(2),
            expression_conditions={"mode": ["fast", "slow"]},
            protein_template=lambda d: np.zeros(d.shape[0]),
        )
        score = gene.matches_environment({"mode": "fast"})
        assert score > 0.0


class TestGenome:
    def test_add_gene(self):
        genome = Genome()
        genome.add_gene(_make_gene("g1"))
        assert genome.gene_count == 1

    def test_domains(self):
        genome = Genome()
        genome.add_gene(_make_gene("g1", domain="math"))
        genome.add_gene(_make_gene("g2", domain="physics"))
        assert genome.domains == {"math", "physics"}

    def test_regulatory_network(self):
        gene = Gene(
            gene_id="g1", structure=np.zeros(2),
            expression_conditions={"x": 1},
            protein_template=lambda d: np.zeros(d.shape[0]),
            promoters=["g2"],
        )
        genome = Genome()
        genome.add_gene(gene)
        assert "g1" in genome.regulatory_network.get("g2", [])


class TestExpressionProfile:
    def test_strongly_expressed(self):
        profile = ExpressionProfile(
            environment={},
            active_genes=["g1", "g2"],
            expression_levels={"g1": 0.9, "g2": 0.5},
        )
        assert profile.strongly_expressed == ["g1"]

    def test_weakly_expressed(self):
        profile = ExpressionProfile(
            environment={},
            active_genes=["g1", "g2"],
            expression_levels={"g1": 0.4, "g2": 0.9},
        )
        assert profile.weakly_expressed == ["g1"]


class TestProtein:
    def test_execute(self):
        protein = Protein(
            protein_id="p1",
            assembled_from=["g1"],
            procedure=lambda d: np.zeros(d.shape[0]),
        )
        data = np.random.randn(5, 3)
        result = protein.execute(data)
        assert result.shape == (5,)

    def test_tick_and_lifecycle(self):
        protein = Protein(
            protein_id="p1",
            assembled_from=["g1"],
            procedure=lambda d: np.zeros(d.shape[0]),
            lifetime=1.0,
            degradation_rate=0.5,
        )
        assert protein.is_alive
        protein.tick()
        assert protein.lifetime == 0.5
        protein.tick()
        assert not protein.is_alive


class TestRibosome:
    def test_transcript(self):
        ribosome = Ribosome()
        genome = Genome()
        genome.add_gene(_make_gene("g1"))
        profile = ribosome.transcript(genome, {"domain": "test"})
        assert "g1" in profile.active_genes

    def test_translate(self):
        ribosome = Ribosome()
        gene = _make_gene()
        protein = ribosome.translate(gene, 0.8, {})
        assert protein.protein_id == "protein_g1"
        assert protein.lifetime > 0

    def test_translate_profile(self):
        ribosome = Ribosome()
        genome = Genome()
        genome.add_gene(_make_gene("g1"))
        profile = ExpressionProfile(
            environment={"domain": "test"},
            active_genes=["g1"],
            expression_levels={"g1": 0.8},
        )
        proteins = ribosome.translate_profile(genome, profile)
        assert len(proteins) == 1


class TestIncubator:
    def test_express(self):
        genome = Genome()
        genome.add_gene(_make_gene("g1"))
        incubator = Incubator(genome)
        result = incubator.express({"domain": "test"})
        assert "profile" in result
        assert "proteins" in result
        assert len(result["proteins"]) >= 1

    def test_express_with_data(self):
        genome = Genome()
        genome.add_gene(_make_gene("g1"))
        incubator = Incubator(genome)
        data = np.random.randn(5, 3)
        result = incubator.express({"domain": "test"}, data=data)
        assert "results" in result

    def test_tick_degrades_proteins(self):
        genome = Genome()
        genome.add_gene(_make_gene("g1"))
        incubator = Incubator(genome)
        incubator.express({"domain": "test"})
        for _ in range(200):
            incubator.tick()
        # Proteins should eventually die
        assert len(incubator.proteins) == 0

    def test_history(self):
        genome = Genome()
        genome.add_gene(_make_gene("g1"))
        incubator = Incubator(genome)
        incubator.express({"domain": "test"})
        assert len(incubator.history) == 1


class TestBuiltins:
    def test_make_range_check(self):
        checker = make_range_check(0, 10)
        data = np.array([[5, 5], [15, 15]])
        result = checker(data)
        assert result[0] == 0.0  # within range
        assert result[1] == 1.0  # outside range

    def test_make_threshold_check_above(self):
        checker = make_threshold_check(5.0, mode="above")
        data = np.array([[3, 3], [7, 7]])
        result = checker(data)
        assert result[0] == 1.0  # below threshold → violation
        assert result[1] == 0.0

    def test_make_variance_check(self):
        checker = make_variance_check(1.0)
        data = np.array([[5, 5, 5], [1, 5, 9]])
        result = checker(data)
        assert result[0] == 0.0  # low variance
        assert result[1] == 1.0  # high variance

    def test_make_monotonic_check(self):
        checker = make_monotonic_check()
        data = np.array([[1, 2, 3], [3, 2, 1]])
        result = checker(data)
        assert result[0] == 0.0
        assert result[1] == 1.0

    def test_make_bounded_deriv_check(self):
        checker = make_bounded_deriv_check(2.0)
        data = np.array([[1, 2, 3], [1, 5, 9]])
        result = checker(data)
        assert result[0] == 0.0
        assert result[1] == 1.0

    def test_constraint_genome(self):
        genome = constraint_genome()
        assert genome.gene_count > 0
        assert len(genome.domains) > 0
