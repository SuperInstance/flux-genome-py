"""
Music Expression — genome-driven musical constraint evolution.

Connects flux-genome-py (genetic expression) to the constraint music
ecosystem (constraint-theory-core + flux-tensor-midi).

A genome encodes a constraint profile.  Evolution optimises music by
evolving constraint parameters.  Same genome, different contexts,
different music.

Pipeline:
    Genome → GenomeToConstraints → constraint config dict →
    evaluate_fitness → evolve → next generation
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from flux_genome.gene import Gene
from flux_genome.genome import Genome
from flux_genome.expression import Incubator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHI = (1 + np.sqrt(5)) / 2

# 5 musical domains, 5 genes each = 25 genes
MUSIC_DOMAINS = ["snap", "funnel", "consensus", "laman", "tempo"]

# Gene templates per domain: (gene_id, param_name, scale, offset, min_val, max_val)
GENE_SPECS: List[Tuple[str, str, str, float, float, float, float]] = [
    # --- SNAP (5) ---
    ("snap_resolution",  "grid_resolution", "snap",  1.0,  1.0, 2.0, 7.0),
    ("snap_tolerance",   "snap_tolerance",  "snap",  0.2,  0.0, 0.0, 1.0),
    ("snap_strength",    "snap_strength",   "snap",  0.3,  0.0, 0.0, 1.0),
    ("snap_phase",       "snap_phase",      "snap",  0.1,  0.0, 0.0, 0.5),
    ("snap_swing",       "swing_ratio",     "snap",  0.15, 0.5, 0.5, 0.75),
    # --- FUNNEL (5) ---
    ("epsilon_0",        "epsilon_0",       "funnel", 50.0, 5.0, 1.0, 200.0),
    ("decay_rate",       "decay_rate",      "funnel", 0.05, 0.01, 0.001, 0.5),
    ("anomaly_thresh",   "anomaly_threshold","funnel", 100.0, 10.0, 10.0, 500.0),
    ("reset_rate",       "reset_rate",      "funnel", 0.3,  0.0, 0.0, 1.0),
    ("drift_adapt",      "drift_adaptation", "funnel", 0.1,  0.01, 0.01, 0.5),
    # --- CONSENSUS (5) ---
    ("coupling_alpha",   "coupling_alpha",  "consensus", 0.4, 0.1, 0.05, 0.95),
    ("consensus_thresh", "consensus_threshold","consensus", 0.2, 0.0, 0.0, 1.0),
    ("listen_depth",     "listen_depth",    "consensus", 0.5, 0.5, 0.5, 5.5),
    ("correct_rate",     "correct_rate",    "consensus", 0.3, 0.0, 0.0, 1.0),
    ("leader_weight",    "leader_weight",   "consensus", 0.4, 0.0, 0.0, 1.0),
    # --- LAMAN (5) ---
    ("edge_density",     "edge_density",    "laman",   0.3,  0.2, 0.2, 1.0),
    ("min_edges",        "min_edges",       "laman",   1.0,  1.0, 1.0, 10.0),
    ("redundancy",       "redundancy",      "laman",   0.2,  0.0, 0.0, 1.0),
    ("voice_indep",      "voice_independence","laman",  0.3,  0.0, 0.0, 1.0),
    ("coupling_topo",    "coupling_topology","laman",   0.3,  0.0, 0.0, 2.0),
    # --- TEMPO (5) ---
    ("bpm",              "bpm",             "tempo",   40.0, 40.0, 40.0, 240.0),
    ("swing_ratio",      "swing_ratio_t",   "tempo",   0.05, 0.5, 0.5, 0.75),
    ("rubato_extent",    "rubato_extent",   "tempo",   0.15, 0.0, 0.0, 1.0),
    ("accel_decel",      "accel_decel",     "tempo",   0.1,  0.0, 0.0, 1.0),
    ("groove_depth",     "groove_depth",    "tempo",   0.2,  0.0, 0.0, 1.0),
]


# ---------------------------------------------------------------------------
# Genre target profiles (constraint config targets for fitness)
# ---------------------------------------------------------------------------

GENRE_TARGETS: Dict[str, Dict[str, float]] = {
    "jazz": {
        "grid_resolution": 3.0,    # triplet grid
        "snap_tolerance": 0.5,     # loose quantize
        "snap_strength": 0.4,      # rubato-friendly
        "snap_phase": 0.33,        # swing phase
        "swing_ratio": 0.67,       # triplet swing
        "epsilon_0": 80.0,         # wide initial tolerance
        "decay_rate": 0.05,        # slow tightening
        "anomaly_threshold": 150.0,
        "reset_rate": 0.3,
        "drift_adaptation": 0.08,
        "coupling_alpha": 0.3,     # loose coupling
        "consensus_threshold": 0.4,
        "listen_depth": 2.5,
        "correct_rate": 0.3,
        "leader_weight": 0.3,
        "edge_density": 0.4,
        "min_edges": 3.0,
        "redundancy": 0.2,
        "voice_independence": 0.8,  # high independence
        "coupling_topology": 1.0,   # ring
        "bpm": 180.0,
        "swing_ratio_t": 0.67,
        "rubato_extent": 0.7,      # expressive timing
        "accel_decel": 0.3,
        "groove_depth": 0.4,
    },
    "electronic": {
        "grid_resolution": 4.0,
        "snap_tolerance": 0.02,    # machine tight
        "snap_strength": 0.98,
        "snap_phase": 0.0,
        "swing_ratio": 0.5,        # straight
        "epsilon_0": 20.0,
        "decay_rate": 0.3,         # fast tightening
        "anomaly_threshold": 50.0,
        "reset_rate": 0.1,
        "drift_adaptation": 0.05,
        "coupling_alpha": 0.9,
        "consensus_threshold": 0.9,
        "listen_depth": 3.5,
        "correct_rate": 0.8,
        "leader_weight": 0.7,
        "edge_density": 0.8,
        "min_edges": 5.0,
        "redundancy": 0.3,
        "voice_independence": 0.1,
        "coupling_topology": 2.0,   # full mesh
        "bpm": 128.0,
        "swing_ratio_t": 0.5,
        "rubato_extent": 0.0,
        "accel_decel": 0.0,
        "groove_depth": 0.95,
    },
    "hiphop": {
        "grid_resolution": 4.0,
        "snap_tolerance": 0.1,
        "snap_strength": 0.9,
        "snap_phase": 0.0,
        "swing_ratio": 0.6,
        "epsilon_0": 40.0,
        "decay_rate": 0.15,
        "anomaly_threshold": 80.0,
        "reset_rate": 0.2,
        "drift_adaptation": 0.12,
        "coupling_alpha": 0.6,
        "consensus_threshold": 0.7,
        "listen_depth": 2.0,
        "correct_rate": 0.5,
        "leader_weight": 0.8,
        "edge_density": 0.5,
        "min_edges": 3.0,
        "redundancy": 0.1,
        "voice_independence": 0.3,
        "coupling_topology": 0.0,   # star
        "bpm": 140.0,
        "swing_ratio_t": 0.6,
        "rubato_extent": 0.1,
        "accel_decel": 0.05,
        "groove_depth": 0.8,
    },
    "classical": {
        "grid_resolution": 2.0,
        "snap_tolerance": 0.15,
        "snap_strength": 0.7,
        "snap_phase": 0.0,
        "swing_ratio": 0.5,
        "epsilon_0": 50.0,
        "decay_rate": 0.08,
        "anomaly_threshold": 100.0,
        "reset_rate": 0.2,
        "drift_adaptation": 0.04,
        "coupling_alpha": 0.5,
        "consensus_threshold": 0.6,
        "listen_depth": 3.0,
        "correct_rate": 0.4,
        "leader_weight": 0.5,
        "edge_density": 0.7,
        "min_edges": 4.0,
        "redundancy": 0.3,
        "voice_independence": 0.7,
        "coupling_topology": 1.0,
        "bpm": 72.0,
        "swing_ratio_t": 0.5,
        "rubato_extent": 0.3,
        "accel_decel": 0.2,
        "groove_depth": 0.3,
    },
    "math": {
        "grid_resolution": 5.0,
        "snap_tolerance": 0.0,      # exact
        "snap_strength": 1.0,
        "snap_phase": 0.0,
        "swing_ratio": 0.5,
        "epsilon_0": 10.0,
        "decay_rate": 0.5,
        "anomaly_threshold": 30.0,
        "reset_rate": 0.05,
        "drift_adaptation": 0.02,
        "coupling_alpha": 0.7,
        "consensus_threshold": 0.95,
        "listen_depth": 2.0,
        "correct_rate": 0.9,
        "leader_weight": 0.6,
        "edge_density": 0.6,
        "min_edges": 3.0,
        "redundancy": 0.0,
        "voice_independence": 0.5,
        "coupling_topology": 0.0,
        "bpm": 100.0,
        "swing_ratio_t": 0.5,
        "rubato_extent": 0.0,
        "accel_decel": 0.0,
        "groove_depth": 0.1,
    },
}


# ---------------------------------------------------------------------------
# Dummy protein template for music genes
# ---------------------------------------------------------------------------

def _music_protein(data: NDArray) -> NDArray:
    """Placeholder protein — music genes don't check sensor data directly."""
    return np.zeros(data.shape[0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Music Genome Builder
# ---------------------------------------------------------------------------

def music_genome() -> Genome:
    """Build a 25-gene musical constraint genome.

    5 domains × 5 genes = 25 genes encoding musical constraint parameters.
    Gene structures are Eisenstein lattice points (phi-structured).
    """
    genome = Genome()
    for i, (gene_id, param_name, domain, scale, offset, lo, hi) in enumerate(GENE_SPECS):
        angle = 2 * np.pi * (i % 5) / 5
        radius = PHI if i < 15 else PHI ** 2
        structure = np.array([radius, scale, angle])

        gene = Gene(
            gene_id=gene_id,
            structure=structure,
            expression_conditions={"domain": "music", "genre": "any"},
            protein_template=_music_protein,
            promoters=[],
            domain=domain,
            description=f"Musical constraint: {param_name} ({domain} domain)",
        )
        genome.add_gene(gene)
    return genome


# ---------------------------------------------------------------------------
# GenomeToConstraints
# ---------------------------------------------------------------------------

class GenomeToConstraints:
    """Convert a music genome to a constraint configuration dict.

    The genome's gene structures encode parameter values.  Expression
    levels (driven by musical environment) modulate the output.
    """

    def __init__(self, genome: Genome, environment: Optional[Dict[str, Any]] = None):
        self.genome = genome
        self.environment = environment or {"domain": "music"}

    def to_config(self, genre: Optional[str] = None) -> Dict[str, float]:
        """Translate genome to a constraint configuration dict.

        Parameters
        ----------
        genre : str, optional
            If provided, also set expression conditions for this genre
            and use expression levels to modulate parameters.

        Returns
        -------
        Dict mapping parameter names to float values.
        """
        env = dict(self.environment)
        if genre:
            env["genre"] = genre

        # Compute expression levels
        levels: Dict[str, float] = {}
        for gene_id, gene in self.genome.genes.items():
            level = gene.matches_environment(env)
            levels[gene_id] = max(level, 0.3)  # floor at 0.3

        config: Dict[str, float] = {}
        for gene_id, param_name, domain, scale, offset, lo, hi in GENE_SPECS:
            if gene_id not in self.genome.genes:
                continue
            gene = self.genome.genes[gene_id]
            raw = gene.structure[1]  # scale-encoded value
            level = levels.get(gene_id, 0.5)

            # Parameter = structure value * expression level, clamped
            value = raw * level
            value = max(lo, min(hi, value))
            config[param_name] = round(value, 6)

        return config

    def to_genre_config(self, genre: str) -> Dict[str, Any]:
        """Full genre configuration for flux-tensor-midi.

        Returns a dict compatible with GenreBrain.PRESETS format.
        """
        config = self.to_config(genre)

        from flux_tensor_midi.core.snap import RhythmicRole

        # Derive musical properties from constraint config
        grid_res = config.get("grid_resolution", 4.0)
        bpm = config.get("bpm", 120.0)

        # Map grid resolution to rhythmic roles
        role_map = {
            2: RhythmicRole.ROOT,
            3: RhythmicRole.TRIPLET,
            4: RhythmicRole.DOUBLETIME,
            5: RhythmicRole.QUINTUPLE,
        }
        primary_role = role_map.get(int(round(grid_res)), RhythmicRole.ROOT)

        return {
            "genre": genre,
            "constraint_config": config,
            "bpm": bpm,
            "grid_resolution": int(round(grid_res)),
            "swing_ratio": config.get("swing_ratio", 0.5),
            "rubato": config.get("rubato_extent", 0.0) > 0.2,
        }


# ---------------------------------------------------------------------------
# Fitness Evaluation
# ---------------------------------------------------------------------------

def evaluate_fitness(
    config: Dict[str, float],
    target_genre: str,
    novelty_bonus: float = 0.0,
) -> float:
    """Score a constraint configuration against a target genre.

    Fitness = weighted sum of:
      - genre_match: distance to target profile
      - constraint_satisfaction: internal consistency
      - novelty: difference from population average
      - listenability: heuristic quality score

    Parameters
    ----------
    config : dict
        Constraint configuration from GenomeToConstraints.to_config().
    target_genre : str
        One of: jazz, electronic, hiphop, classical, math.
    novelty_bonus : float
        Bonus for being different from population (injected externally).

    Returns
    -------
    float in [0, 1].
    """
    if target_genre not in GENRE_TARGETS:
        raise ValueError(f"Unknown genre '{target_genre}'. "
                         f"Available: {list(GENRE_TARGETS.keys())}")

    target = GENRE_TARGETS[target_genre]

    # 1. Genre match (cosine similarity in parameter space)
    keys = list(target.keys())
    vec_a = np.array([config.get(k, 0.0) for k in keys])
    vec_b = np.array([target[k] for k in keys])

    # Normalise each dimension to [0,1] using known ranges
    for i, key in enumerate(keys):
        spec = next((s for s in GENE_SPECS if s[1] == key), None)
        if spec:
            lo, hi = spec[5], spec[6]
            if hi > lo:
                vec_a[i] = (vec_a[i] - lo) / (hi - lo)
                vec_b[i] = (vec_b[i] - lo) / (hi - lo)

    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    genre_score = dot / max(norm_a * norm_b, 1e-8)
    genre_score = max(0.0, min(1.0, genre_score))

    # 2. Constraint satisfaction (internal consistency checks)
    cs_score = 0.0
    checks = 0

    # Snap strength should be inverse of tolerance (consistent quantization)
    tol = config.get("snap_tolerance", 0.5)
    strength = config.get("snap_strength", 0.5)
    if tol < 0.1 and strength > 0.8:
        cs_score += 1.0  # tight & strong — consistent
    elif tol > 0.3 and strength < 0.6:
        cs_score += 1.0  # loose & weak — consistent
    else:
        cs_score += 0.5  # mixed — somewhat consistent
    checks += 1

    # Coupling should increase with edge density
    coupling = config.get("coupling_alpha", 0.5)
    density = config.get("edge_density", 0.5)
    cs_score += 1.0 - abs(coupling - density)
    checks += 1

    # Rubato inversely correlated with snap strength
    rubato = config.get("rubato_extent", 0.0)
    cs_score += 1.0 - abs(rubato + (1.0 - strength) - 1.0)
    checks += 1

    cs_score /= checks

    # 3. Listenability heuristic
    listen_score = 0.5  # baseline

    # BPM in human range
    bpm = config.get("bpm", 120.0)
    if 40 <= bpm <= 200:
        listen_score += 0.2

    # Some swing is nice
    swing = config.get("swing_ratio", 0.5)
    if 0.5 <= swing <= 0.67:
        listen_score += 0.15

    # Some groove is nice
    groove = config.get("groove_depth", 0.5)
    if 0.1 <= groove <= 0.9:
        listen_score += 0.15

    listen_score = min(1.0, listen_score)

    # Combined fitness
    fitness = (
        0.40 * genre_score +
        0.25 * cs_score +
        0.20 * listen_score +
        0.15 * min(1.0, novelty_bonus)
    )
    return round(fitness, 6)


# ---------------------------------------------------------------------------
# Genetic Operators
# ---------------------------------------------------------------------------

def mutate_genome(
    genome: Genome,
    mutation_rate: float = 0.15,
    mutation_scale: float = 0.2,
) -> Genome:
    """Mutate a genome by perturbing gene structures.

    Returns a new genome (original unchanged).
    """
    new_genome = Genome()
    for gene_id, gene in genome.genes.items():
        g = copy.deepcopy(gene)
        if random.random() < mutation_rate:
            perturbation = np.random.randn(3) * mutation_scale
            g.structure = g.structure + perturbation
        new_genome.add_gene(g)
    return new_genome


def crossover(parent_a: Genome, parent_b: Genome) -> Genome:
    """Single-point crossover between two genomes."""
    gene_ids = list(parent_a.genes.keys())
    if len(gene_ids) < 2:
        return copy.deepcopy(parent_a)

    point = random.randint(1, len(gene_ids) - 1)
    child = Genome()
    for i, gid in enumerate(gene_ids):
        source = parent_a if i < point else parent_b
        child.add_gene(copy.deepcopy(source.genes[gid]))
    return child


def tournament_select(
    population: List[Tuple[Genome, float]],
    k: int = 3,
) -> Genome:
    """Tournament selection: pick k random, return the fittest."""
    contestants = random.sample(population, min(k, len(population)))
    return max(contestants, key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Evolution Engine
# ---------------------------------------------------------------------------

@dataclass
class EvolutionResult:
    """Result of an evolutionary run."""
    best_genome: Genome
    best_fitness: float
    best_config: Dict[str, float]
    history: List[Dict[str, Any]] = field(default_factory=list)
    population: List[Tuple[Genome, float]] = field(default_factory=list)


def evolve(
    target_genre: str,
    population_size: int = 30,
    generations: int = 50,
    mutation_rate: float = 0.15,
    mutation_scale: float = 0.2,
    elitism: int = 2,
    seed: Optional[int] = None,
) -> EvolutionResult:
    """Evolve a population of genomes toward a target genre.

    Parameters
    ----------
    target_genre : str
        Target genre: jazz, electronic, hiphop, classical, math.
    population_size : int
        Number of genomes per generation.
    generations : int
        Number of generations to evolve.
    mutation_rate : float
        Probability of mutating each gene.
    mutation_scale : float
        Standard deviation of mutation perturbation.
    elitism : int
        Number of top genomes carried forward unchanged.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    EvolutionResult with best genome, fitness, config, and history.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Initialise population with random perturbations around base genome
    base = music_genome()
    population: List[Tuple[Genome, float]] = []

    for _ in range(population_size):
        g = mutate_genome(base, mutation_rate=0.5, mutation_scale=0.4)
        config = GenomeToConstraints(g).to_config(target_genre)
        fitness = evaluate_fitness(config, target_genre)
        population.append((g, fitness))

    history: List[Dict[str, Any]] = []

    for gen in range(generations):
        # Sort by fitness
        population.sort(key=lambda x: x[1], reverse=True)

        best_genome, best_fitness = population[0]
        avg_fitness = np.mean([f for _, f in population])
        worst_fitness = population[-1][1]

        # Compute population diversity (avg pairwise distance in config space)
        configs = [
            GenomeToConstraints(g).to_config(target_genre) for g, _ in population
        ]
        keys = list(GENRE_TARGETS[target_genre].keys())
        vectors = np.array([[c.get(k, 0.0) for k in keys] for c in configs])
        if len(vectors) > 1:
            pairwise = 0.0
            count = 0
            for i in range(min(len(vectors), 10)):
                for j in range(i + 1, min(len(vectors), 10)):
                    pairwise += np.linalg.norm(vectors[i] - vectors[j])
                    count += 1
            diversity = pairwise / max(count, 1)
        else:
            diversity = 0.0

        history.append({
            "generation": gen,
            "best_fitness": best_fitness,
            "avg_fitness": round(avg_fitness, 4),
            "worst_fitness": worst_fitness,
            "diversity": round(float(diversity), 4),
        })

        # Selection + reproduction
        new_population: List[Tuple[Genome, float]] = []

        # Elitism
        for i in range(min(elitism, len(population))):
            new_population.append(population[i])

        # Fill rest with crossover + mutation
        while len(new_population) < population_size:
            parent_a = tournament_select(population)
            parent_b = tournament_select(population)
            child = crossover(parent_a, parent_b)
            child = mutate_genome(child, mutation_rate, mutation_scale)

            config = GenomeToConstraints(child).to_config(target_genre)
            # Compute novelty as distance from population average
            novelty = diversity / 100.0  # normalised
            fitness = evaluate_fitness(config, target_genre, novelty_bonus=novelty)
            new_population.append((child, fitness))

        population = new_population

    # Final sort
    population.sort(key=lambda x: x[1], reverse=True)
    best_genome, best_fitness = population[0]
    best_config = GenomeToConstraints(best_genome).to_config(target_genre)

    return EvolutionResult(
        best_genome=best_genome,
        best_fitness=best_fitness,
        best_config=best_config,
        history=history,
        population=population,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_evolve_jazz(
    population_size: int = 30,
    generations: int = 50,
    seed: int = 42,
) -> EvolutionResult:
    """Demo: evolve a population toward jazz sound.

    Example
    -------
    >>> result = demo_evolve_jazz()
    >>> print(f"Best fitness: {result.best_fitness:.4f}")
    >>> print(f"BPM evolved to: {result.best_config.get('bpm', 0):.1f}")
    >>> print(f"Swing ratio: {result.best_config.get('swing_ratio', 0):.3f}")
    >>> print(f"Snap tolerance: {result.best_config.get('snap_tolerance', 0):.3f}")
    """
    return evolve(
        target_genre="jazz",
        population_size=population_size,
        generations=generations,
        seed=seed,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Genome-Music Evolution Demo: Evolving toward Jazz")
    print("=" * 60)

    result = demo_evolve_jazz(population_size=30, generations=50, seed=42)

    print(f"\nBest fitness: {result.best_fitness:.4f}")
    print(f"\nEvolved constraint profile:")
    for key, val in sorted(result.best_config.items()):
        target_val = GENRE_TARGETS["jazz"].get(key, "?")
        print(f"  {key:25s}: {val:8.4f}  (target: {target_val})")

    print(f"\nEvolution history (every 10 generations):")
    for entry in result.history:
        if entry["generation"] % 10 == 0 or entry["generation"] == len(result.history) - 1:
            print(f"  Gen {entry['generation']:3d}: "
                  f"best={entry['best_fitness']:.4f}  "
                  f"avg={entry['avg_fitness']:.4f}  "
                  f"diversity={entry['diversity']:.2f}")

    print("\n" + "=" * 60)
    print("Done.  Genome evolved jazz-like constraint parameters.")
    print("=" * 60)
