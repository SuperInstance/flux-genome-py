"""
Built-in constraint genomes — pre-built for immediate use.

constraint_genome(): 25 genes across 5 domains (maritime, medical,
automotive, aerospace, industrial). Users can start immediately.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from flux_genome.gene import Gene
from flux_genome.genome import Genome


# ---------------------------------------------------------------------------
# Checker factories
# ---------------------------------------------------------------------------

def make_range_check(lo: float, hi: float, tol: float = 0.0) -> Callable:
    """Factory: range constraint checker."""
    def check(data: NDArray) -> NDArray:
        violations = np.zeros(data.shape[0], dtype=np.float64)
        for i, row in enumerate(data):
            val = float(np.mean(row))
            if val < lo - tol or val > hi + tol:
                violations[i] = 1.0
        return violations
    return check


def make_threshold_check(threshold: float, mode: str = "above") -> Callable:
    """Factory: threshold constraint checker."""
    def check(data: NDArray) -> NDArray:
        means = np.mean(data, axis=1)
        if mode == "above":
            return (means < threshold).astype(np.float64)
        else:
            return (means > threshold).astype(np.float64)
    return check


def make_variance_check(max_var: float) -> Callable:
    """Factory: variance constraint checker."""
    def check(data: NDArray) -> NDArray:
        variances = np.var(data, axis=1)
        return (variances > max_var).astype(np.float64)
    return check


def make_monotonic_check() -> Callable:
    """Factory: monotonicity constraint checker."""
    def check(data: NDArray) -> NDArray:
        violations = np.zeros(data.shape[0], dtype=np.float64)
        for i, row in enumerate(data):
            diffs = np.diff(row)
            if np.any(diffs < 0):
                violations[i] = 1.0
        return violations
    return check


def make_symmetry_check() -> Callable:
    """Factory: symmetry constraint checker."""
    def check(data: NDArray) -> NDArray:
        violations = np.zeros(data.shape[0], dtype=np.float64)
        for i, row in enumerate(data):
            half = len(row) // 2
            if half > 0 and np.max(np.abs(row[:half] - row[-half:])) > 0.5:
                violations[i] = 1.0
        return violations
    return check


def make_bounded_deriv_check(max_deriv: float) -> Callable:
    """Factory: bounded derivative constraint checker."""
    def check(data: NDArray) -> NDArray:
        violations = np.zeros(data.shape[0], dtype=np.float64)
        for i, row in enumerate(data):
            if len(row) > 1:
                derivs = np.abs(np.diff(row))
                if np.max(derivs) > max_deriv:
                    violations[i] = 1.0
        return violations
    return check


def make_integral_check(max_integral: float) -> Callable:
    """Factory: integral bound constraint checker."""
    def check(data: NDArray) -> NDArray:
        integrals = np.sum(np.abs(data), axis=1)
        return (integrals > max_integral).astype(np.float64)
    return check


def make_orthogonality_check(min_dot: float) -> Callable:
    """Factory: orthogonality constraint checker."""
    def check(data: NDArray) -> NDArray:
        violations = np.zeros(data.shape[0], dtype=np.float64)
        for i in range(0, data.shape[0] - 1, 2):
            dot = np.abs(np.dot(data[i], data[i + 1]))
            if dot > min_dot:
                violations[i] = 1.0
                violations[i + 1] = 1.0
        return violations
    return check


def make_noise_floor_check(floor: float) -> Callable:
    """Factory: noise floor constraint checker."""
    def check(data: NDArray) -> NDArray:
        mins = np.min(np.abs(data), axis=1)
        return (mins < floor).astype(np.float64)
    return check


def make_latency_check(max_latency: float) -> Callable:
    """Factory: latency constraint checker."""
    def check(data: NDArray) -> NDArray:
        violations = np.zeros(data.shape[0], dtype=np.float64)
        for i, row in enumerate(data):
            target = row[-1]
            if abs(target) > 0.01:
                settled_idx = len(row)
                for j in range(len(row)):
                    if abs(row[j] - target) / abs(target) < 0.05:
                        settled_idx = j
                        break
                if settled_idx / len(row) > max_latency:
                    violations[i] = 1.0
        return violations
    return check


def make_redundancy_check(min_overlap: int) -> Callable:
    """Factory: redundancy constraint checker."""
    def check(data: NDArray) -> NDArray:
        violations = np.zeros(data.shape[0], dtype=np.float64)
        for i in range(data.shape[0]):
            nonzero = np.count_nonzero(data[i])
            if nonzero < min_overlap:
                violations[i] = 1.0
        return violations
    return check


def make_emission_check(max_level: float) -> Callable:
    """Factory: emission level constraint checker."""
    def check(data: NDArray) -> NDArray:
        means = np.mean(data, axis=1)
        return (means > max_level).astype(np.float64)
    return check


def make_corrosion_check(max_rate: float) -> Callable:
    """Factory: corrosion/degradation rate checker."""
    def check(data: NDArray) -> NDArray:
        violations = np.zeros(data.shape[0], dtype=np.float64)
        for i, row in enumerate(data):
            if len(row) > 1:
                rate = (row[-1] - row[0]) / max(abs(row[0]), 0.001)
                if rate > max_rate:
                    violations[i] = 1.0
        return violations
    return check


def make_stability_check(max_drift: float) -> Callable:
    """Factory: long-term stability checker."""
    def check(data: NDArray) -> NDArray:
        violations = np.zeros(data.shape[0], dtype=np.float64)
        for i, row in enumerate(data):
            if len(row) >= 4:
                quarter = len(row) // 4
                drift = abs(np.mean(row[:quarter]) - np.mean(row[-quarter:]))
                if drift > max_drift:
                    violations[i] = 1.0
        return violations
    return check


def make_spatial_check(max_gradient: float) -> Callable:
    """Factory: spatial gradient constraint checker."""
    def check(data: NDArray) -> NDArray:
        violations = np.zeros(data.shape[0], dtype=np.float64)
        for i, row in enumerate(data):
            if len(row) > 1:
                gradient = np.max(np.abs(np.diff(row)))
                if gradient > max_gradient:
                    violations[i] = 1.0
        return violations
    return check


def make_compatibility_check(standard: str) -> Callable:
    """Factory: standards compatibility checker."""
    std_thresholds = {
        "DO-178C": 0.001,
        "ISO-26262": 0.01,
        "IEC-62304": 0.005,
        "SOLAS": 0.02,
        "IEC-61511": 0.008,
    }
    threshold = std_thresholds.get(standard, 0.01)

    def check(data: NDArray) -> NDArray:
        variances = np.var(data, axis=1)
        return (variances > threshold).astype(np.float64)
    return check


def make_throughput_check(min_rate: float) -> Callable:
    """Factory: throughput constraint checker."""
    def check(data: NDArray) -> NDArray:
        throughputs = np.sum(np.abs(data), axis=1)
        return (throughputs < min_rate).astype(np.float64)
    return check


def make_spectral_check(max_peak: float) -> Callable:
    """Factory: spectral purity constraint checker."""
    def check(data: NDArray) -> NDArray:
        violations = np.zeros(data.shape[0], dtype=np.float64)
        for i, row in enumerate(data):
            fft = np.abs(np.fft.rfft(row))
            if len(fft) > 1 and np.max(fft[1:]) / max(np.mean(fft[1:]), 0.001) > max_peak:
                violations[i] = 1.0
        return violations
    return check


def make_fault_tolerance_check(min_survivors: int) -> Callable:
    """Factory: fault tolerance constraint checker."""
    def check(data: NDArray) -> NDArray:
        violations = np.zeros(data.shape[0], dtype=np.float64)
        for i, row in enumerate(data):
            operational = np.sum(row > 0)
            if operational < min_survivors:
                violations[i] = 1.0
        return violations
    return check


# ---------------------------------------------------------------------------
# Pre-built genome: 25 genes, 5 domains
# ---------------------------------------------------------------------------

def constraint_genome() -> Genome:
    """
    Build the constraint genome: 25 genes covering 5 domains.

    Gene loci are Eisenstein lattice points (golden-ratio structured).
    Each gene encodes a constraint protein triggered by specific environments.
    """
    phi = (1 + np.sqrt(5)) / 2

    genes = [
        # --- MARITIME (5) ---
        Gene(
            gene_id="nav_position",
            structure=np.array([phi, 1.0, 0.0]),
            expression_conditions={"domain": "maritime"},
            protein_template=make_range_check(-180, 180),
            domain="maritime",
            description="Navigation position bounds check",
        ),
        Gene(
            gene_id="nav_heading",
            structure=np.array([phi, 1.0, 2 * np.pi / 5]),
            expression_conditions={"domain": "maritime"},
            protein_template=make_range_check(0, 360),
            domain="maritime",
            description="Heading angle bounds check",
        ),
        Gene(
            gene_id="nav_stability",
            structure=np.array([phi**2, 1.0, 4 * np.pi / 5]),
            expression_conditions={"domain": "maritime"},
            protein_template=make_variance_check(10.0),
            domain="maritime",
            description="Vessel stability variance check",
        ),
        Gene(
            gene_id="solas_compliance",
            structure=np.array([phi**2, 1.0, 6 * np.pi / 5]),
            expression_conditions={"domain": "maritime", "regulatory": True},
            protein_template=make_compatibility_check("SOLAS"),
            domain="maritime",
            description="SOLAS regulatory compliance check",
            promoters=["nav_position", "nav_heading"],
        ),
        Gene(
            gene_id="wave_response",
            structure=np.array([phi, 2.0, 8 * np.pi / 5]),
            expression_conditions={"domain": "maritime"},
            protein_template=make_bounded_deriv_check(3.0),
            domain="maritime",
            description="Wave response rate-of-change check",
        ),

        # --- MEDICAL (5) ---
        Gene(
            gene_id="patient_vitals",
            structure=np.array([phi**2, 0.0, np.pi / 3]),
            expression_conditions={"domain": "medical"},
            protein_template=make_range_check(60, 200),
            domain="medical",
            description="Patient vital signs range check",
        ),
        Gene(
            gene_id="drug_dosage",
            structure=np.array([phi, 0.0, 2 * np.pi / 3]),
            expression_conditions={"domain": "medical", "safety_critical": True},
            protein_template=make_threshold_check(5.0, mode="above"),
            domain="medical",
            description="Drug dosage safety threshold",
        ),
        Gene(
            gene_id="alarms",
            structure=np.array([phi**3, 0.0, np.pi]),
            expression_conditions={"domain": "medical"},
            protein_template=make_noise_floor_check(0.01),
            domain="medical",
            description="Alarm signal integrity check",
            promoters=["patient_vitals"],
        ),
        Gene(
            gene_id="iec62304",
            structure=np.array([phi**2, 0.0, 4 * np.pi / 3]),
            expression_conditions={"domain": "medical", "regulatory": True},
            protein_template=make_compatibility_check("IEC-62304"),
            domain="medical",
            description="IEC-62304 medical software compliance",
            promoters=["patient_vitals", "drug_dosage"],
        ),
        Gene(
            gene_id="contamination",
            structure=np.array([phi, 0.0, 5 * np.pi / 3]),
            expression_conditions={"domain": "medical"},
            protein_template=make_integral_check(50.0),
            domain="medical",
            description="Cumulative contamination check",
        ),

        # --- AUTOMOTIVE (5) ---
        Gene(
            gene_id="speed_limit",
            structure=np.array([1.0, phi, np.pi / 5]),
            expression_conditions={"domain": "automotive"},
            protein_template=make_threshold_check(130.0),
            domain="automotive",
            description="Speed limit constraint",
        ),
        Gene(
            gene_id="brake_distance",
            structure=np.array([1.0, phi**2, 2 * np.pi / 5]),
            expression_conditions={"domain": "automotive"},
            protein_template=make_monotonic_check(),
            domain="automotive",
            description="Braking distance monotonicity",
        ),
        Gene(
            gene_id="iso26262",
            structure=np.array([1.0, phi, 3 * np.pi / 5]),
            expression_conditions={"domain": "automotive", "regulatory": True},
            protein_template=make_compatibility_check("ISO-26262"),
            domain="automotive",
            description="ISO-26262 functional safety compliance",
            promoters=["speed_limit", "brake_distance"],
        ),
        Gene(
            gene_id="latency_auto",
            structure=np.array([1.0, phi**2, 4 * np.pi / 5]),
            expression_conditions={"domain": "automotive", "realtime": True},
            protein_template=make_latency_check(0.1),
            domain="automotive",
            description="Real-time latency constraint",
        ),
        Gene(
            gene_id="redundancy_auto",
            structure=np.array([1.0, phi, np.pi]),
            expression_conditions={"domain": "automotive"},
            protein_template=make_redundancy_check(2),
            domain="automotive",
            description="System redundancy check",
        ),

        # --- AEROSPACE (5) ---
        Gene(
            gene_id="altitude",
            structure=np.array([2.0, phi, 0.0]),
            expression_conditions={"domain": "aerospace"},
            protein_template=make_range_check(0, 45000),
            domain="aerospace",
            description="Altitude envelope check",
        ),
        Gene(
            gene_id="g_force",
            structure=np.array([2.0, phi**2, 2 * np.pi / 5]),
            expression_conditions={"domain": "aerospace"},
            protein_template=make_threshold_check(9.0),
            domain="aerospace",
            description="G-force structural limit",
        ),
        Gene(
            gene_id="do178c",
            structure=np.array([2.0, phi, 4 * np.pi / 5]),
            expression_conditions={"domain": "aerospace", "regulatory": True},
            protein_template=make_compatibility_check("DO-178C"),
            domain="aerospace",
            description="DO-178C airborne software compliance",
            promoters=["altitude", "g_force"],
        ),
        Gene(
            gene_id="spectral_purity",
            structure=np.array([2.0, phi**2, 6 * np.pi / 5]),
            expression_conditions={"domain": "aerospace"},
            protein_template=make_spectral_check(10.0),
            domain="aerospace",
            description="Signal spectral purity check",
        ),
        Gene(
            gene_id="fault_tolerance",
            structure=np.array([2.0, phi, 8 * np.pi / 5]),
            expression_conditions={"domain": "aerospace"},
            protein_template=make_fault_tolerance_check(2),
            domain="aerospace",
            description="Fault tolerance check (min surviving channels)",
        ),

        # --- INDUSTRIAL (5) ---
        Gene(
            gene_id="temperature",
            structure=np.array([0.0, phi, np.pi / 5]),
            expression_conditions={"domain": "industrial"},
            protein_template=make_range_check(-40, 85),
            domain="industrial",
            description="Operating temperature range check",
        ),
        Gene(
            gene_id="emissions",
            structure=np.array([0.0, phi**2, 3 * np.pi / 5]),
            expression_conditions={"domain": "industrial"},
            protein_template=make_emission_check(50.0),
            domain="industrial",
            description="Emission level constraint",
        ),
        Gene(
            gene_id="corrosion",
            structure=np.array([0.0, phi, np.pi]),
            expression_conditions={"domain": "industrial"},
            protein_template=make_corrosion_check(0.1),
            domain="industrial",
            description="Corrosion/degradation rate check",
        ),
        Gene(
            gene_id="throughput",
            structure=np.array([0.0, phi**2, 7 * np.pi / 5]),
            expression_conditions={"domain": "industrial"},
            protein_template=make_throughput_check(10.0),
            domain="industrial",
            description="Production throughput minimum",
        ),
        Gene(
            gene_id="iec61511",
            structure=np.array([0.0, phi, 9 * np.pi / 5]),
            expression_conditions={"domain": "industrial", "regulatory": True},
            protein_template=make_compatibility_check("IEC-61511"),
            domain="industrial",
            description="IEC-61511 process safety compliance",
            promoters=["temperature", "emissions"],
        ),
    ]

    genome = Genome()
    for gene in genes:
        genome.add_gene(gene)
    return genome
