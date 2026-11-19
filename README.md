<div align="center">

# ALADIN Plasma Cosmology

**Native cylindrical MHD for Z-pinch and J × B plasma dynamics, from the laboratory to astrophysical scales**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![CUDA / CuPy](https://img.shields.io/badge/CUDA-CuPy%20RawKernel-76B900.svg)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--4457--557X-A6CE39.svg)](https://orcid.org/0009-0006-4457-557X)
[![Command Center](https://img.shields.io/badge/Command%20Center-Hugging%20Face-blueviolet.svg)](https://huggingface.co/spaces/aladinibz/PlasmaCosmology)

[Overview](#overview) · [Z-pinch](#the-z-pinch) · [J × B](#the-j--b-force) · [History](#historical-foundations) · [Driver](#the-aladin-driver) · [Kernels](#the-eleven-kernels) · [Programs](#research-programs) · [Lab](#the-aladin-lab)

</div>

---

## Overview

Almost all visible matter in the universe is plasma: an electrically conducting medium that carries currents and responds to magnetic fields. In plasma, electromagnetic forces do not just act on matter; they **organise** it. Currents draw plasma into columns, filaments and flux ropes, and those structures appear at every scale, from laboratory discharges to the solar corona, planetary magnetospheres and the interstellar medium.

**ALADIN** studies this organising role of electromagnetism. Its core is two closely linked pieces of physics:

- **The Z-pinch**: the basic configuration in which an axial current confines its own plasma.
- **The J × B force**: the Lorentz force through which currents and magnetic fields self-organise plasma into coherent structures.

To investigate them, the project develops the **ALADIN Driver**: a GPU-accelerated MHD solver in native cylindrical coordinates $(r,\phi,z)$, built from **eleven computational kernels**. Its purpose is to test plasma-cosmology ideas through controlled numerical experiments on pinches, instabilities, field-aligned currents and magnetized astrophysical environments.

---

## The Z-pinch

An axial current $J_z$ generates an azimuthal magnetic field $B_\phi$, and the interaction between the two compresses the plasma toward the axis:

$$
\mathbf B = B_\phi(r)\,\hat{\boldsymbol\phi}, \qquad \mathbf J = J_z(r)\,\hat{\mathbf z}.
$$

The Z-pinch is one of the oldest and most fundamental configurations in plasma physics. It is the reference system for:

- **radial force balance** between magnetic compression and plasma pressure,
- **equilibrium** of current-carrying columns,
- **instabilities** that reshape them: the $m=0$ **sausage** mode and the $m=1$ **kink** mode,
- the behaviour of cylindrical geometry near the **axis** and the **outer boundary**.

It also serves as ALADIN's primary numerical benchmark.

---

## The J × B force

The force density on a current-carrying plasma is the Lorentz force

$$
\boxed{\;\mathbf f_L = \mathbf J \times \mathbf B\;}, \qquad \mathbf J = \frac{1}{\mu_0}\,\nabla\times\mathbf B .
$$

**J × B is the self-organising mechanism of plasma.** It concentrates currents into filaments, holds pinches together, twists flux ropes, and drives the structures that emerge in magnetized plasma. In the Z-pinch it takes its simplest form:

$$
J_z = \frac{1}{\mu_0\, r}\frac{d}{dr}\bigl(r B_\phi\bigr), \qquad f_r = -J_z B_\phi ,
$$

an inward radial force. In a static equilibrium it is balanced by the pressure gradient, $dP/dr = f_r$.

Because J × B is built directly from the current, an accurate discrete curl $\nabla\times\mathbf B$ is the foundation of everything ALADIN computes.

### Reference benchmark

The normalised analytical Z-pinch ($\mu_0 = 1$):

$$
B_\phi = \tfrac12 r \quad\Longrightarrow\quad J_z = 1, \qquad f_r = -\tfrac12 r, \qquad \nabla\cdot\mathbf B = 0, \qquad P(r) = P_0 - \tfrac14 r^2 .
$$

| Observable | Analytical value | Tests |
|---|---|---|
| Azimuthal field | $B_\phi = 0.5\,r$ | cylindrical field representation |
| Axial current | $J_z = 1$ | discrete curl |
| Radial Lorentz force | $f_r = -0.5\,r$ | J × B reconstruction |
| Divergence | $\nabla\cdot\mathbf B = 0$ | magnetic constraint |
| Pressure balance | $dP/dr = -0.5\,r$ | force equilibrium |

---

## Historical foundations

ALADIN builds on more than two centuries of work on currents, magnetic forces and pinches.

**The J × B force**

| Year | | Contribution |
|---|---|---|
| 1820–1826 | **André-Marie Ampère** | Forces between electric currents; the foundation of electrodynamics |
| 1820 | **Jean-Baptiste Biot & Félix Savart** | The magnetic field produced by a current |
| 1865 | **James Clerk Maxwell** | Unified equations of electricity and magnetism |
| 1889 | **Oliver Heaviside** | The magnetic force on a moving charge |
| 1892–1895 | **Hendrik Lorentz** | The complete force law $\mathbf F = q(\mathbf E + \mathbf v\times\mathbf B)$, whose continuum form is $\mathbf J\times\mathbf B$ |

**The pinch**

| Year | | Contribution |
|---|---|---|
| 1790 | **Martinus van Marum** | High-current discharges through wires: the first observed pinch effects |
| 1905 | **J. A. Pollock & S. Barraclough** | Explained a lightning-crushed copper tube as magnetic compression by the current: the first explained Z-pinch |
| 1934 | **Willard H. Bennett** | The Bennett relation: equilibrium of a self-confined current-carrying plasma |
| 1937 | **Lewi Tonks** | Named the "pinch effect" |
| 1954 | **Martin Kruskal & Martin Schwarzschild** | Theory of pinch instabilities, including the kink and sausage modes |
| 1950s | **ZETA, Perhapsatron and Kurchatov programs** | Z-pinches as the first large fusion experiments |

**Plasma in space**

| Year | | Contribution |
|---|---|---|
| 1942–1970s | **Hannes Alfvén** | Founder of magnetohydrodynamics (Nobel Prize in Physics, 1970); argued that currents and magnetic forces shape cosmic plasma |
| 1986 | **Anthony Peratt** | Simulations of interacting current filaments forming galaxy-like structures |

**Plasma laboratories today**

The same physics is studied experimentally every day. A few facilities closest to ALADIN's focus:

| Facility | Location | Relevance |
|---|---|---|
| **Z Machine** — Sandia National Laboratories | New Mexico, USA | The world's most powerful pulsed-power Z-pinch, driving currents of tens of megaamperes |
| **MAGPIE** — Imperial College London | London, UK | Pulsed-power laboratory astrophysics: Z-pinch-driven jets, shocks and magnetized flows |
| **Zap Energy** | Washington, USA | Sheared-flow-stabilised Z-pinch, aiming at fusion with a pinch that resists its instabilities |
| **Bellan Plasma Lab** — Caltech | California, USA | Laboratory flux ropes and jets that mimic solar and astrophysical structures |
| **MRX** — Princeton Plasma Physics Laboratory | New Jersey, USA | Magnetic reconnection, the process that releases magnetic energy in plasma |
| **LAPD** — Basic Plasma Science Facility, UCLA | California, USA | Large-scale plasma device for Alfvén waves and current-carrying plasma |
| **Wendelstein 7-X** — Max Planck IPP | Greifswald, Germany | Stellarator for long-duration magnetic confinement |
| **ITER** | Cadarache, France | International tokamak, the largest magnetic-confinement fusion experiment under construction |

And in space, missions such as **NASA's Magnetospheric Multiscale (MMS)** and the **Parker Solar Probe** measure currents, fields and reconnection directly in natural plasma.

---

## The ALADIN Driver

The Driver is a modular MHD solver written in **Python and CUDA** (CuPy `RawKernel`) for NVIDIA GPUs. It coordinates geometry, state, boundary conditions, time integration and diagnostics, and delegates each physical operation to a dedicated kernel.

**Design**

- **Native cylindrical coordinates.** Metric factors, radial face areas and cylindrical cell volumes are part of every operator:
  $$\Delta V = \frac{r_{i+1/2}^2 - r_{i-1/2}^2}{2}\,\Delta\phi\,\Delta z .$$
- **Staggered magnetic fields.** Face-centred $\mathbf B$ and edge-centred EMFs, in a constrained-transport layout.
- **GPU execution.** Each heavy operation is its own CUDA kernel, run on L4- and A100-class hardware.
- **Diagnostics.** Current density, Lorentz force, magnetic and kinetic energy, $\nabla\cdot\mathbf B$, conservation and residuals.
- **Modularity.** Every kernel can be tested in isolation before it is coupled into the full solver.

---

## The eleven kernels

| # | Kernel | Role | Status |
|:-:|---|---|---|
| 01 | **PRIM** | Primitive variables (density, velocity, pressure) from the conserved state. | In development |
| 02 | **GYRO** | Gyrotropic pressure anisotropy and closures beyond the bi-Maxwellian model. | In development |
| 03 | **G2** | G₂ / octonion geometric diagnostics of the plasma state (read-only, adds no force). | **Implementation verified** (v01, self-test 122/122 · [DOI](https://doi.org/10.5281/zenodo.24369578)) |
| 04 | **HLLD-R** | HLLD Riemann fluxes across radial faces. | In development |
| 05 | **HLLD-φ** | HLLD fluxes across azimuthal faces. | In development |
| 06 | **HLLD-Z** | HLLD fluxes across axial faces. | In development |
| 07 | **RESIDUAL** | Flux divergences and cylindrical source terms assembled into the evolution residual. | In development |
| 08 | **UCT-EMF** | Edge-centred electromotive force, including Hall and resistive terms. | Under verification |
| 09 | **UCT-CURL** | Current density $\mathbf J = \nabla\times\mathbf B$, the basis of the J × B force. | **Verified** (v1.3, Campaigns 01–05) |
| 10 | **E-GLM** | Divergence cleaning and control of $\nabla\cdot\mathbf B$ errors. | Under verification |
| 11 | **UCT-ASSEMBLY** | Constrained-transport assembly and the staggered magnetic-field update. | In development |

*Status levels:* **Verified** — passed a full adversarial campaign with a published evidence bundle · **Implementation verified** — passed a published self-test suite with independent references and negative controls; not yet a full adversarial campaign or Driver-integrated · **Under verification** — implemented, campaign in progress · **In development** — being implemented · **Experimental** — research model, not part of the core MHD solver.

**How they fit together:** PRIM → HLLD (R, φ, Z) → RESIDUAL advance the plasma; UCT-CURL → UCT-EMF → UCT-ASSEMBLY advance the magnetic field; E-GLM keeps it divergence-free; GYRO extends the physics model; G2 analyses the geometry of the evolving state.

---

## Verification

Each kernel is validated in a dedicated test campaign before integration. Its source is frozen by SHA-256 hash, its results are compared against independent analytical references, and it runs on both CPU and GPU in FP32 and FP64.

Every campaign is published as a **TestReport Bundle**: source and hashes, configuration, raw results, figures, verdicts, recorded limitations and a one-command reproduction script. Anyone can rerun the work and check it.

**UCT-CURL v1.3** has completed five campaigns (50 tests): second-order convergence against independent analytical, exact-discrete and DEC references, including Bessel force-free fields, the Z-pinch and helical perturbations, on CPU and GPU in FP32 and FP64. The campaign reports also record its known limitations: it is not a mimetic operator (div J ≠ 0 at truncation level) and grid-scale checkerboard modes of B are invisible to it. Full integration testing waits for the remaining kernels. A DEC-based successor, `uct_curl_v02`, with edge-centred currents and exact discrete identities, is planned.

**G2 v01** ([10.5281/zenodo.24369578](https://doi.org/10.5281/zenodo.24369578)) passes 122/122 self-test checks on CPU and GPU: octonion identities, Z-pinch and helical reference fields against an independent implementation, second-order convergence against an analytic oracle, rotation and Galilean invariance, and negative controls that fire as designed. Its main exact result: in the default mode the pulled-back G₂ 3-form equals Re det ∇(u + i b), a combination of the classical Elsässer gradient invariants. An independent MHD study (turbulence and a kinking Z-pinch) found no additional predictive power in the regimes tested; these negative results are published with the kernel. Behaviour inside the coupled Driver is future work.

A kernel is marked *Verified* only after its campaign is complete. A kernel-level result is not a full-solver result: coupled time-evolution benchmarks follow once all kernels are integrated.

---

## ALADIN Online Command Center

**[huggingface.co/spaces/aladinibz/PlasmaCosmology](https://huggingface.co/spaces/aladinibz/PlasmaCosmology)**

The Driver connects to an online research environment through its **Adaptive Intelligence Layer (AIL)**. Experiments upload selected diagnostics, which can then be visualised, compared across runs and analysed with AI assistance to identify trends, anomalies and follow-up experiments.

---

## Research programs

| Program | Focus | Directory |
|---|---|---|
| **Z-pinch** | Equilibrium, confinement and benchmarks | [`Z-pinch/`](Z-pinch/) |
| **J × B** | Lorentz-force dynamics and plasma self-organisation | [`JxB/`](JxB/) |
| **Kink** | $m=1$ current-driven instability | [`Kink/`](Kink/) |
| **Sausage** | $m=0$ pinch instability | [`Sausage/`](Sausage/) |
| **Birkeland** | Field-aligned current systems | [`Birkeland/`](Birkeland/) |
| **Galactic Disk** | Magnetized disks and galactic rotation | [`GalacticDisk/`](GalacticDisk/) |
| **Gravity** | Gravitational forces compared with electromagnetic ones | [`Gravity/`](Gravity/) |

---

## Repository structure

```text
Aladin-Plasma-Cosmology/
├── AladinDriver/                  # Driver framework
├── Karnels/                       # Kernels and their test campaigns
├── Karnel/Entropy/Manifold/Gyro/  # Gyro, entropy and manifold research
├── Z-pinch/                       # Z-pinch studies and benchmarks
├── JxB/                           # Lorentz-force studies
├── Kink/  Sausage/                # Pinch instabilities
├── Birkeland/                     # Field-aligned currents
├── GalacticDisk/  Gravity/        # Astrophysical applications
├── CITATION.cff
└── LICENSE
```

## Getting started

```bash
git clone https://github.com/aladinibz/Aladin-Plasma-Cosmology.git
cd Aladin-Plasma-Cosmology
```

Requires Python ≥ 3.10 and NumPy; GPU runs need CuPy and an NVIDIA GPU with a matching CUDA 12.x driver. Each experiment directory includes its own launcher and documentation.

**Status:** active development. Kernel campaigns are being completed one by one; runnable examples and coupled-solver benchmarks will be added as the kernels are integrated.

---

## The ALADIN Lab

**Mihai Alexandru Bucurenciu (ALADIN)**: founder and lead researcher

**AI research team**

| Member | Role |
|---|---|
| **SuperGrok** (xAI) | exploration and research ideas |
| **ChatGPT** (OpenAI) | scientific review and verification standards |
| **Claude** (Anthropic) | adversarial test campaigns, evidence bundles and technical documentation |

Research direction and scientific responsibility remain with the lead researcher.

---

## Citation

```bibtex
@software{aladin_plasma_cosmology_2026,
  author  = {Bucurenciu, Mihai Alexandru},
  title   = {ALADIN Plasma Cosmology},
  year    = {2026},
  url     = {https://github.com/aladinibz/Aladin-Plasma-Cosmology},
  license = {MIT}
}
```

**Mihai Alexandru Bucurenciu (Aladin)** · [ORCID 0009-0006-4457-557X](https://orcid.org/0009-0006-4457-557X) · aladinibz@proton.me
Archival releases and kernel verification reports on Zenodo.

Released under the [MIT License](LICENSE).
