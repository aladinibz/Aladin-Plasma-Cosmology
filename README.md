# ALADIN Plasma Cosmology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CuPy](https://img.shields.io/badge/CuPy-A100-green.svg)](https://cupy.dev/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)]()

**High-performance explicit cylindrical MHD / QMHD driver for plasma cosmology.**

ALADIN Plasma Cosmology is an open-source research code focused on GPU-accelerated, staggered-mesh simulations of magnetized plasmas in cylindrical geometry. It targets applications in galactic disks, Z-pinch dynamics, current-driven instabilities (kink, sausage), Birkeland currents, and J×B force support — all within the broader ALADIN plasma cosmology framework.

The code is designed for Colab Pro / A100-class hardware and emphasizes:

- Explicit cylindrical staggered constrained transport (CT)
- Production UCT + Hall EMF kernels
- Cylindrical HLLD Riemann solvers
- Gyrotropic kinetic closures
- G₂ / octonion entropy geometry kernels
- Axis-safe discrete exterior calculus operators

---

## Repository Structure

```
Aladin-Plasma-Cosmology/
├── AladinDriver/          # Main driver and SSP-RK3 integration
├── kernels/               # RawKernel implementations (UCT, HLLD, G2, gyro…)
├── Z-pinch/               # Z-pinch configurations and diagnostics
├── Kink/                  # Kink-mode modules
├── Sausage/               # Sausage-mode modules
├── JxB/                   # J×B force analysis
├── Birkeland/             # Birkeland current modules
├── GalacticDisk/          # Galactic disk setups and rotation curves
├── Gravity/               # Gravity solvers (Miyamoto-Nagai + NFW, etc.)
├── examples/              # Runnable examples
├── tests/                 # Validation and unit tests
└── docs/                  # Documentation
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/aladinibz/Aladin-Plasma-Cosmology.git
cd Aladin-Plasma-Cosmology

# Install dependencies (Colab / CUDA 12.x recommended)
pip install -r requirements.txt

# Run a basic example (once examples are populated)
python examples/basic_driver.py
```

**Requirements**: Python ≥ 3.10, CUDA 12.x, CuPy, NumPy, Matplotlib.

---

## Key Components

| Component              | Description                                      | Status     |
|------------------------|--------------------------------------------------|------------|
| AladinDriver           | 9–10 kernel SSP-RK3 sequence, cylindrical        | Active     |
| UCT + Hall EMF         | Londrillo–Mignone UCT, axis-safe, DEC J          | Production |
| Cylindrical HLLD       | Explicit r/φ/z HLLD face fluxes                  | Production |
| G₂ Entropy Kernel      | Octonion / G₂ geometry + DEC torsion projection  | Active     |
| Gyrotropic Closure     | 12-feature max-entropy kinetic closure           | Production |
| Discrete Exterior Calc | Axis-safe cylindrical DEC operators              | Production |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{aladin_plasma_cosmology_2026,
  author = {Bucurenciu, Mihai Alexandru},
  title = {ALADIN Plasma Cosmology},
  year = {2026},
  url = {https://github.com/aladinibz/Aladin-Plasma-Cosmology},
  license = {MIT}
}
```

See `CITATION.cff` for machine-readable citation data.

---

## Related Work

- Theory repository: [AladinEquationVinfinity](https://github.com/aladinibz/AladinEquationVinfinity)
- Zenodo deposits (ALADIN ∞ ℂ(t) series and plasma papers)

---

## Author

**Mihai Alexandru Bucurenciu (Aladin)**  
Independent Researcher  
ORCID: [0009-0006-4457-557X](https://orcid.org/0009-0006-4457-557X)  
Contact: aladinibz@proton.me

---

## License

MIT License — see [LICENSE](LICENSE) for details.
