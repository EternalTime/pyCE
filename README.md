# pyCE

A Python library for work on Configurational Entropy (CE): an
information-theoretic measure of spatial complexity for field configurations.

## Installation

```bash
git clone https://github.com/EternalTime/pyCE.git
cd pyCE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Requires Python 3.10+; numpy, scipy, matplotlib, astropy, and tqdm are installed
automatically. The authoritative reference for installation is
[Getting Started](https://damiansowinski.com/pyCE/getting_started.html).

## Modules

| Module | Description |
|---|---|
| `pyCE.math` | Radial Fourier transforms and d-dimensional radial integration |
| `pyCE.cosmology` | Analysis of the Cosmic Microwave Background (Planck and WMAP angular power spectra; bundled WMAP 9-yr data) |
| `pyCE.instantons` | Generating and analyzing instantons |
| `pyCE.bosonstars` | Ground-state boson stars of the Einstein-Klein-Gordon system |
| `pyCE.oscillons` | Generating and analyzing oscillons |
| `pyCE.polytropes` | Polytropic models of stars |

## Example

```python
import numpy as np
from pyCE.math import radialFT, radial_integrate

r = np.linspace(0, 10, 1000)
f = np.exp(-r**2)
ft, k = radialFT(3, f, r)            # radial FT in d = 3
norm = radial_integrate(r, f**2, 3)  # d-dimensional radial integration
```

## Documentation

The documentation is hosted at [damiansowinski.com/pyCE](https://damiansowinski.com/pyCE/),
or run `import pyCE; pyCE.docs()` to open it. To build the Sphinx sources in
`docs/` locally:

```bash
source .venv/bin/activate
pip install -e '.[docs]'
make -C docs html
```

## Testing

```bash
source .venv/bin/activate
pip install -e '.[test]'
python -m pytest
```

## License

MIT
