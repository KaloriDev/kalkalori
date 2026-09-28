## Third-Party Software

This project uses the following third-party libraries:

### PsychroLib
- License: MIT
- Repository: https://github.com/psychrometrics/psychrolib
- Purpose: Psychrometric property calculations for moist air

PsychroLib is used as an external dependency and is not modified or redistributed
as part of the KalKalori source code.

### iapws
- License: GPLv3
- Repository: https://github.com/jjgomera/iapws
- Purpose: IAPWS-IF97 water/steam property calculations

iapws is used as an external dependency and is not modified or redistributed
as part of the KalKalori source code.

### CoolProp
- License: MIT
- Repository: https://github.com/CoolProp/CoolProp
- Purpose: Optional thermophysical property backend for pure fluids and mixtures

CoolProp is used as an optional external dependency and is not modified or
redistributed as part of the KalKalori source code.

### REFPROP

- License: proprietary / NIST
- Purpose: Optional high-accuracy thermophysical property backend

REFPROP is not a dependency of KalKalori and is not distributed with this
project. If selected as a CoolProp backend, it must be installed, licensed,
and configured locally by the user.

### EnergyPlus equation reference (verification stage)

The isolated Elmahdy-Mitalas kernel is an original GPL-3.0-only implementation
of documented equations, referenced to EnergyPlus v25.2.0 commit
`cf7368216c73c43181e057fa33b479c4e0c86df0`. Its four-condition upstream license
is not treated as BSD-3-Clause or relicensed. The optional native verification
tool reads separately obtained, hash-checked sources, retains their notices
in generated C++, and runs them in a separate process. No generated native
source/binary is distributed or linked with the production package.
See [the equation and interface map](docs/elmahdy_mitalas.md).
