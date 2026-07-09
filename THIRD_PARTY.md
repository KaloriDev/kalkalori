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