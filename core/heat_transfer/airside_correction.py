# SPDX-License-Identifier: GPL-3.0-only
"""Small explicit correction contract for physical film h and pressure estimates.

Providers are dependency-injected objects, never discovered by name. They
must be deterministic for the same frozen baseline/geometry/state. Exceptions
propagate; invalid responses never trigger a fallback. No provider is identity.
"""
from dataclasses import dataclass
import math
from typing import Protocol, Any


@dataclass(frozen=True)
class AirsideCorrection:
    provenance: str = 'identity'
    h_film_multiplier: float | None = None
    h_film_replacement: float | None = None
    pressure_drop_multiplier: float | None = None
    pressure_drop_replacement: float | None = None

    def __post_init__(self):
        if not isinstance(self.provenance,str) or not self.provenance.strip():
            raise ValueError('Correction provenance must be nonempty.')
        for channel in ('h_film','pressure_drop'):
            multiplier=getattr(self,channel+'_multiplier')
            replacement=getattr(self,channel+'_replacement')
            if multiplier is not None and replacement is not None:
                raise ValueError(f'Choose a multiplier or replacement for {channel}, not both.')
            for value in (multiplier,replacement):
                if value is not None and (isinstance(value,bool) or not math.isfinite(value) or value<=0):
                    raise ValueError(f'{channel} correction must be finite and positive.')

    def apply(self, h_film, pressure_drop):
        values=[]
        for channel,base in (('h_film',h_film),('pressure_drop',pressure_drop)):
            replacement=getattr(self,channel+'_replacement')
            multiplier=getattr(self,channel+'_multiplier')
            value=replacement if replacement is not None else base*(1. if multiplier is None else multiplier)
            if not math.isfinite(value) or value<=0:
                raise ValueError(f'Corrected {channel} must be finite and positive.')
            values.append(value)
        return tuple(values)


class AirsideCorrectionProvider(Protocol):
    def correct(self, base_result: Any, *, bundle: Any, props: Any) -> AirsideCorrection:
        """Return corrections at the supplied constitutive state, before eta."""
        ...


@dataclass(frozen=True)
class MultiplicativeAirsideCorrection:
    """Explicit, independent engineering multipliers using the existing contract.

    No geometry/count-to-factor law is implied. The model applies physical h
    before fin efficiency and pressure to the final source estimate.
    """
    h_film_multiplier: float = 1.0
    pressure_drop_multiplier: float = 1.0
    provenance: str = 'explicit_multiplicative_adjustment'

    def __post_init__(self):
        if self.h_film_multiplier is None or self.pressure_drop_multiplier is None:
            raise ValueError('Multipliers must be finite and positive, not None.')
        self.correct(None, bundle=None, props=None)  # Shared correction validation.

    def correct(self, base_result, *, bundle, props) -> AirsideCorrection:
        return AirsideCorrection(provenance=self.provenance,
            h_film_multiplier=self.h_film_multiplier,
            pressure_drop_multiplier=self.pressure_drop_multiplier)
