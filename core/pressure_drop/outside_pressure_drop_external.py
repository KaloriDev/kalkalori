# KalKalori — Heat Exchanger Open Engine
# GNU GPL v3 only
#
# -------------------------------------------------------------------------
# OUTSIDE PRESSURE DROP – EXTERNAL EULER PROVIDER ADAPTER
# -------------------------------------------------------------------------
#
# This module provides an adapter for external / out-of-process Euler-number
# providers. Its purpose is to let GPL core communicate with a separate
# executable or service without importing proprietary code into the GPL codebase.
#
# The adapter launches an external executable, sends an EulerRequest as JSON
# via stdin, and expects an EulerResult-compatible JSON object on stdout.
#
# Expected JSON input schema (stdin):
# {
#   "Re": 1234.5,
#   "ST_over_D": 1.5,
#   "SL_over_D": 1.25,
#   "layout": "inline",
#   "n_rows": 6,
#   "is_finned": false,
#   "geometry_meta": {...}
# }
#
# Expected JSON output schema (stdout):
# {
#   "Eu": 7.84,
#   "source": "external_vdi_provider",
#   "model": "vdi_external",
#   "validity_note": "optional text"
# }
#
# IMPORTANT:
#   This file does not contain proprietary correlations or data.
#   It is only a transport adapter.
#
# -------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from .outside_pressure_drop import EulerProvider, EulerRequest, EulerResult


class ExternalCliEulerProvider(EulerProvider):
    """
    External Euler provider adapter using subprocess + JSON over stdin/stdout.

    Parameters
    ----------
    executable:
        Path to external executable.
    timeout_s:
        Maximum allowed execution time in seconds.
    extra_args:
        Optional additional CLI arguments passed to the executable.

    Notes
    -----
    The external program must:
      - read one JSON object from stdin
      - write one JSON object to stdout
      - exit with code 0 on success
    """

    def __init__(
        self,
        executable: str,
        *,
        timeout_s: float = 5.0,
        extra_args: list[str] | None = None,
    ) -> None:
        exe_path = Path(executable)

        if not executable:
            raise ValueError("executable must be a non-empty path.")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive.")

        self.executable = str(exe_path)
        self.timeout_s = timeout_s
        self.extra_args = list(extra_args) if extra_args is not None else []

    def evaluate(self, request: EulerRequest) -> EulerResult:
        payload = asdict(request)
        cmd = [self.executable, *self.extra_args]

        try:
            completed = subprocess.run(
                cmd,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"External Euler provider executable not found: {self.executable}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"External Euler provider timed out after {self.timeout_s:.2f} s: "
                f"{self.executable}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Failed to start external Euler provider: {self.executable}"
            ) from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise RuntimeError(
                "External Euler provider returned non-zero exit status "
                f"{completed.returncode}. stderr={stderr!r}"
            )

        stdout = (completed.stdout or "").strip()
        if not stdout:
            raise RuntimeError("External Euler provider returned empty stdout.")

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "External Euler provider returned invalid JSON on stdout."
            ) from exc

        return self._parse_result(data)

    @staticmethod
    def _parse_result(data: object) -> EulerResult:
        if not isinstance(data, dict):
            raise RuntimeError(
                "External Euler provider JSON response must be an object."
            )

        if "Eu" not in data:
            raise RuntimeError(
                "External Euler provider response missing required field: 'Eu'."
            )
        if "source" not in data:
            raise RuntimeError(
                "External Euler provider response missing required field: 'source'."
            )
        if "model" not in data:
            raise RuntimeError(
                "External Euler provider response missing required field: 'model'."
            )

        try:
            eu = float(data["Eu"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "External Euler provider field 'Eu' must be numeric."
            ) from exc

        source = str(data["source"])
        model = str(data["model"])
        validity_note_raw = data.get("validity_note")
        validity_note = None if validity_note_raw is None else str(validity_note_raw)

        euler_basis = str(data.get("euler_basis", "complete_bank"))
        if euler_basis not in ("per_row", "complete_bank"):
            raise RuntimeError(
                "External Euler provider field 'euler_basis' must be 'per_row' "
                "or 'complete_bank'."
            )
        n_rows_effective_raw = data.get("n_rows_effective")
        n_rows_effective = (
            None if n_rows_effective_raw is None else float(n_rows_effective_raw)
        )

        if eu < 0.0:
            raise RuntimeError("External Euler provider returned negative Eu.")

        return EulerResult(
            Eu=eu,
            source=source,
            model=model,
            validity_note=validity_note,
            euler_basis=euler_basis,
            n_rows_effective=n_rows_effective,
        )
