# KalKalori — Code Acceptance and Licensing Policy

This document defines how external code is evaluated
before inclusion in the KalKalori codebase.

---

## 1. Accepted Licenses

The following licenses are **compatible with GPLv3** and may be accepted:

* GNU GPL v3
* GNU LGPL v2.1 / v3 (with restrictions)
* MIT
* BSD 2-Clause / 3-Clause
* Apache License 2.0

All accepted code will be redistributed under **GPLv3**.

---

## 2. Unacceptable Licenses

The following are **not acceptable** for direct inclusion:

* Proprietary / closed-source licenses
* Licenses without an explicit grant (no LICENSE file)
* EPL, MS-PL, custom non-standard licenses

Such code may only be used as **conceptual reference**, not copied.

---

## 3. Handling MIT/BSD/Apache Code

Permissive-licensed code:

* may be included,
* must retain original copyright notices,
* must be clearly documented in `THIRD_PARTY.md`.

Once merged, the combined work is distributed under GPLv3.

---

## 4. GPL Code

GPL-licensed code may be included if:

* license version is **GPLv3-compatible**,
* no additional restrictions are imposed.

---

## 5. External Tools and APIs

External proprietary tools may be:

* invoked via external processes,
* used for validation only.

They must not be linked or embedded into KalKalori.

---

## 6. Final Authority

The KalKalori maintainers reserve the right to:

* reject contributions for legal or technical reasons,
* request relicensing clarification,
* refactor or rewrite contributions if necessary.

---

This policy exists to protect:

* contributors,
* users,
* and the long-term integrity of the KalKalori project.
