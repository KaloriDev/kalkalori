# 🔥 KalKalori — Heat Exchanger Open Engine

**KalKalori** is an open-source computational engine for the **design and analysis of tubular heat exchangers**, with liquid or steam on the pressurized tube side and air or gas on the external side.

The project is intended for:

* industrial heat exchanger design,
* academic research and education,
* collaborative development between companies and universities.

KalKalori is released as **free software under the GNU General Public License v3 (GPLv3 only)**.

---

## 🎯 Project Scope

KalKalori focuses on:

* bare-tube and finned-tube heat exchangers,
* air-to-liquid and air-to-steam configurations,
* steady-state thermal and hydraulic calculations,
* transparent, literature-based engineering correlations.

The engine is designed as a **backend library**, independent of any specific UI.
Excel add-ins, web interfaces, and commercial frontends are expected to be developed as **separate projects**.

---

## 🧱 Architecture Overview

```
KalKalori/
├── core/              # Core computational engine (GPLv3)
├── modules/           # Optional GPL extensions (e.g. advanced solvers)
├── data/                 # Validation data, examples (CC-BY)
├── docs/                 # Documentation and theory
├── tests/                # Unit and validation tests
└── LICENSE               # GNU GPL v3
```

Key design principles:

* strict separation between core engine and external interfaces,
* explicit, testable physical models,
* no hidden heuristics or black-box solvers.

---

## ⚙️ Intended Use

KalKalori **may be used commercially**, including:

* internal engineering calculations,
* consulting and design services,
* academic–industrial joint projects.

If KalKalori is **redistributed** (in source or binary form),
the distributor must comply with the terms of the **GPLv3**.

Use of KalKalori for engineering results or reports is permitted without restriction.
No obligation exists to publish calculation results.

---

## 📜 License

KalKalori is licensed under the **GNU General Public License v3 (GPLv3 only)**.

* Commercial use is permitted.
* Redistribution requires providing source code under GPLv3.
* Closed-source redistribution is **not permitted**.

See the `LICENSE` file for the full license text.

---

## Disclaimer

KalKalori is provided **without any warranty**.
Results must be independently verified by qualified engineers
before use in real-world applications.

---

## 🤝 Contributing

Contributions from industry and academia are welcome.

Please read:

* `CONTRIBUTING.md`
* `POLICY_CODE_ACCEPTANCE.md`

before submitting a pull request.

---

© 2025 — KalKalori Project
