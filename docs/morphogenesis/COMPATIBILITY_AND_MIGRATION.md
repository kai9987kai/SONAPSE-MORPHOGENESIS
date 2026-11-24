# SONAPSE MORPHOGENESIS — Compatibility & Migration Guide

## 1. Backward-Compatibility Guarantees

SONAPSE MORPHOGENESIS extends `Sonapse-LAB` (`0.3.0`) under strict non-destructive additive contracts:

1. **Zero-Impact Default Configuration**:
   - `SynapseConfig.morphogenesis` defaults to `None`.
   - When `morphogenesis is None`, `SynapseConfig.to_dict()` omits the `morphogenesis` key completely, preserving byte-for-byte identical config serialization and checkpoint hashes with pre-Morphogenesis `Sonapse-LAB`.
2. **Legacy Checkpoint Loading**:
   - All existing `synapse-state-v1` checkpoints (including `artifacts/tidal_core.pt` and saved `LabRuntime` checkpoints lacking `morphogenesis_v1`) load cleanly with `morphogenesis` inactive unless additively enabled in zero-gain/observer mode.
   - Original checkpoints and experimental receipts in `artifacts/`, `docs/results/`, and `synapse/_vendor/` are never overwritten or modified.
3. **Zero-Gain Graft Bus Identity**:
   - `MorphogenesisBusChannel` implements `synapse.graft_bus.NeuralGraftProtocol`.
   - When `enabled=False` or `gain == 0.0`, `compute_current(r, ...)` returns `torch.zeros_like(r)` bit-for-bit, guaranteeing $F_{\text{host}+\text{morph}}(x, s; \gamma=0) \equiv F_{\text{host}}(x, s)$.
   - Compatible neural grafts support function-preserving zero-output projection initialization so attaching a graft never destabilizes an already-trained connectome at $t=0$.
4. **Preserved UI, CLI & Simulation Modes**:
   - All existing `python -m synapse ...` CLI subcommands, all 25 existing virtual tasks (including `world4d_tesseract_bypass`), all 156 existing pytest modules, and all 7 existing Workbench tabs (`01`–`07`) remain fully operational and unaltered.
   - Morphogenesis adds:
     - Standalone CLI: `python -m sonapse_morphogenesis ...`
     - Additive `synapse` CLI subcommand: `python -m synapse morphogenesis ...`
     - Additive HTTP API namespace: `/api/morphogenesis/*` on `synapse.server.Workbench`
     - Additive Workbench Tab `08 Morphogenesis Lab` in `synapse/web/index.html` (plus full standalone UI at `/morphogenesis`).
5. **Transactional Reversible Structural Edits**:
   - Every genome/morphology/connectome modification records an atomic before-snapshot digest and inverse delta in the persistent SQLite/JSON registry so any experiment or graft can be cleanly rolled back.

## 2. Schema Migration Rules

- **`morphogenesis-schema-v1`** payloads carry explicit `schema_version: "1.0.0"`.
- Legacy `NEUROWEAVE` (`1.0.0`/`1.1.0`/`1.2.0`), `DSAGSpecV1`, and `synapse-state-v1` payloads are imported via explicit read-only adapters in `sonapse_morphogenesis.core.bridge` without mutating source files.
- Archived checkpoints vs. live accessible behavioral knowledge are tracked separately in `RetentionAuditRecord` (`archived_recoverable` vs. `live_behavioral_retention`).
