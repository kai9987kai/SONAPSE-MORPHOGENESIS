"""Evidence-gated promotion ledger for NEUROWEAVE mechanism presets.

A JSON ledger records candidate mechanism sets, the laboratory receipts that
support them, evaluations, promotions, rollbacks and an append-only history.
Promotion only changes the ledger's own NEUROWEAVE **preset registry**
(``presets['neuroweave_lab_stable']``); it never touches Sonapse defaults,
``SynapseConfig`` or any checkpoint.

Rules
-----
* ``propose(mechanism_set, evidence_receipts)`` stores the candidate together
  with full copies of its receipts.
* ``evaluate(candidate_id, compatibility_tests_passed=...)`` passes only if
  the compatibility flag is ``True`` and every receipt (a) verifies against
  its SHA-256 digest, (b) carries ``evidence_tier='synthetic_abstraction'``,
  (c) has verdict ``'improved'`` whose held-out 95 % CI excludes zero in the
  favourable direction, and (d) tests only mechanisms inside the candidate set.
  A failing candidate is moved to ``archive`` with its receipts and reasons;
  nothing is ever deleted.
* ``promote(candidate_id)`` (approved candidates only) keeps the previous
  stable entry on a stack; ``rollback()`` restores it and archives the
  rolled-back entry.
* Every mutation is written atomically: JSON to a temporary file in the same
  directory, ``fsync``, then ``os.replace``. If the write fails, both the file
  and the in-memory ledger are unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .experiments import verify_receipt
from .spec import EVIDENCE_TIER, MECHANISM_IDS, NeuroWeaveSpecV1

__all__ = ['PromotionLedger', 'LEDGER_SCHEMA', 'STABLE_PRESET']

LEDGER_SCHEMA = 'NeuroWeavePromotionLedgerV1'
STABLE_PRESET = 'neuroweave_lab_stable'


def _empty() -> dict[str, Any]:
    return {'schema': LEDGER_SCHEMA, 'version': 1, 'seq': 0, 'stable': None, 'previous_stable': [],
            'candidates': {}, 'archive': [], 'history': [], 'presets': {}, 'evidence_tier': EVIDENCE_TIER}


def _canon_mechanisms(mechanism_set: Iterable[str]) -> list[str]:
    if isinstance(mechanism_set, (str, bytes)):
        raise ValueError('mechanism_set must be an iterable of mechanism ids')
    ms = list(mechanism_set)
    bad = [m for m in ms if m not in MECHANISM_IDS]
    if bad:
        raise ValueError(f'unknown mechanisms: {bad}')
    if not ms:
        raise ValueError('mechanism_set must not be empty')
    return [m for m in MECHANISM_IDS if m in set(ms)]


def _receipt_check(receipt: Mapping[str, Any], mechanisms: Sequence[str]) -> list[str]:
    """Reasons a receipt fails the promotion gate (empty list = passes)."""
    reasons: list[str] = []
    rid = receipt.get('experiment_id', '?') if isinstance(receipt, Mapping) else '?'
    if not isinstance(receipt, Mapping) or not verify_receipt(receipt):
        return [f'{rid}: receipt digest does not verify']
    if receipt.get('evidence_tier') != EVIDENCE_TIER:
        reasons.append(f'{rid}: evidence_tier must be {EVIDENCE_TIER!r}')
    if receipt.get('verdict') != 'improved':
        reasons.append(f"{rid}: verdict is {receipt.get('verdict')!r}, not 'improved'")
    basis = receipt.get('verdict_basis') or {}
    heldout = basis.get('section') == 'heldout' or str(basis.get('metric', '')).startswith('heldout')
    if not heldout:
        reasons.append(f'{rid}: verdict is not based on a held-out metric')
    comp = (receipt.get('comparisons') or {}).get(basis.get('comparison') or '', {}) or {}
    primary = comp.get('primary') or comp.get('heldout_accuracy') or {}
    ci = primary.get('ci95')
    higher = primary.get('higher_is_better', True)
    if not (isinstance(ci, list) and len(ci) == 2 and all(isinstance(v, (int, float)) for v in ci)):
        reasons.append(f'{rid}: no held-out 95% CI')
    else:
        excludes = ci[0] > 0.0 if higher else ci[1] < 0.0
        if not excludes:
            reasons.append(f'{rid}: held-out 95% CI {ci} does not exclude 0 in the favourable direction')
    tested = set(receipt.get('mechanisms') or [])
    outside = sorted(tested - set(mechanisms))
    if outside:
        reasons.append(f'{rid}: receipt tests mechanisms outside the candidate set: {outside}')
    return reasons


class PromotionLedger:
    """JSON promotion ledger at ``path`` (created on the first mutation)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or data.get('schema') != LEDGER_SCHEMA:
                raise ValueError('not a NEUROWEAVE promotion ledger')
            missing = set(_empty()) - set(data)
            if missing:
                raise ValueError(f'ledger is missing fields: {sorted(missing)}')
            self._state = data
        else:
            self._state = _empty()

    # ------------------------------------------------------------------ persistence
    def _commit(self, new_state: dict[str, Any]) -> None:
        """Atomically persist ``new_state``; on failure nothing (file or memory) changes."""
        text = json.dumps(new_state, sort_keys=True, indent=1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f'{self.path.name}.tmp-{os.getpid()}')
        try:
            with open(tmp, 'w', encoding='utf-8') as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        self._state = new_state

    def _next(self) -> tuple[dict[str, Any], int]:
        st = copy.deepcopy(self._state)
        st['seq'] += 1
        return st, st['seq']

    @staticmethod
    def _log(st: dict[str, Any], seq: int, event: str, **fields: Any) -> None:
        st['history'].append({'seq': seq, 'event': event, **fields})

    # ------------------------------------------------------------------ views
    @property
    def history(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._state['history'])

    @property
    def archive(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._state['archive'])

    @property
    def stable(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._state['stable'])

    @property
    def presets(self) -> dict[str, Any]:
        return copy.deepcopy(self._state['presets'])

    def candidate(self, candidate_id: str) -> dict[str, Any]:
        if candidate_id not in self._state['candidates']:
            raise KeyError(candidate_id)
        return copy.deepcopy(self._state['candidates'][candidate_id])

    def stable_spec(self) -> NeuroWeaveSpecV1 | None:
        """The promoted preset as a (neutral, opt-in) spec, or None."""
        data = self._state['presets'].get(STABLE_PRESET)
        return None if data is None else NeuroWeaveSpecV1.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    # ------------------------------------------------------------------ workflow
    def propose(self, mechanism_set: Iterable[str], evidence_receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        mechanisms = _canon_mechanisms(mechanism_set)
        if isinstance(evidence_receipts, Mapping) or not isinstance(evidence_receipts, Sequence):
            raise ValueError('evidence_receipts must be a sequence of receipts')
        receipts = [copy.deepcopy(dict(r)) for r in evidence_receipts]
        if any(not isinstance(r, dict) for r in receipts):
            raise ValueError('every receipt must be a mapping')
        st, seq = self._next()
        digests = [r.get('digest') for r in receipts]
        cid = hashlib.sha256(json.dumps({'m': mechanisms, 'd': digests, 'seq': seq}, sort_keys=True)
                             .encode('utf-8')).hexdigest()[:16]
        cand = {'id': cid, 'mechanisms': mechanisms, 'receipts': receipts, 'receipt_digests': digests,
                'status': 'proposed', 'proposed_seq': seq, 'evaluation': None}
        st['candidates'][cid] = cand
        self._log(st, seq, 'propose', candidate_id=cid, mechanisms=mechanisms, receipts=len(receipts))
        self._commit(st)
        return copy.deepcopy(cand)

    def evaluate(self, candidate_id: str, *, compatibility_tests_passed: bool) -> dict[str, Any]:
        if candidate_id not in self._state['candidates']:
            raise KeyError(candidate_id)
        cand = self._state['candidates'][candidate_id]
        if cand['status'] not in ('proposed', 'approved'):
            raise ValueError(f"candidate is {cand['status']}; only proposed candidates can be evaluated")
        reasons: list[str] = []
        if compatibility_tests_passed is not True:
            reasons.append('compatibility tests did not pass (compatibility_tests_passed must be True)')
        if not cand['receipts']:
            reasons.append('no evidence receipts')
        for r in cand['receipts']:
            reasons.extend(_receipt_check(r, cand['mechanisms']))
        passed = not reasons
        st, seq = self._next()
        evaluation = {'passed': passed, 'reasons': reasons, 'compatibility_tests_passed': compatibility_tests_passed is True,
                      'receipts_checked': len(cand['receipts']), 'seq': seq}
        c = st['candidates'][candidate_id]
        c['evaluation'] = evaluation
        if passed:
            c['status'] = 'approved'
        else:
            c['status'] = 'failed'
            st['archive'].append({**copy.deepcopy(c), 'archived_seq': seq, 'archive_reason': 'failed_evaluation'})
            del st['candidates'][candidate_id]
        self._log(st, seq, 'evaluate', candidate_id=candidate_id, passed=passed, reasons=reasons)
        self._commit(st)
        return copy.deepcopy(evaluation)

    def promote(self, candidate_id: str) -> dict[str, Any]:
        if candidate_id not in self._state['candidates']:
            raise KeyError(candidate_id)
        cand = self._state['candidates'][candidate_id]
        if cand['status'] != 'approved':
            raise ValueError(f"only approved candidates can be promoted (status: {cand['status']})")
        st, seq = self._next()
        spec = NeuroWeaveSpecV1(mechanisms=tuple(cand['mechanisms'])).to_dict()
        entry = {'candidate_id': candidate_id, 'mechanisms': list(cand['mechanisms']), 'spec': spec,
                 'receipt_digests': list(cand['receipt_digests']), 'promoted_seq': seq}
        if st['stable'] is not None:
            st['previous_stable'].append(st['stable'])
        st['stable'] = entry
        st['presets'][STABLE_PRESET] = spec
        st['candidates'][candidate_id]['status'] = 'promoted'
        self._log(st, seq, 'promote', candidate_id=candidate_id, mechanisms=entry['mechanisms'],
                  previous=None if not st['previous_stable'] else st['previous_stable'][-1]['candidate_id'])
        self._commit(st)
        return copy.deepcopy(entry)

    def rollback(self) -> dict[str, Any] | None:
        """Restore the previous stable entry (``None`` = no preset). Raises if there is nothing to roll back."""
        if self._state['stable'] is None:
            raise ValueError('nothing to roll back')
        st, seq = self._next()
        current = st['stable']
        cid = current['candidate_id']
        cand = st['candidates'].pop(cid, None)
        archived = {**(cand or {'id': cid, 'mechanisms': current['mechanisms']}), 'status': 'rolled_back',
                    'stable_entry': current, 'archived_seq': seq, 'archive_reason': 'rollback'}
        st['archive'].append(archived)
        previous = st['previous_stable'].pop() if st['previous_stable'] else None
        st['stable'] = previous
        if previous is None:
            st['presets'].pop(STABLE_PRESET, None)
        else:
            st['presets'][STABLE_PRESET] = previous['spec']
        self._log(st, seq, 'rollback', rolled_back=cid, restored=None if previous is None else previous['candidate_id'])
        self._commit(st)
        return copy.deepcopy(previous)
