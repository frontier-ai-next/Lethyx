# Memory Research Framing

Formal set model:

```text
M = D ∪ S ∪ E ∪ A ∪ C
D = directive memory
S = current run state
E = supplemental evidence
A = archive hints
C = reflection candidates
```

Directive set:

```text
B_t = { m ∈ M : surface(m)=directive ∧ lifecycle(m)=active ∧ trust(m)∈VerifiedTrust ∧ applies_to(m,t) ∧ not expired(m) ∧ not contradicted(m) }
```

Retrieval set:

```text
R_t = { m ∈ M : surface(m) ∈ {supplemental_evidence, archive_hint} ∧ relevant(m,t) }
```

Invariant:

```text
R_t ∩ B_t = ∅
```

If retrieval-derived items are assigned supplemental/archive/candidate surfaces and the renderer injects directive memory only when accepted, unverified retrieval-derived memory cannot enter directive prompt sections without promotion.

Utility objective:

```text
S* = argmax_S Σ U(m,t) subject to Σ tokens(m) <= budget and every m is eligible directive memory
```
