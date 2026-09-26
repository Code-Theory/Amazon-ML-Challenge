# Risk register

| Risk | Why it matters | Current control |
|---|---|---|
| Pair-level leakage | Same business can appear on both sides of validation | Entity-group fold manifest and overlap assertions |
| Candidate recall ceiling | Matcher cannot recover a blocked-out true pair | Separate blocking recall report; no validation-positive injection |
| Singleton false merges | Macro F0.5 gives a singleton zero for any false match | Empty predictions allowed; threshold selected inside validation |
| Leaderboard overfitting | Repeated public-score tuning harms final generalization | Nested local validation; leaderboard treated only as operational feedback |
| France domain shift | France is absent from training | Open-set country handling and country-agnostic text features; no hard-coded list |
| External lookup violation | Causes disqualification | Offline pipeline; provenance review for every external model/resource |
| License violation | Final model must satisfy stated license rule | Record model card, license, parameter count, and artifact hash before use |
| Cartesian explosion | Full cross-source comparison may be infeasible | Candidate generation precedes feature/model inference |
| Threshold overfitting | Tuning on reported validation biases the score | Inner-fold out-of-fold threshold selection only |
| Silent file-format errors | Can reject an otherwise valid submission | Strict contracts plus organizer validator before upload |
| Nondeterminism | Prevents reproducibility | Versioned seeds, environment lock, data/config/output hashes |
| Unknown dataset scale | Invalidates premature memory/model choices | Required data audit before finalizing architecture |
| Missing-value false evidence | Two empty fields can look like a perfect match | Missing-safe similarities plus explicit missing indicators |
| Smaller validation retrieval pools | Fold target pools may make top-k blocking easier than full test retrieval | Measure candidate recall versus pool size and stress candidate budgets after the data audit |
| Pair-count dominance | Entities with more candidates can dominate pairwise loss | Normalize sample weights so each Source 1 entity has equal total weight |
| Dependency/runtime drift | Unpinned or incompatible libraries can change behavior | `uv.lock`, metadata, versioned configs, doctor command, and model schema check |
