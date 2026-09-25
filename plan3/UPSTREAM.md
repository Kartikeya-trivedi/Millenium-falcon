# Imported baseline

The sibling plan1/ package was imported from:

- Repository: https://github.com/sohambuilds/icmliclrneuripsaaaiemlnlpmaxxing
- Commit: 2691386 (plan 1 complete).
- Published copies include non-functional wording edits in comments and docstrings. Normalization, feature, split and metric logic are retained.

The supplied repository excludes work/ and output/. This repository includes source, not the teammate's model or data artifacts. Preparation builds the split manifest and normalized data for the new experiment; it does not retrain the first-generation matcher. Its reported score is accepted as supplied evidence.

Run preparation from this checkout on the server. Prepared metadata from a different checkout may contain different source hashes even when only docstrings differ. Preparation records hashes of every imported Python file and the input datasets; incompatible artifacts are rejected.

Plan 3 writes only its own artifacts. Its scripts do not invoke Plan 1's work/output paths. The Plan 1 source is retained as the regression reference for normalization, splits, features and metric semantics.
