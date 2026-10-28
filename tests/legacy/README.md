# Frozen reference implementation (aist 0.1.0, main 6eb647c)

`aist_v01/` is a verbatim copy of the package at commit 6eb647c (only `aist.` imports renamed to `aist_v01.`).
The tests use it as the oracle for the agnostic core: every compatibility function must give the same arrays,
bit for bit, as this code. It is not part of the installed package and goes away once `aist_paper` has switched
to the new API.
