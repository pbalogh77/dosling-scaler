# Validation scope

The initial public snapshot is runtime V49 / quality-v49-concave-bevel with student-v37. Public runtime source and model bytes match the retained development snapshot. The public CMake build has **19 portable native tests** covering runtime/API behaviour, HQ3x fallback conformance, embedded-model use, fitting exceptions and synthetic geometry cases.

The original local suite has 25 native tests. Six tests containing game-derived or provenance-unclear pixel fixtures are not redistributed: aa_cap, aa_consensus, thin_corner, mixed_corner, general_notch and local_range. Their private inputs and results are retained locally. They are not counted as passing public tests. The separate fixed 32-image / 96-variant local gallery was regenerated for this publication, preserving feedback; game screenshots are not included in the public project.

Local release checks are recorded in RELEASE_VALIDATION.json. GitHub CI builds and tests public source on macOS and Linux. A configured workflow is not evidence of a completed CI run; consult actual run results. No universal visual-quality, cross-platform bit-identity, iPhone throughput, App Store acceptance or legal-compliance guarantee is made.

Future changes should keep deterministic input/output checks, compare scalar and NEON paths on supported ARM hosts, preserve fallback conformance, check memory/UB with sanitizers, and review both text and graphics on legally usable material. Metrics alone do not establish visual quality. New pixel revisions should receive explicit versioning and fresh comparison evidence.
