# Dosling scaler

An experimental, CPU-based **3× pixel-art scaler** developed for [Dosling](https://github.com/pbalogh77/dosling). It combines local geometry and anti-aliasing rules, a compact lookup model and an **HQ3x-derived fallback** to reconstruct low-resolution game frames.

Current source release: **V49**, `quality-v49-concave-bevel`, paired with the **student-v37** model (30,238 bytes). Runtime and model version numbers intentionally differ.

## Build and test

Requires CMake 3.18+, a C/C++17 compiler, and no game files or external AI service.

```sh
cmake -S native/lut -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/install"
```

The build produces `dos_lut` and its separate `dos_lut_hqx` shared library, examples, offline fitting tools and 19 portable native tests. Apple builds can use ARM NEON and GCD row workers; other platforms have a scalar path. See [integration](docs/INTEGRATION.md), [model editing](docs/MODEL.md) and [validation scope](docs/VALIDATION.md).

## Use

The public API is C, with C++ internals. Create a context once, reuse it across frames, then destroy it. RGB24 and BGRA8 input produce exactly three times the width and height. BGRA alpha is ignored on input and written as 255. A context owns scratch buffers and accepts only one caller at a time; separate contexts may run concurrently.

```c
#include "dosling_scaler.h"
#include "student-v37.h"

dl_context *ctx = dl_create_from_memory(dos_lut_student_v37,
                                        dos_lut_student_v37_size);
/* Check ctx. Allocate output for (width * 3) by (height * 3). */
int status = dl_scale3_bgra(ctx, input, width, height, input_stride,
                            output, output_stride, NULL, 0, NULL);
/* Check status before using output. Reuse ctx for subsequent frames. */
dl_destroy(ctx);
```

`dl_create("student-v37.bin")` is the file-based alternative. Optional rectangles are caller-provided text hints, not OCR. Production frames can be rendered without hints. The scaler does not correct pixel aspect ratio itself: presentation is the caller's responsibility.

## What it does and does not do

- Uses source-supported corner, join and coverage rules, 3×3 lookup tables, flagged 5×5 exceptions and HQ3x fallback.
- Keeps each output block within its immediate 3×3 source neighbourhood's per-channel RGB limits.
- Runs offline on the CPU; no remote API, OCR, generative model, Metal compute or Neural Engine is required.
- Does not implement de-dithering or arbitrary output scales.
- Is experimental: synthetic tests and local visual comparisons do not establish universal visual improvement or device performance.

## Licence and credits

**LGPL-2.1-or-later**, including the Dosling-authored code and distributed model tables. See [LICENSE](LICENSE) and [NOTICE](NOTICE.md). This is not a permissively licensed or licence-free replacement for HQ3x.

HQx was created by **Maxim Stepin**; **Cameron Zemek** maintained the package and added 24-bit colour support; **François Gannaz** added transparency support. The project includes and modifies code from [grom358/hqx](https://github.com/grom358/hqx), pinned at `a1c7d415f4bb7b947c4c2e47faf220b0148f4bac`. The fallback contains derived HQ3x code, rather than merely taking inspiration from it. Original notices and modification records are retained.

**AI-assisted development:** OpenAI ChatGPT/Codex was used extensively for research, implementation, debugging, experiments and documentation under Peter Balogh's direction. Read [AI_DEVELOPMENT.md](AI_DEVELOPMENT.md) for scope and limitations. AI assistance does not remove upstream attribution or licensing obligations.

The public repository excludes private game screenshots, game binaries, feedback and historical research outputs. Its published source and editable model are sufficient to build and change the runtime. It does not reproduce the complete historical training/review campaign.
