# Licence, attribution and modifications

Copyright © 2026 Peter Balogh and Dosling scaler contributors for the Dosling-authored portions, distributed under GNU Lesser General Public License version 2.1 or, at your option, any later version. The same grant applies to the distributed model tables, their binary, embedded header and editable JSON representation. No warranty; see LICENSE.

Existing third-party copyrights and notices remain in force. Nothing here relicenses third-party material or grants rights to any games or screenshots.

## HQx / HQ3x

- Upstream: https://github.com/grom358/hqx
- Revision: `a1c7d415f4bb7b947c4c2e47faf220b0148f4bac`
- Maxim Stepin: original HQx algorithm, copyright 2003.
- Cameron Zemek: package maintenance and 24-bit support, copyright 2010.
- François Gannaz: transparency support, as recorded in upstream AUTHORS.
- Licence: LGPL-2.1-or-later; complete text in LICENSE and native/vendor/hqx/COPYING.

The vendored tree retains AUTHORS, README, COPYING and ORIGIN.txt. Local September/October 2026 changes are documented in ORIGIN.txt: signed colour differences, defined signed chroma conversion, and integer RGB-to-YUV conversion replacing the large initialisation table. HQ pattern/blend sources remain available for comparison.

`native/lut/src/hq_block.c` derives directly from HQ3x. The October 17 modification isolates one 3×3 block and permits cached YUV/comparison keys; see native/lut/HQX-NOTICE.txt. It retains the upstream copyright and licence header. The runtime dynamically links its separate `dos_lut_hqx` library; it is not just a benchmark-only dependency.

The other runtime files implement Dosling geometry, table loading and rendering. This initial public source snapshot retains the V49 algorithm/model bytes. Its public CMake test registration omits six corpus-dependent tests; no runtime algorithm changes were made for publication.

## Distribution

Retain applicable notices and provide the LGPL source/materials and modification/relinking rights required for the distribution method you choose. This source release is not a certification that a particular signed app, store agreement or binary package is LGPL-compliant. App packaging must be assessed separately. See docs/INTEGRATION.md.
