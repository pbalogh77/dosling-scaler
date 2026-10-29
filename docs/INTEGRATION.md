# Integration

Keep native/lut and native/vendor/hqx in their relative layout. Link your application to dos_lut and package its dependency dos_lut_hqx. Use the public include directory and either student-v37.bin or its embedded header. Do not share one context concurrently. dl_set_workers selects automatic, serial or requested row workers on Apple; other builds remain serial. Check return codes and allocation sizes.

## macOS

```sh
cmake -S native/lut -B build-macos -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 -DDL_BUILD_TOOLS=OFF
cmake --build build-macos --parallel 4
ctest --test-dir build-macos --output-on-failure
cmake --install build-macos --prefix "$PWD/install-macos"
```

## iOS frameworks

```sh
cmake -S native/lut -B build-ios -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
  -DDL_BUILD_TOOLS=OFF -DBUILD_TESTING=OFF
cmake --build build-ios --config Release --parallel 4
cmake --install build-ios --config Release --prefix "$PWD/install-ios"
```

Use iphonesimulator with suitable architectures for a simulator build. Set your own deployment/signing settings and verify on real supported systems; a compile does not prove oldest-device or thermal performance. Both frameworks must be embedded with suitable loading paths and signing for your app. The iOS module map is installed in the main framework. No app or prebuilt framework is supplied in this release.

## Replacement and licensing

A user can build modified libraries from this source with CMake and use their own lawful development/signing environment. Library separation is preserved; there is no cross-library IPO into the HQx fallback. This is a technical build arrangement, not proof of LGPL compliance for a signed store app. Preserve notices, supply corresponding source and any necessary relinking/installation materials, and avoid terms that prohibit debugging modifications to LGPL components. Assess the actual delivered binaries and distribution terms separately.

## Options

- DL_BUILD_TOOLS=OFF: omit fitting tools and host CLI.
- DL_ENABLE_NEON=OFF: force scalar comparison path for differential checks.
- DL_ENABLE_IPO=OFF: disable runtime interprocedural optimisation.
- DL_SANITIZE=ON: address/undefined behaviour instrumentation with a supported host compiler.
- BUILD_TESTING=OFF: omit portable tests (also omitted when cross compiling).

Pixel aspect ratio, final screen scaling, colour management, input timing and emulator behaviour belong to the caller. This library always returns exact 3× dimensions.
