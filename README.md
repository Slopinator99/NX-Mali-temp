# eden_mali_driver

A Vulkan driver for Arm Mali GPUs (kbase kernel driver, command-stream
frontend), aimed at running **Nintendo Switch games in the
[Eden](https://git.eden-emu.dev/eden-emu/eden) emulator** on Android devices
and handhelds that ship Mali GPUs. It is installed like any custom GPU driver
in Eden: a zip with the library and a `meta.json`.

## Origin

This project is a fork of [bmdhacks/malisx2](https://github.com/bmdhacks/malisx2),
an independent, clean-room Vulkan driver for Mali that was first written for a
different emulator. What we build on is its foundation: a panvk-style driver on
Mesa's shared Vulkan runtime and the Panfrost/kraid shader compiler, its own
kbase/CSF queue, memory and command-stream layer, and the Android HAL
packaging. What this fork changes is the target: the work here is making the
driver correct and complete for what Eden's Vulkan renderer needs (indirect
draws, mutable-format views, Maxwell-style topologies, queries and conditional
rendering, ...), rather than the entry points of the original application.
Entry points the driver does not implement yet log and fail.

Target hardware so far: Mali-G615 (Mesa architecture v11, CSF), tested on an
Anbernic RG 557 (MediaTek MT6897, Android 14).

## Testing

Correctness is driven by [edenprobe](https://github.com/scarletnx/edenprobe),
a homebrew program of hardware probes whose cases are judged against what a
real console does. The workflow: run its graphics suites in Eden on the stock
Arm driver (the reference) and on this driver, diff the reports case by case,
and fix the cases that fail only here. Fixes found this way so far:
format-mismatched image views, and `vkCmdDrawIndirect` /
`vkCmdDrawIndexedIndirect`.

## Status

Work in progress. Open areas, from the edenprobe diff: quads and some
index-buffer cases, conditional rendering and query results, stencil/early-Z
with shader depth writes, bindless textures, a few formats. There is also an
unresolved rare single-frame flicker; the diagnostic code for it is still
compiled in.

## Building

The driver builds against Mesa sources checked out at `mesa/`: branch
`malisx2` of [bmdhacks/armsx2-turnip](https://github.com/bmdhacks/armsx2-turnip)
at the commit `meson.build` pins (`mesa_commit`). Only the sources are used;
this repository's `mesa-glue/` builds the parts the driver needs.

    git init mesa
    git -C mesa fetch --depth 1 https://github.com/bmdhacks/armsx2-turnip.git <mesa_commit>
    git -C mesa checkout FETCH_HEAD

Host (Linux):

    meson setup build
    ninja -C build

Needs meson >= 1.7, ninja, a C/C++ compiler, rustc >= 1.85, bindgen >= 0.71.1
(not 0.72.0, 0.73.0 or 0.73.1) and Python 3 with mako and PyYAML.
`meson setup` downloads eight small Rust crates (`subprojects/`).

Android (arm64, API level 29) is a cross build with the NDK r28c at
`~/Android/Sdk/ndk/28.2.13676358` (see `cross/android-aarch64.ini`):

    meson setup build-android --cross-file cross/android-aarch64.ini
    ninja -C build-android
    llvm-strip -o libvulkan_malisx2.so build-android/src/libvulkan_malisx2.so

The result is a Vulkan HAL module (`HMI`) with 16 KiB-aligned segments,
needing only `libc`, `libm`, `libdl` and `liblog`. Eden loads it as a custom
GPU driver: the library plus a `meta.json` in a zip.

ROCKNIX (arm64 Linux, RK3566/RK3568 devices running ROCKNIX's
`mali_kbase`) is a cross build against the toolchain of a ROCKNIX build
tree, with the build machine's clang and lld. Name the tree's toolchain
directory in a machine file of your own and pass it first (see
`cross/rocknix-aarch64.ini`):

    # ~/rocknix-toolchain.ini
    [constants]
    toolchain = '/path/to/distribution/build.ROCKNIX-RK3566.aarch64/toolchain'

    meson setup build-rocknix \
        --cross-file ~/rocknix-toolchain.ini --cross-file cross/rocknix-aarch64.ini \
        --native-file ~/rocknix-toolchain.ini --native-file cross/rocknix-native.ini
    ninja -C build-rocknix

With libclang 22, bindgen 0.71.x generates broken bindings; use 0.72.1.

On Linux the driver presents `VK_KHR_display` swapchains itself, through
Mesa's WSI code built into the library. Wayland swapchains come from Arm's
[vulkan-wsi-layer](https://gitlab.freedesktop.org/mesa/vulkan-wsi-layer)
(MIT licensed), a separate implicit Vulkan layer that the system installs,
as ROCKNIX does. The layer allocates linear dma-bufs from a dma-buf heap and
imports them into the driver (`VK_EXT_image_drm_format_modifier` with
`DRM_FORMAT_MOD_LINEAR`, `VK_EXT_external_memory_dma_buf`); none of its code
is part of this repository or linked into the library.

## Measurement

`MALISX2_MEASURE=timing` (or `draws`, `csf`, `both`, `shaders`; on Android
the property `debug.malisx2.measure`) makes the driver write GPU timestamps
per render pass, dispatch and draw, captures of the submitted command
streams, or the shaders it compiles.

## Rules

- Nothing decompiled goes in: no decompiled code, no Arm binaries.
- License: MIT (`LICENSE`). Mesa is not part of this repository and keeps
  its own licenses.
