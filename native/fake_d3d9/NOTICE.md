# Source and scope

The `d3d9.rs`, `backing.rs`, and `out.rs` implementations are adapted from
`astral4/nokozero`, local reference `third_party/nokozero-20261022/nokozero_hook/src/headless/`.
Upstream author: Kyle Chen. Upstream licensing is **MIT OR Apache-2.0**;
the original license files accompany this crate.

This standalone DLL removes nokozero's IAT installation and game-specific code.
It exports a Direct3DCreate9 replacement and diagnostic counters, retaining
COM resource lifetimes and host-memory texture/buffer locks. Rasterization and
presentation are deliberately absent. TH14's ANM and logical callbacks remain
in the original executable. Unimplemented zero-valued getters return E_NOTIMPL
and increment a diagnostic counter; they are not represented as valid state.

This backend needs original-game validation. It is not a complete Direct3D9
renderer and is not assumed to reproduce rendered pixels.

Render targets and depth surfaces retain their bound COM identities, use four
render-target slots, restore defaults on Reset, and update viewport dimensions
when target zero changes. The default depth format follows presentation parameters.
Replacing, clearing, or resetting a binding releases its retained reference.
Because upstream surfaces retain their parent device, a currently bound surface
can form a reference cycle with that device. This backend is used inside one
owned native process per run; final process/Job cleanup reclaims that bounded
lifetime. It is not advertised as a general renderer with complete COM final-
Release lifetime equivalence. No raw borrowed pointer is substituted for AddRef.
