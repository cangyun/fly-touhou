//! Per-device Set/Get state for the host-memory D3D9 implementation.
//!
//! This is not a claim of complete D3D9 equivalence. Values without a documented
//! implemented default must first be set; an unknown Get returns E_NOTIMPL and
//! increments the crate audit counter without overwriting the caller's output.
//! Reference contracts:
//! https://learn.microsoft.com/en-us/windows/win32/direct3d9/d3dsamplerstatetype
//! https://learn.microsoft.com/en-us/windows/win32/direct3d9/d3dtexturestagestatetype
//! https://learn.microsoft.com/en-us/windows/win32/direct3d9/d3dtransformstatetype
//! https://learn.microsoft.com/en-us/windows/win32/api/d3d9/nf-d3d9-idirect3ddevice9-multiplytransform
//! https://learn.microsoft.com/en-us/windows/win32/api/d3d9/nf-d3d9-idirect3ddevice9-setviewport

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};
use windows::core::{Error, HRESULT, Result};
use windows::Win32::Foundation::E_FAIL;
use windows::Win32::Graphics::Direct3D9::D3DVIEWPORT9;
use windows_numerics::Matrix4x4;

const INVALID_CALL: HRESULT = HRESULT(0x8876_086C_u32 as i32);

#[rustfmt::skip]
const IDENTITY: Matrix4x4 = Matrix4x4 {
    M11:1., M12:0., M13:0., M14:0.,
    M21:0., M22:1., M23:0., M24:0.,
    M31:0., M32:0., M33:1., M34:0.,
    M41:0., M42:0., M43:0., M44:1.,
};

fn invalid() -> Error { Error::from_hresult(INVALID_CALL) }

unsafe fn input<T: Copy>(pointer: *const T) -> Result<T> {
    if pointer.is_null() { return Err(invalid()); }
    Ok(unsafe { pointer.read() })
}

unsafe fn output<T>(pointer: *mut T, value: Option<T>, missing_name: &str) -> Result<()> {
    if pointer.is_null() { return Err(invalid()); }
    match value {
        Some(value) => { unsafe { pointer.write(value) }; Ok(()) }
        None => crate::unsupported_call(missing_name),
    }
}

fn transform_key(key: i32) -> bool {
    matches!(key, 2 | 3 | 16..=23 | 256..=511)
}

fn sampler_key(sampler: u32, key: i32) -> bool {
    matches!(sampler, 0..=15 | 256..=260) && (1..=13).contains(&key)
}

fn stage_key(stage: u32, key: i32) -> bool {
    stage < 8 && matches!(key, 1..=11 | 22..=24 | 26..=28 | 32)
}

// These are D3D9's documented sampler defaults, not blanket zero substitutes.
fn sampler_default(key: i32) -> u32 {
    match key {
        1..=3 | 5 | 6 | 10 => 1, // WRAP; POINT; MaxAnisotropy=1.
        4 | 7..=9 | 11..=13 => 0,
        _ => unreachable!("validated sampler key"),
    }
}

fn stage_default(stage: u32, key: i32) -> Option<u32> {
    match key {
        1 => Some(if stage == 0 { 4 } else { 1 }), // MODULATE / DISABLE.
        4 => Some(if stage == 0 { 2 } else { 1 }), // SELECTARG1 / DISABLE.
        2 => Some(2), // COLORARG1 = TEXTURE.
        3 | 6 | 26..=28 => Some(1), // CURRENT.
        7..=10 | 22..=24 => Some(0),
        11 => Some(stage),
        // ALPHAARG1 can depend on the bound texture. CONSTANT's default is
        // deliberately not inferred here. Set values always take precedence.
        _ => None,
    }
}

struct State {
    render: HashMap<i32, u32>,
    transforms: HashMap<i32, Matrix4x4>,
    viewport: D3DVIEWPORT9,
    target_width: u32,
    target_height: u32,
    fvf: Option<u32>,
    sampler: HashMap<(u32, i32), u32>,
    texture_stage: HashMap<(u32, i32), u32>,
}

impl State {
    fn new(width: u32, height: u32) -> Self {
        Self {
            render: HashMap::new(), transforms: HashMap::from([(2, IDENTITY), (3, IDENTITY)]),
            viewport: D3DVIEWPORT9 { X:0, Y:0, Width:width, Height:height, MinZ:0., MaxZ:1. },
            target_width:width, target_height:height, fvf:None,
            sampler:HashMap::new(), texture_stage:HashMap::new(),
        }
    }
}

pub(crate) struct DeviceState(Mutex<State>);

impl DeviceState {
    pub(crate) fn new(width: u32, height: u32) -> Self {
        Self(Mutex::new(State::new(width, height)))
    }

    fn lock(&self) -> Result<MutexGuard<'_, State>> {
        self.0.lock().map_err(|_| Error::from_hresult(E_FAIL))
    }

    pub(crate) fn reset(&self, width: u32, height: u32) -> Result<()> {
        *self.lock()? = State::new(width, height);
        Ok(())
    }

    /// A successful SetRenderTarget(0, ...) resets the viewport to that surface.
    pub(crate) fn render_target_size(&self, width: u32, height: u32) -> Result<()> {
        let mut state = self.lock()?;
        state.target_width = width; state.target_height = height;
        state.viewport = D3DVIEWPORT9 { X:0, Y:0, Width:width, Height:height, MinZ:0., MaxZ:1. };
        Ok(())
    }

    pub(crate) fn render_set(&self, key: i32, value: u32) -> Result<()> {
        if key <= 0 || key == i32::MAX { return Err(invalid()); }
        self.lock()?.render.insert(key, value);
        Ok(())
    }

    pub(crate) unsafe fn render_get(&self, key: i32, out: *mut u32) -> Result<()> {
        unsafe { output(out, self.lock()?.render.get(&key).copied(), "GetRenderState before implemented initialization") }
    }

    pub(crate) unsafe fn transform_set(&self, key: i32, matrix: *const Matrix4x4) -> Result<()> {
        if !transform_key(key) { return Err(invalid()); }
        let matrix = unsafe { input(matrix)? };
        self.lock()?.transforms.insert(key, matrix);
        Ok(())
    }

    pub(crate) unsafe fn transform_get(&self, key: i32, out: *mut Matrix4x4) -> Result<()> {
        if !transform_key(key) { return Err(invalid()); }
        unsafe { output(out, self.lock()?.transforms.get(&key).copied(), "GetTransform before implemented initialization") }
    }

    pub(crate) unsafe fn transform_multiply(&self, key: i32, matrix: *const Matrix4x4) -> Result<()> {
        if !transform_key(key) { return Err(invalid()); }
        let incoming = unsafe { input(matrix)? };
        let mut state = self.lock()?;
        let Some(previous) = state.transforms.get(&key).copied() else {
            return crate::unsupported_call("MultiplyTransform on an uninitialized transform");
        };
        // D3D9 specifies pMatrix times State, not State times pMatrix.
        state.transforms.insert(key, multiply(incoming, previous));
        Ok(())
    }

    pub(crate) unsafe fn viewport_set(&self, viewport: *const D3DVIEWPORT9) -> Result<()> {
        let viewport = unsafe { input(viewport)? };
        let mut state = self.lock()?;
        if viewport.Width == 0 || viewport.Height == 0
            || viewport.X.checked_add(viewport.Width).is_none_or(|end| end > state.target_width)
            || viewport.Y.checked_add(viewport.Height).is_none_or(|end| end > state.target_height)
            || !viewport.MinZ.is_finite() || !viewport.MaxZ.is_finite()
            || !(0.0..=1.0).contains(&viewport.MinZ) || !(0.0..=1.0).contains(&viewport.MaxZ)
        { return Err(invalid()); }
        state.viewport = viewport;
        Ok(())
    }

    pub(crate) unsafe fn viewport_get(&self, out: *mut D3DVIEWPORT9) -> Result<()> {
        unsafe { output(out, Some(self.lock()?.viewport), "GetViewport") }
    }

    pub(crate) fn fvf_set(&self, value: u32) -> Result<()> {
        self.lock()?.fvf = Some(value);
        Ok(())
    }

    pub(crate) unsafe fn fvf_get(&self, out: *mut u32) -> Result<()> {
        unsafe { output(out, self.lock()?.fvf, "GetFVF before SetFVF") }
    }

    pub(crate) fn sampler_set(&self, sampler: u32, key: i32, value: u32) -> Result<()> {
        if !sampler_key(sampler, key) { return Err(invalid()); }
        self.lock()?.sampler.insert((sampler, key), value);
        Ok(())
    }

    pub(crate) unsafe fn sampler_get(&self, sampler: u32, key: i32, out: *mut u32) -> Result<()> {
        if !sampler_key(sampler, key) { return Err(invalid()); }
        let value = self.lock()?.sampler.get(&(sampler, key)).copied().unwrap_or_else(|| sampler_default(key));
        unsafe { output(out, Some(value), "GetSamplerState") }
    }

    pub(crate) fn texture_stage_set(&self, stage: u32, key: i32, value: u32) -> Result<()> {
        if !stage_key(stage, key) { return Err(invalid()); }
        self.lock()?.texture_stage.insert((stage, key), value);
        Ok(())
    }

    pub(crate) unsafe fn texture_stage_get(&self, stage: u32, key: i32, out: *mut u32) -> Result<()> {
        if !stage_key(stage, key) { return Err(invalid()); }
        let value = self.lock()?.texture_stage.get(&(stage, key)).copied().or_else(|| stage_default(stage, key));
        unsafe { output(out, value, "GetTextureStageState before implemented initialization") }
    }
}

fn multiply(a: Matrix4x4, b: Matrix4x4) -> Matrix4x4 {
    let rows = |m:Matrix4x4| [[m.M11,m.M12,m.M13,m.M14],[m.M21,m.M22,m.M23,m.M24],
                            [m.M31,m.M32,m.M33,m.M34],[m.M41,m.M42,m.M43,m.M44]];
    let (a,b) = (rows(a),rows(b)); let mut c = [[0.;4];4];
    for i in 0..4 { for j in 0..4 { for k in 0..4 { c[i][j] += a[i][k]*b[k][j]; } } }
    Matrix4x4 {
        M11:c[0][0],M12:c[0][1],M13:c[0][2],M14:c[0][3],
        M21:c[1][0],M22:c[1][1],M23:c[1][2],M24:c[1][3],
        M31:c[2][0],M32:c[2][1],M33:c[2][2],M34:c[2][3],
        M41:c[3][0],M42:c[3][1],M43:c[3][2],M44:c[3][3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::E_NOTIMPL;

    #[test]
    fn scalar_states_roundtrip_without_cross_device_or_stage_aliasing() {
        let first=DeviceState::new(640,480); let second=DeviceState::new(640,480);
        first.render_set(27,0x76543210).unwrap(); first.fvf_set(0x144).unwrap();
        first.sampler_set(0,5,2).unwrap(); first.sampler_set(1,5,3).unwrap();
        first.texture_stage_set(0,5,7).unwrap(); first.texture_stage_set(1,5,8).unwrap();
        let mut out=0;
        unsafe {
            first.render_get(27,&mut out).unwrap(); assert_eq!(out,0x76543210);
            first.fvf_get(&mut out).unwrap(); assert_eq!(out,0x144);
            first.sampler_get(0,5,&mut out).unwrap(); assert_eq!(out,2);
            first.sampler_get(1,5,&mut out).unwrap(); assert_eq!(out,3);
            first.texture_stage_get(0,5,&mut out).unwrap(); assert_eq!(out,7);
            first.texture_stage_get(1,5,&mut out).unwrap(); assert_eq!(out,8);
            assert_eq!(second.render_get(27,&mut out).unwrap_err().code(),E_NOTIMPL);
            assert_eq!(out,8); // Failed getters do not fabricate output.
        }
    }

    #[test]
    fn supported_defaults_and_reset_are_explicit() {
        let device=DeviceState::new(640,480); let mut out=99;
        unsafe {
            device.sampler_get(0,1,&mut out).unwrap(); assert_eq!(out,1);
            device.sampler_get(260,7,&mut out).unwrap(); assert_eq!(out,0);
            device.texture_stage_get(0,1,&mut out).unwrap(); assert_eq!(out,4);
            device.texture_stage_get(3,1,&mut out).unwrap(); assert_eq!(out,1);
            device.texture_stage_get(3,11,&mut out).unwrap(); assert_eq!(out,3);
            assert_eq!(device.texture_stage_get(0,5,&mut out).unwrap_err().code(),E_NOTIMPL);
            device.render_set(27,7).unwrap(); device.fvf_set(0x144).unwrap();
            device.reset(800,600).unwrap();
            assert_eq!(device.render_get(27,&mut out).unwrap_err().code(),E_NOTIMPL);
            assert_eq!(device.fvf_get(&mut out).unwrap_err().code(),E_NOTIMPL);
            let mut viewport=D3DVIEWPORT9::default(); device.viewport_get(&mut viewport).unwrap();
            assert_eq!((viewport.Width,viewport.Height),(800,600));
        }
    }

    #[test]
    fn transform_roundtrip_and_multiply_order_match_d3d9() {
        let device=DeviceState::new(640,480); let mut out=IDENTITY;
        let scale=Matrix4x4 {M11:2.,M22:3.,..IDENTITY};
        let translate=Matrix4x4 {M41:5.,M42:7.,..IDENTITY};
        unsafe {
            device.transform_get(2,&mut out).unwrap(); assert_eq!(out.M11,1.);
            assert_eq!(device.transform_get(256,&mut out).unwrap_err().code(),E_NOTIMPL);
            device.transform_set(256,&scale).unwrap();
            device.transform_get(256,&mut out).unwrap(); assert_eq!(out.M22,3.);
            device.transform_multiply(256,&translate).unwrap();
            device.transform_get(256,&mut out).unwrap();
            assert_eq!(out.M41,10.); assert_eq!(out.M42,21.); // translate * scale.
            assert_eq!(out.M11,2.); assert_eq!(out.M22,3.);
        }
    }

    #[test]
    fn viewport_roundtrip_target_reset_and_bounds() {
        let device=DeviceState::new(640,480); let mut out=D3DVIEWPORT9::default();
        let viewport=D3DVIEWPORT9 {X:20,Y:30,Width:100,Height:200,MinZ:0.1,MaxZ:0.8};
        unsafe {
            device.viewport_set(&viewport).unwrap(); device.viewport_get(&mut out).unwrap();
            assert_eq!(out.X,20); assert_eq!(out.MaxZ,0.8);
            let bad=D3DVIEWPORT9 {X:u32::MAX,..viewport};
            assert_eq!(device.viewport_set(&bad).unwrap_err().code(),INVALID_CALL);
            device.viewport_get(&mut out).unwrap(); assert_eq!(out.X,20);
            device.render_target_size(1024,768).unwrap(); device.viewport_get(&mut out).unwrap();
            assert_eq!((out.X,out.Width,out.Height),(0,1024,768));
        }
    }

    #[test]
    fn invalid_pointers_and_indices_fail_without_writing() {
        let device=DeviceState::new(640,480); let mut out=42;
        unsafe {
            assert_eq!(device.fvf_get(std::ptr::null_mut()).unwrap_err().code(),INVALID_CALL);
            assert_eq!(device.transform_set(2,std::ptr::null()).unwrap_err().code(),INVALID_CALL);
            assert_eq!(device.viewport_set(std::ptr::null()).unwrap_err().code(),INVALID_CALL);
            assert_eq!(device.sampler_get(16,1,&mut out).unwrap_err().code(),INVALID_CALL);
            assert_eq!(device.texture_stage_get(8,1,&mut out).unwrap_err().code(),INVALID_CALL);
        }
        assert_eq!(out,42);
    }
}
