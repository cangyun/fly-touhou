//! Process-local host-memory D3D9 COM for the privately isolated TH14 process.
//! Adapted from astral4/nokozero; see NOTICE.md and dual licenses.
use std::sync::atomic::{AtomicU32, Ordering};
static UNSUPPORTED_GETTERS: AtomicU32 = AtomicU32::new(0);
static CREATED_DEVICES: AtomicU32 = AtomicU32::new(0);

macro_rules! fatal {
    ($($arg:tt)*) => {{
        crate::diagnostic(&format!("fatal: {}", format_args!($($arg)*)));
        std::process::abort()
    }};
}
mod backing;
mod out;
mod d3d9;
mod state;

pub(crate) fn unsupported_getter() -> windows::core::Result<()> {
    unsupported_call("uninitialized state getter")
}

pub(crate) fn unsupported_call<T>(name: &str) -> windows::core::Result<T> {
    UNSUPPORTED_GETTERS.fetch_add(1, Ordering::Relaxed);
    diagnostic(&format!("unsupported {name} (E_NOTIMPL)"));
    Err(windows::core::Error::from_hresult(windows::Win32::Foundation::E_NOTIMPL))
}

pub(crate) fn diagnostic(message: &str) {
    use std::io::Write;
    if let Some(folder) = std::env::var_os("APPDATA") {
        let path = std::path::PathBuf::from(folder).join("fly_touhou_fake_d3d9.log");
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(file, "{message}");
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn FlyTouhouDirect3DCreate9(sdk_version: u32) -> *mut std::ffi::c_void {
    unsafe { d3d9::create(sdk_version) }
}

#[unsafe(no_mangle)]
pub extern "system" fn FlyTouhouFakeD3D9Audit(index: u32) -> u32 {
    match index {
        0 => 1, // schema version
        1 => CREATED_DEVICES.load(Ordering::Relaxed),
        2 => UNSUPPORTED_GETTERS.load(Ordering::Relaxed),
        _ => u32::MAX,
    }
}
