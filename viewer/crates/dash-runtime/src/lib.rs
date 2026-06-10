//! Reusable, viewer-agnostic native bridge to a DASH (PyTorch/CUDA) sidecar.
//!
//! The sidecar loads a trained DASH model once and, for a normalized time
//! `t in [0, 1)`, returns a packed binary frame of deformed Gaussians whose byte
//! layout matches the renderer's `Gaussian3d` ABI (see `docs/ARCHITECTURE.md`).
//!
//! This crate owns only the **transport**: sidecar process lifecycle, the
//! length-prefixed TCP protocol, and stride/size validation. It deliberately
//! does *not* interpret the bytes, so any wgpu Gaussian-splatting viewer can
//! reuse it by `bytemuck`-casting [`DashFrame::bytes`] into its own
//! `#[repr(C)]` Gaussian struct.
//!
//! Environment variables (see [`DashSessionConfig::from_env`]):
//!   DASH_ROOT, DASH_PYTHON, DASH_SIDECAR_SCRIPT, DASH_ITERATION, DASH_MASK_MODE.

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::TryFrom;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// Default Gaussian3d ABI stride in bytes:
/// `position[3] + opacity + scale[3] + _pad + rotation[4] + sh[48]` = 240.
pub const GAUSSIAN3D_STRIDE: usize = 240;

/// How to launch and address a DASH sidecar.
#[derive(Debug, Clone)]
pub struct DashSessionConfig {
    pub model_dir: PathBuf,
    pub dash_root: PathBuf,
    pub sidecar_script: PathBuf,
    pub python: String,
    pub iteration: i32,
    pub mask_mode: String,
    /// Expected per-Gaussian stride; validated against the sidecar's report.
    pub expected_stride: usize,
}

impl DashSessionConfig {
    /// Explicit paths with sensible defaults (python3, iteration -1, ply_dynamic).
    pub fn new(
        model_dir: impl AsRef<Path>,
        dash_root: impl AsRef<Path>,
        sidecar_script: impl AsRef<Path>,
    ) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            dash_root: dash_root.as_ref().to_path_buf(),
            sidecar_script: sidecar_script.as_ref().to_path_buf(),
            python: "python3".to_string(),
            iteration: -1,
            mask_mode: "ply_dynamic".to_string(),
            expected_stride: GAUSSIAN3D_STRIDE,
        }
    }

    /// Build from environment variables. `model_dir` is supplied by the caller
    /// (e.g. the dropped/auto-loaded directory).
    pub fn from_env(model_dir: impl AsRef<Path>) -> Result<Self> {
        let dash_root = std::env::var_os("DASH_ROOT")
            .map(PathBuf::from)
            .ok_or_else(|| anyhow!("DASH_ROOT is required to start the DASH sidecar"))?;
        let python = std::env::var("DASH_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let sidecar_script = match std::env::var_os("DASH_SIDECAR_SCRIPT") {
            Some(p) => PathBuf::from(p),
            None => std::env::current_dir()
                .context("failed to determine current directory")?
                .join("sidecar")
                .join("dash_sidecar.py"),
        };
        let iteration = std::env::var("DASH_ITERATION")
            .ok()
            .map(|v| v.parse::<i32>())
            .transpose()
            .context("DASH_ITERATION must be an integer")?
            .unwrap_or(-1);
        let mask_mode =
            std::env::var("DASH_MASK_MODE").unwrap_or_else(|_| "ply_dynamic".to_string());
        Ok(Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            dash_root,
            sidecar_script,
            python,
            iteration,
            mask_mode,
            expected_stride: GAUSSIAN3D_STRIDE,
        })
    }
}

/// One decoded frame: raw packed bytes plus their shape.
#[derive(Debug, Clone)]
pub struct DashFrame {
    pub gaussian_count: usize,
    pub stride: usize,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Deserialize)]
struct ReadyLine {
    host: String,
    port: u16,
}

#[derive(Debug, Default, Deserialize)]
struct ResponseMeta {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    gaussian_count: Option<usize>,
    #[serde(default)]
    stride: Option<usize>,
    #[serde(default)]
    byte_len: Option<usize>,
}

/// A live sidecar process plus its TCP connection.
pub struct DashSession {
    child: Child,
    stream: TcpStream,
    gaussian_count: usize,
    stride: usize,
}

impl DashSession {
    /// Spawn the sidecar (CWD = DASH root, so it reuses DASH's prebuilt
    /// hashencoder JIT cache in `./tmp_build`), wait for readiness, connect,
    /// and validate the reported stride.
    pub fn start(config: DashSessionConfig) -> Result<Self> {
        let model_dir = canonicalize_dir(&config.model_dir)
            .with_context(|| format!("DASH model directory not found: {:?}", config.model_dir))?;
        let dash_root = canonicalize_dir(&config.dash_root)
            .with_context(|| format!("DASH_ROOT not found: {:?}", config.dash_root))?;
        if !config.sidecar_script.is_file() {
            bail!("DASH sidecar script not found: {:?}", config.sidecar_script);
        }
        let sidecar_script = config
            .sidecar_script
            .canonicalize()
            .with_context(|| format!("sidecar script: {:?}", config.sidecar_script))?;

        let mut child = Command::new(&config.python)
            .arg("-u")
            .arg(&sidecar_script)
            .arg("--dash-root")
            .arg(&dash_root)
            .arg("--model-dir")
            .arg(&model_dir)
            .arg("--iteration")
            .arg(config.iteration.to_string())
            .arg("--mask-mode")
            .arg(&config.mask_mode)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg("0")
            .current_dir(&dash_root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to spawn DASH sidecar with {}", config.python))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("failed to capture sidecar stdout"))?;
        let mut reader = BufReader::new(stdout);
        let ready =
            read_ready_line(&mut reader).context("DASH sidecar did not report readiness")?;

        let stream = TcpStream::connect((ready.host.as_str(), ready.port))
            .with_context(|| format!("connect sidecar at {}:{}", ready.host, ready.port))?;
        let _ = stream.set_nodelay(true);

        let mut session = Self {
            child,
            stream,
            gaussian_count: 0,
            stride: 0,
        };
        let (meta, _) = session.request(json!({"cmd":"info"}))?;
        let count = meta
            .gaussian_count
            .ok_or_else(|| anyhow!("info response missing gaussian_count"))?;
        let stride = meta
            .stride
            .ok_or_else(|| anyhow!("info response missing stride"))?;
        if stride != config.expected_stride {
            bail!(
                "DASH stride mismatch: sidecar={}, expected={}",
                stride,
                config.expected_stride
            );
        }
        session.gaussian_count = count;
        session.stride = stride;
        Ok(session)
    }

    pub fn gaussian_count(&self) -> usize {
        self.gaussian_count
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Request the deformed frame for normalized time `time` in `[0, 1)`.
    pub fn request_frame(&mut self, time: f32) -> Result<DashFrame> {
        let (meta, payload) = self.request(json!({"cmd":"frame","time":time}))?;
        let count = meta
            .gaussian_count
            .ok_or_else(|| anyhow!("frame response missing gaussian_count"))?;
        let stride = meta
            .stride
            .ok_or_else(|| anyhow!("frame response missing stride"))?;
        let byte_len = meta.byte_len.unwrap_or(payload.len());
        validate_frame(
            count,
            stride,
            byte_len,
            payload.len(),
            self.stride,
            self.gaussian_count,
        )?;
        Ok(DashFrame {
            gaussian_count: count,
            stride,
            bytes: payload,
        })
    }

    fn request(&mut self, value: Value) -> Result<(ResponseMeta, Vec<u8>)> {
        write_request(&mut self.stream, &value)?;
        self.stream.flush().context("flush DASH request")?;
        let (meta, payload) = read_response(&mut self.stream)?;
        if !meta.ok {
            bail!(
                "DASH sidecar error: {}",
                meta.error.clone().unwrap_or_else(|| "unknown error".into())
            );
        }
        Ok((meta, payload))
    }
}

impl Drop for DashSession {
    fn drop(&mut self) {
        let _ = write_request(&mut self.stream, &json!({"cmd":"shutdown"}));
        let _ = self.stream.flush();
        let _ = read_response(&mut self.stream);
        match self.child.try_wait() {
            Ok(Some(_)) => {}
            _ => {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }
}

/// Pure frame-shape validator (unit tested).
fn validate_frame(
    count: usize,
    stride: usize,
    byte_len: usize,
    payload_len: usize,
    session_stride: usize,
    session_count: usize,
) -> Result<()> {
    if stride != session_stride {
        bail!(
            "DASH frame stride mismatch: {} != {}",
            stride,
            session_stride
        );
    }
    let expected = count
        .checked_mul(stride)
        .ok_or_else(|| anyhow!("DASH frame byte length overflow"))?;
    if byte_len != expected || payload_len != expected {
        bail!(
            "DASH frame size mismatch: meta_byte_len={}, payload={}, expected={}",
            byte_len,
            payload_len,
            expected
        );
    }
    if count != session_count {
        bail!(
            "DASH gaussian_count changed: {} -> {}",
            session_count,
            count
        );
    }
    Ok(())
}

fn canonicalize_dir(p: &Path) -> Result<PathBuf> {
    let c = p.canonicalize()?;
    if !c.is_dir() {
        bail!("not a directory: {:?}", c);
    }
    Ok(c)
}

fn read_ready_line<R: BufRead>(reader: &mut R) -> Result<ReadyLine> {
    const PREFIX: &str = "DASH_SIDECAR_READY ";
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            bail!("sidecar stdout closed before readiness line");
        }
        if let Some(j) = line.trim().strip_prefix(PREFIX) {
            return serde_json::from_str(j)
                .with_context(|| format!("failed to parse readiness line: {line:?}"));
        }
    }
}

// ---- wire protocol (length-prefixed) ----

fn write_request<W: Write>(w: &mut W, value: &Value) -> Result<()> {
    let body = serde_json::to_vec(value).context("serialize DASH request")?;
    let len = u32::try_from(body.len()).context("DASH request too large")?;
    w.write_all(&len.to_le_bytes())
        .context("write request len")?;
    w.write_all(&body).context("write request body")?;
    Ok(())
}

fn read_response<R: Read>(r: &mut R) -> Result<(ResponseMeta, Vec<u8>)> {
    let meta_len = read_u32(r)? as usize;
    if meta_len > 1_000_000 {
        bail!("DASH response metadata too large: {} bytes", meta_len);
    }
    let mut meta_bytes = vec![0u8; meta_len];
    r.read_exact(&mut meta_bytes)
        .context("read DASH response metadata")?;
    let meta: ResponseMeta = serde_json::from_slice(&meta_bytes)
        .with_context(|| String::from_utf8_lossy(&meta_bytes).to_string())?;
    let payload_len = read_u64(r)? as usize;
    let mut payload = vec![0u8; payload_len];
    if payload_len > 0 {
        r.read_exact(&mut payload)
            .context("read DASH response payload")?;
    }
    Ok((meta, payload))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).context("read u32")?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).context("read u64")?;
    Ok(u64::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::net::TcpListener;
    use std::thread;

    fn write_response_test<W: Write>(w: &mut W, meta: &Value, payload: &[u8]) {
        let mb = serde_json::to_vec(meta).unwrap();
        w.write_all(&(mb.len() as u32).to_le_bytes()).unwrap();
        w.write_all(&mb).unwrap();
        w.write_all(&(payload.len() as u64).to_le_bytes()).unwrap();
        w.write_all(payload).unwrap();
    }

    #[test]
    fn request_framing_is_len_prefixed_json() {
        let mut buf = Vec::new();
        write_request(&mut buf, &json!({"cmd":"frame","time":0.25})).unwrap();
        let len = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        assert_eq!(len, buf.len() - 4);
        let v: Value = serde_json::from_slice(&buf[4..]).unwrap();
        assert_eq!(v["cmd"], "frame");
    }

    #[test]
    fn read_response_parses_meta_and_payload() {
        let mut buf = Vec::new();
        let payload = vec![7u8; 480];
        write_response_test(
            &mut buf,
            &json!({"ok":true,"gaussian_count":2,"stride":240,"byte_len":480}),
            &payload,
        );
        let mut cur = Cursor::new(buf);
        let (meta, p) = read_response(&mut cur).unwrap();
        assert!(meta.ok);
        assert_eq!(meta.gaussian_count, Some(2));
        assert_eq!(meta.stride, Some(240));
        assert_eq!(p.len(), 480);
    }

    #[test]
    fn validate_frame_accepts_good_and_rejects_bad() {
        // good
        assert!(validate_frame(3, 240, 720, 720, 240, 3).is_ok());
        // stride mismatch
        assert!(validate_frame(3, 200, 600, 600, 240, 3).is_err());
        // size mismatch
        assert!(validate_frame(3, 240, 480, 720, 240, 3).is_err());
        // count changed
        assert!(validate_frame(4, 240, 960, 960, 240, 3).is_err());
    }

    /// Full client wire round-trip against an in-process fake sidecar (no Python).
    #[test]
    fn client_wire_roundtrip_over_loopback() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            loop {
                let mut lb = [0u8; 4];
                if sock.read_exact(&mut lb).is_err() {
                    break;
                }
                let n = u32::from_le_bytes(lb) as usize;
                let mut jb = vec![0u8; n];
                sock.read_exact(&mut jb).unwrap();
                let req: Value = serde_json::from_slice(&jb).unwrap();
                match req["cmd"].as_str() {
                    Some("info") => write_response_test(
                        &mut sock,
                        &json!({"ok":true,"gaussian_count":3,"stride":240}),
                        &[],
                    ),
                    Some("frame") => {
                        let payload = vec![1u8; 3 * 240];
                        write_response_test(
                            &mut sock,
                            &json!({"ok":true,"gaussian_count":3,"stride":240,"byte_len":720}),
                            &payload,
                        );
                    }
                    Some("shutdown") => {
                        write_response_test(&mut sock, &json!({"ok":true}), &[]);
                        break;
                    }
                    _ => write_response_test(&mut sock, &json!({"ok":false,"error":"bad"}), &[]),
                }
            }
        });

        let mut stream = TcpStream::connect(addr).unwrap();
        write_request(&mut stream, &json!({"cmd":"info"})).unwrap();
        stream.flush().unwrap();
        let (meta, _) = read_response(&mut stream).unwrap();
        assert!(meta.ok && meta.stride == Some(240) && meta.gaussian_count == Some(3));

        write_request(&mut stream, &json!({"cmd":"frame","time":0.5})).unwrap();
        stream.flush().unwrap();
        let (fmeta, payload) = read_response(&mut stream).unwrap();
        assert_eq!(payload.len(), 720);
        assert_eq!(fmeta.byte_len, Some(720));

        write_request(&mut stream, &json!({"cmd":"shutdown"})).unwrap();
        stream.flush().unwrap();
        let _ = read_response(&mut stream);
        server.join().unwrap();
    }
}
