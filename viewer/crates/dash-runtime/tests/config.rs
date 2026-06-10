use dash_runtime::{DashSessionConfig, GAUSSIAN3D_STRIDE};

#[test]
fn abi_stride_is_240() {
    assert_eq!(GAUSSIAN3D_STRIDE, 240);
}

#[test]
fn config_new_has_defaults() {
    let c = DashSessionConfig::new(
        "/tmp/model",
        "/tmp/dash",
        "/tmp/dash/sidecar/dash_sidecar.py",
    );
    assert_eq!(c.iteration, -1);
    assert_eq!(c.mask_mode, "ply_dynamic");
    assert_eq!(c.python, "python3");
    assert_eq!(c.expected_stride, GAUSSIAN3D_STRIDE);
}

#[test]
fn config_from_env_reads_vars() {
    std::env::set_var("DASH_ROOT", "/tmp");
    std::env::set_var("DASH_ITERATION", "-2");
    std::env::set_var("DASH_MASK_MODE", "all");
    let c = DashSessionConfig::from_env("/tmp/model").unwrap();
    assert_eq!(c.iteration, -2);
    assert_eq!(c.mask_mode, "all");
    assert_eq!(c.dash_root, std::path::PathBuf::from("/tmp"));
    assert_eq!(c.model_dir, std::path::PathBuf::from("/tmp/model"));
}
