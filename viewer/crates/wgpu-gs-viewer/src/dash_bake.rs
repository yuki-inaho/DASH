//! Headless offscreen bake: render a DASH model to a sequence of PNG frames
//! using the *same* GPU pipeline as the live viewer, but with no window/surface.
//!
//! This is the verification backbone:
//!   * Native E2E asserts the frames are non-blank and change over time.
//!   * The web harness plays the PNG sequence so playwright-cli can verify
//!     4D playback in a browser without depending on headless WebGPU.

use crate::camera::Camera;
use crate::gaussian::{Gaussian3d, GaussianResources, Gaussians};
use crate::passes;
use crate::scene::{SceneType, SceneUniform, SCREEN_HEIGHT, SCREEN_WIDTH};
use anyhow::{anyhow, Context, Result};
use dash_runtime::{DashSession, DashSessionConfig};
use std::path::PathBuf;
use wgpu::util::DeviceExt;

/// Bake options (parsed by the `dash_bake` binary).
#[derive(Debug, Clone)]
pub struct Options {
    pub model_dir: PathBuf,
    pub out_dir: PathBuf,
    pub frames: u32,
    /// Camera orbit turns across the whole sequence. The DASH 4D pipeline is
    /// evaluated for each time `t` regardless; the orbit makes the rendered
    /// output visibly vary even when a model is (near-)static in time, which is
    /// the case for short "smoke" trainings.
    pub orbit_turns: f32,
}

pub fn run(opts: Options) -> Result<()> {
    pollster::block_on(run_async(opts))
}

async fn run_async(opts: Options) -> Result<()> {
    std::fs::create_dir_all(&opts.out_dir)
        .with_context(|| format!("create out dir {:?}", opts.out_dir))?;

    // 1. Start the DASH sidecar and fetch the first frame.
    let config = DashSessionConfig::from_env(&opts.model_dir)?;
    let mut session = DashSession::start(config).context("start DASH sidecar")?;
    let first = session.request_frame(0.0)?;
    let initial: Vec<Gaussian3d> =
        bytemuck::cast_slice::<u8, Gaussian3d>(&first.bytes).to_vec();
    let count = initial.len() as u32;
    log::info!("baking {} gaussians over {} frames", count, opts.frames);

    // 2. Headless GPU device (no surface).
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: Default::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let adapter = adapters
        .iter()
        .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .or_else(|| adapters.first())
        .ok_or_else(|| anyhow!("no wgpu adapter available"))?;
    log::info!("bake adapter: {:?}", adapter.get_info());
    let adapter_limits = adapter.limits();
    let required_limits = wgpu::Limits {
        max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
        max_buffer_size: adapter_limits.max_buffer_size,
        ..wgpu::Limits::default()
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("dash-bake device"),
            required_features: wgpu::Features::empty(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            required_limits,
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off,
        })
        .await?;

    // 3. Scene resources + uniform (auto-fit camera so gaussians are framed).
    let resources =
        GaussianResources::new(&device, &Gaussians::Gaussian3d(initial.clone()));
    let fit = fit_params(&initial);
    let mut scene_uniform = SceneUniform::new();
    scene_uniform.update_camera(&orbit_camera(&fit, 0.0));
    scene_uniform.update_gaussian_count(resources.gaussian_count);
    let scene_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scene Uniform Buffer"),
        contents: bytemuck::cast_slice(&[scene_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // 4. Offscreen color target (same format/size the live viewer renders into).
    let render_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bake render texture"),
        size: wgpu::Extent3d {
            width: SCREEN_WIDTH,
            height: SCREEN_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let render_texture_view = render_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // 5. The same compute pipeline the viewer uses.
    let preprocess = passes::preprocess::PreprocessPass::new(
        &device,
        &scene_uniform_buffer,
        &resources,
        SceneType::Gaussian3d,
    );
    let prefix_scan = passes::prefix_scan::PrefixScanPass::new(&device, &resources);
    let duplicate = passes::duplicate::DuplicatePass::new(&device, &resources);
    let radix_sort = passes::radix_sort::RadixSortPass::new(&device, &resources);
    let tile_range = passes::tile_range::TileRangePass::new(&device, &resources);
    let tile_render = passes::tile_render::TileRenderPass::new(
        &device,
        &scene_uniform_buffer,
        &render_texture_view,
        &resources,
    );

    // 6. Readback buffer (bytes_per_row = width*4 = 5120 is already 256-aligned).
    let bytes_per_row = SCREEN_WIDTH * 4;
    assert_eq!(bytes_per_row % 256, 0, "row stride must be 256-aligned");
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bake readback"),
        size: (bytes_per_row * SCREEN_HEIGHT) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let frames = opts.frames.max(1);
    for i in 0..frames {
        let t = i as f32 / frames as f32;

        // Stream the DASH frame for time t into the gaussian buffer.
        let frame = session.request_frame(t)?;
        if frame.gaussian_count as u32 != resources.gaussian_count {
            anyhow::bail!("gaussian_count changed during bake");
        }
        queue.write_buffer(&resources.gaussian_buffer, 0, &frame.bytes);

        scene_uniform.set_time(t);
        let angle = opts.orbit_turns * std::f32::consts::TAU * (i as f32 / frames as f32);
        scene_uniform.update_camera(&orbit_camera(&fit, angle));
        queue.write_buffer(
            &scene_uniform_buffer,
            0,
            bytemuck::cast_slice(&[scene_uniform]),
        );
        // reset visible counter (the live viewer does this each frame)
        queue.write_buffer(&resources.visible_count_buffer, 0, bytemuck::bytes_of(&0u32));

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bake") });
        preprocess.encode(&mut encoder, resources.gaussian_count);
        prefix_scan.encode(&mut encoder, &resources);
        duplicate.encode(&mut encoder, &resources);
        radix_sort.encode(&mut encoder, &resources);
        tile_range.encode(&mut encoder, &resources);
        tile_render.encode(&mut encoder, resources.tiles_width, resources.tiles_height);

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &render_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(SCREEN_HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: SCREEN_WIDTH,
                height: SCREEN_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Map + read back.
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .context("map_async channel closed")?
            .context("map readback buffer")?;
        let data = readback.slice(..).get_mapped_range();
        let rgba = data.to_vec();
        drop(data);
        readback.unmap();

        let path = opts.out_dir.join(format!("frame_{i:04}.png"));
        image::RgbaImage::from_raw(SCREEN_WIDTH, SCREEN_HEIGHT, rgba)
            .ok_or_else(|| anyhow!("failed to build image buffer"))?
            .save(&path)
            .with_context(|| format!("save {path:?}"))?;
    }

    // Manifest for the web player.
    let manifest = format!(
        "{{\"frames\":{},\"width\":{},\"height\":{},\"gaussians\":{}}}\n",
        frames, SCREEN_WIDTH, SCREEN_HEIGHT, count
    );
    std::fs::write(opts.out_dir.join("manifest.json"), manifest)?;
    log::info!("baked {} frames to {:?}", frames, opts.out_dir);
    println!("BAKE_OK frames={} dir={}", frames, opts.out_dir.display());
    Ok(())
}

/// Camera fit derived from the point-cloud bounding box.
struct FitParams {
    center: glam::Vec3,
    dist: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

fn fit_params(gaussians: &[Gaussian3d]) -> FitParams {
    let n = gaussians.len();
    let fovy = 45.0_f32;
    if n == 0 {
        return FitParams {
            center: glam::Vec3::ZERO,
            dist: 4.0,
            fovy,
            znear: 0.01,
            zfar: 100.0,
        };
    }
    // Per-axis median center + p95 radius: robust to the far outlier points that
    // COLMAP/DASH scenes contain (which otherwise blow up a min/max bbox and push
    // the camera so far the scene becomes a faint smudge).
    let mut center = [0.0f32; 3];
    for (k, c) in center.iter_mut().enumerate() {
        let mut col: Vec<f32> = gaussians.iter().map(|g| g.position[k]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        *c = col[n / 2];
    }
    let c = glam::vec3(center[0], center[1], center[2]);
    let mut dists: Vec<f32> = gaussians
        .iter()
        .map(|g| (glam::vec3(g.position[0], g.position[1], g.position[2]) - c).length())
        .collect();
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let radius = dists[((n as f32 * 0.95) as usize).min(n - 1)].max(1e-3);
    let dist = (radius / (fovy.to_radians() * 0.5).tan()) * 1.4;
    FitParams {
        center: c,
        dist,
        fovy,
        znear: (radius * 0.01).max(0.01),
        zfar: dist + radius * 20.0,
    }
}

/// Place the camera on an orbit around the fitted center (diagonal vantage,
/// rotated by `angle_rad` about the up axis).
fn orbit_camera(f: &FitParams, angle_rad: f32) -> Camera {
    let base = glam::vec3(1.0, 0.6, 1.0).normalize();
    let dir = glam::Mat3::from_rotation_y(angle_rad) * base;
    Camera {
        eye: f.center + dir * f.dist,
        target: f.center,
        up: glam::Vec3::Y,
        aspect: SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32,
        fovy: f.fovy,
        znear: f.znear,
        zfar: f.zfar,
    }
}
