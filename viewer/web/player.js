// DASH 4DGS web player: a 2D-canvas flipbook of the GPU-rendered frame sequence
// baked by the native `dash_bake` binary. Deliberately WebGPU-free so it runs in
// any headless browser (the frames are genuine GPU renders of the DASH model).
//
// Exposes verification hooks for playwright-cli:
//   window.__viewer  = { ready, frameCount, currentFrame, drawnFrames }
//   window.__sigs    = Set of per-frame pixel signatures (size > 1 => frames differ)
//   window.__errors  = [] of captured page/console errors
//   window.__report() = single status string for `playwright-cli --raw eval`
(function () {
  const canvas = document.getElementById("view");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const hud = document.getElementById("hud");

  window.__errors = [];
  window.addEventListener("error", (e) => window.__errors.push(String(e.message || e)));
  window.addEventListener("unhandledrejection", (e) =>
    window.__errors.push("promise:" + String(e.reason)),
  );
  const _err = console.error.bind(console);
  console.error = function () {
    window.__errors.push(Array.from(arguments).map(String).join(" "));
    _err.apply(console, arguments);
  };

  window.__viewer = { ready: false, frameCount: 0, currentFrame: -1, drawnFrames: 0 };
  window.__sigs = new Set();

  function sample() {
    try {
      const w = canvas.width, h = canvas.height;
      const data = ctx.getImageData(0, 0, w, h).data;
      let nonBlank = 0, total = 0, sig = 0;
      for (let i = 0; i < data.length; i += 4 * 197) {
        total++;
        const r = data[i], g = data[i + 1], b = data[i + 2];
        if (r > 12 || g > 12 || b > 12) nonBlank++;
        sig = (sig + r * 3 + g * 5 + b * 7) >>> 0;
      }
      return { nonBlankRatio: total ? nonBlank / total : 0, sig };
    } catch (e) {
      window.__errors.push("sample:" + e);
      return { nonBlankRatio: 0, sig: 0, error: String(e) };
    }
  }
  window.__sampleCanvas = sample;

  window.__report = function () {
    const s = sample();
    return [
      window.__viewer.ready ? "ready" : "notready",
      "frames=" + window.__viewer.frameCount,
      "drawn=" + window.__viewer.drawnFrames,
      "distinct=" + window.__sigs.size,
      "nonblank=" + s.nonBlankRatio.toFixed(3),
      "errors=" + window.__errors.length,
    ].join(" ");
  };

  async function main() {
    const manifest = await (await fetch("baked/manifest.json")).json();
    canvas.width = manifest.width;
    canvas.height = manifest.height;
    window.__viewer.frameCount = manifest.frames;

    const imgs = [];
    for (let i = 0; i < manifest.frames; i++) {
      const im = new Image();
      im.src = "baked/frame_" + String(i).padStart(4, "0") + ".png";
      try {
        await im.decode();
      } catch (e) {
        window.__errors.push("img" + i + ":" + e);
      }
      imgs.push(im);
    }
    window.__viewer.ready = true;

    let i = 0;
    function tick() {
      const im = imgs[i];
      if (im && im.complete) ctx.drawImage(im, 0, 0, canvas.width, canvas.height);
      window.__viewer.currentFrame = i;
      window.__viewer.drawnFrames++;
      window.__sigs.add(sample().sig);
      hud.textContent =
        "DASH 4DGS  frame " + (i + 1) + "/" + manifest.frames +
        "  gaussians " + manifest.gaussians +
        "  t=" + (i / manifest.frames).toFixed(2);
      i = (i + 1) % manifest.frames;
    }
    tick();
    setInterval(tick, 200);
  }

  main().catch((e) => window.__errors.push("main:" + e));
})();
