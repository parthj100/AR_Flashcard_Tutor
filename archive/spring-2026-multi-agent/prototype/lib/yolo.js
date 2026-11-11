// === YOLO client — multi-object capture mode ===
// Talks to the local FastAPI server in YOLOv8-Detection/serve.py.
// Only used when the user opts into multi-object mode in the scan view.
//
// Public API:
//   Yolo.setEndpoint(url)                   // defaults to http://127.0.0.1:8765
//   await Yolo.health()                     // { ok, device, model } | { ok: false, error }
//   await Yolo.detect(canvas, { conf })     // { inferenceMs, detections: [...] }

const state = {
  endpoint: 'http://127.0.0.1:8765',
};

export function setEndpoint(url) {
  state.endpoint = url.replace(/\/+$/, '');
}

export function getEndpoint() {
  return state.endpoint;
}

export async function health() {
  try {
    const r = await fetch(`${state.endpoint}/health`, { method: 'GET' });
    if (!r.ok) return { ok: false, error: `HTTP ${r.status}` };
    const j = await r.json();
    return { ok: true, device: j.device, model: j.model, conf: j.conf };
  } catch (err) {
    return { ok: false, error: err.message || String(err) };
  }
}

/**
 * Run detection on a canvas. The canvas is JPEG-encoded client-side so we
 * don't pay for a full PNG round-trip on every capture.
 *
 * Returns:
 *   {
 *     inferenceMs: number,
 *     device: 'mps' | 'cuda' | 'cpu',
 *     imageSize: { w, h },
 *     detections: [{
 *       className, confidence,
 *       box: { x, y, w, h },           // image-space pixels
 *       boxRel: { x, y, w, h },        // 0..1
 *       crop: string                   // data: URL (JPEG) of the box crop
 *     }]
 *   }
 */
export async function detect(canvas, { conf = 0.3, maxDets = 6 } = {}) {
  const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
  const r = await fetch(`${state.endpoint}/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataUrl, conf, max_dets: maxDets }),
  });
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new Error(`YOLO /detect HTTP ${r.status}: ${text.slice(0, 140)}`);
  }
  const j = await r.json();
  return {
    inferenceMs: j.inference_ms,
    device: j.device,
    imageSize: j.image_size,
    detections: (j.detections || []).map(d => ({
      className: d.class_name,
      confidence: d.confidence,
      box: d.box,
      boxRel: d.box_rel,
      crop: d.crop,
    })),
  };
}

/**
 * Load a base64 data URL into an HTMLCanvasElement, resized to `side` on the
 * longest edge. Used to feed YOLO crops back into CLIP.
 */
export async function cropToCanvas(dataUrl, side = 224) {
  const img = await loadImage(dataUrl);
  const long = Math.max(img.naturalWidth, img.naturalHeight);
  const scale = side / long;
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));
  const c = document.createElement('canvas');
  c.width = side;
  c.height = side;
  const ctx = c.getContext('2d');
  // Letterbox to square so CLIP sees the whole crop, not a stretched one.
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, side, side);
  ctx.drawImage(img, Math.floor((side - w) / 2), Math.floor((side - h) / 2), w, h);
  return c;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}
