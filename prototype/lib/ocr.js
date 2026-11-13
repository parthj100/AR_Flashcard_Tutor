// === OCR client — text extraction mode ===
// Talks to the local EasyOCR FastAPI server in OCR/ocr_serve.py (/ocr route).
// The scan view's only capture path.
//
// Public API:
//   Ocr.setEndpoint(url)                    // defaults to http://127.0.0.1:8766
//   await Ocr.health()                      // { ok, models, defaultModel } | { ok: false, error }
//   await Ocr.extractText(canvas, { conf, model }) // { model, inferenceMs, rawText, lines }
//
// model: 'easyocr' or 'pix2text' (see OCR/engines.py). Omit for the server default.

const state = {
    endpoint: 'http://127.0.0.1:8766',
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
      return {
        ok: true,
        ocr: j.ocr,
        defaultModel: j.default_model || 'easyocr',
        models: j.models || {},   // { easyocr: {name, version, settings, loaded}, ... }
      };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  }
  
  /**
   * Send a canvas frame to /ocr and get extracted text back.
   *
   * Returns:
   *   {
   *     model: string,          // which OCR model read it, e.g. 'pix2text'
   *     modelVersion: string,
   *     modelSettings: object,
   *     inferenceMs: number,
   *     imageSize: { w, h },
   *     rawText: string,        // all lines joined — send directly to Ollama
   *     lines: [{
   *       text: string,
   *       confidence: number,
   *       bbox: { x1, y1, x2, y2 },
   *       kind: 'text' | 'math' | 'mixed'   // math is LaTeX wrapped in $...$
   *     }]
   *   }
   */
  export async function extractText(canvas, { conf = 0.5, model } = {}) {
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
    const body = { image: dataUrl, conf };
    if (model) body.model = model;
    const r = await fetch(`${state.endpoint}/ocr`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const text = await r.text().catch(() => '');
      throw new Error(`OCR /ocr HTTP ${r.status}: ${text.slice(0, 140)}`);
    }
    const j = await r.json();
    return {
      model:         j.model,
      modelVersion:  j.model_version,
      modelSettings: j.model_settings,
      inferenceMs: j.inference_ms,
      imageSize:   j.image_size,
      rawText:     j.raw_text,
      lines:       j.lines || [],
    };
  }