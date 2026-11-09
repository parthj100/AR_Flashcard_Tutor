// === OCR client — text extraction mode ===
// Talks to the local EasyOCR FastAPI server in OCR/ocr_serve.py (/ocr route).
// The scan view's only capture path.
//
// Public API:
//   Ocr.setEndpoint(url)                    // defaults to http://127.0.0.1:8766
//   await Ocr.health()                      // { ok, ... } | { ok: false, error }
//   await Ocr.extractText(canvas, { conf }) // { inferenceMs, rawText, lines }

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
      return { ok: true, device: j.device, ocr: j.ocr };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  }
  
  /**
   * Send a canvas frame to /ocr and get extracted text back.
   *
   * Returns:
   *   {
   *     inferenceMs: number,
   *     imageSize: { w, h },
   *     rawText: string,        // all lines joined — send directly to Ollama
   *     lines: [{
   *       text: string,
   *       confidence: number,
   *       bbox: { x1, y1, x2, y2 }
   *     }]
   *   }
   */
  export async function extractText(canvas, { conf = 0.5 } = {}) {
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
    const r = await fetch(`${state.endpoint}/ocr`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl, conf }),
    });
    if (!r.ok) {
      const text = await r.text().catch(() => '');
      throw new Error(`OCR /ocr HTTP ${r.status}: ${text.slice(0, 140)}`);
    }
    const j = await r.json();
    return {
      inferenceMs: j.inference_ms,
      imageSize:   j.image_size,
      rawText:     j.raw_text,
      lines:       j.lines || [],
    };
  }