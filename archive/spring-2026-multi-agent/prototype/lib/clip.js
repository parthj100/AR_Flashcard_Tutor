// === CLIP zero-shot recognition via Transformers.js ===
// Runs Xenova/clip-vit-base-patch32 in the browser (ONNX runtime, WebGPU if available).
// We cache text-prompt embeddings once at startup, then every camera frame is
// encoded to an image embedding and cosine-scored against the text cache.
//
// Public API:
//   await Clip.init({ onProgress })         // downloads model (~150 MB, cached after)
//   Clip.indexVocabulary(vocab)             // vocab = [{ id, prompts: [string] }]
//   await Clip.scoreCanvas(canvas, {topK})  // -> { top, confidence, results: [{id, score}] }
//   Clip.isReady()
//
// The vocabulary index can be rebuilt at any time without reloading the model.

import {
  AutoTokenizer,
  AutoProcessor,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';

// Prefer WebGPU when available; fall back to WASM automatically.
env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL_ID = 'Xenova/clip-vit-base-patch32';

const state = {
  ready: false,
  loading: false,
  tokenizer: null,
  processor: null,
  textModel: null,
  visionModel: null,
  device: 'wasm',
  // Flat arrays kept in the same order for fast scoring:
  promptEmbeddings: null, // Float32Array[] length N
  promptOwners: [],       // vocabId for each prompt row
  promptTexts: [],        // raw text for each prompt row
  vocabIds: [],           // unique ids, stable order
};

function hasWebGPU() {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

async function tryLoad(device) {
  // Force fp32 AND the dedicated text/vision ONNX subgraphs. Transformers.js
  // 3.0.2 otherwise falls back to the combined `model.onnx` for this checkpoint
  // when using WebGPU, which expects BOTH text + image inputs at once and
  // breaks vision-only calls with "Missing the following inputs: input_ids".
  const base = device === 'webgpu' ? { device, dtype: 'fp32' } : { device };
  const [tokenizer, processor, textModel, visionModel] = await Promise.all([
    AutoTokenizer.from_pretrained(MODEL_ID),
    AutoProcessor.from_pretrained(MODEL_ID),
    CLIPTextModelWithProjection.from_pretrained(MODEL_ID,   { ...base, model_file_name: 'text_model' }),
    CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { ...base, model_file_name: 'vision_model' }),
  ]);
  return { tokenizer, processor, textModel, visionModel };
}

export async function init({ onProgress } = {}) {
  if (state.ready) return { device: state.device };
  if (state.loading) {
    // Wait for in-flight init
    while (state.loading) await new Promise(r => setTimeout(r, 100));
    return { device: state.device };
  }
  state.loading = true;
  onProgress?.({ phase: 'downloading', message: 'Loading CLIP (first run downloads ~150 MB, cached after)…' });

  try {
    // Prefer WebGPU, fall back to WASM.
    let loaded;
    if (hasWebGPU()) {
      try {
        loaded = await tryLoad('webgpu');
        state.device = 'webgpu';
      } catch (e) {
        console.warn('[clip] WebGPU load failed, falling back to WASM:', e);
        loaded = await tryLoad('wasm');
        state.device = 'wasm';
      }
    } else {
      loaded = await tryLoad('wasm');
      state.device = 'wasm';
    }
    Object.assign(state, loaded);
    state.ready = true;
    onProgress?.({ phase: 'ready', message: `CLIP ready (${state.device})` });
    return { device: state.device };
  } finally {
    state.loading = false;
  }
}

export function isReady() { return state.ready; }
export function getDevice() { return state.device; }

// --- Text indexing ---------------------------------------------------------

function l2normalize(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i] * arr[i];
  const n = Math.sqrt(s) || 1;
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[i] / n;
  return out;
}

async function embedTexts(texts) {
  const inputs = state.tokenizer(texts, { padding: true, truncation: true });
  const { text_embeds } = await state.textModel(inputs);
  // text_embeds is a Tensor of shape [N, D]
  const data = text_embeds.data;
  const [N, D] = text_embeds.dims;
  const out = [];
  for (let i = 0; i < N; i++) {
    out.push(l2normalize(data.slice(i * D, (i + 1) * D)));
  }
  return out;
}

/**
 * vocab: [{ id: string, prompts: [string] }]
 * Multiple prompts per id is supported — we keep them flat and later take the
 * max score per id to get a single per-topic confidence.
 */
export async function indexVocabulary(vocab) {
  if (!state.ready) throw new Error('Clip.init() must be called first');
  const flatTexts = [];
  const owners = [];
  const ids = [];
  for (const v of vocab) {
    if (!v.id || !Array.isArray(v.prompts) || v.prompts.length === 0) continue;
    ids.push(v.id);
    for (const p of v.prompts) {
      flatTexts.push(p);
      owners.push(v.id);
    }
  }
  // Batch embed in chunks to keep memory bounded.
  const CHUNK = 64;
  const all = [];
  for (let i = 0; i < flatTexts.length; i += CHUNK) {
    const part = await embedTexts(flatTexts.slice(i, i + CHUNK));
    all.push(...part);
  }
  state.promptEmbeddings = all;
  state.promptOwners = owners;
  state.promptTexts = flatTexts;
  state.vocabIds = ids;
}

// --- Image scoring ---------------------------------------------------------

async function embedCanvas(canvas) {
  // RawImage.fromCanvas handles conversion to the format the processor expects.
  const image = await RawImage.fromCanvas(canvas);
  const pixel = await state.processor(image);
  const { image_embeds } = await state.visionModel(pixel);
  return l2normalize(image_embeds.data);
}

/**
 * Returns the top matches for a frame.
 *   topK: how many distinct vocab ids to return (default 3)
 *   minConfidence: anything below this is flagged as low-confidence
 *
 * Score is cosine similarity in [-1,1]. In practice CLIP matches for real
 * scenes live in [0.15, 0.35]; we normalize to a 0..1 "confidence" via a
 * softmax over the top candidates for display purposes, but also expose the
 * raw cosine so callers can threshold.
 */
export async function scoreCanvas(canvas, { topK = 3 } = {}) {
  if (!state.ready) throw new Error('Clip.init() must be called first');
  if (!state.promptEmbeddings) throw new Error('indexVocabulary() must be called first');

  const img = await embedCanvas(canvas);

  // Per-prompt raw cosine scores
  const D = img.length;
  const perPrompt = new Float32Array(state.promptEmbeddings.length);
  for (let i = 0; i < state.promptEmbeddings.length; i++) {
    const emb = state.promptEmbeddings[i];
    let s = 0;
    for (let k = 0; k < D; k++) s += emb[k] * img[k];
    perPrompt[i] = s;
  }

  // Max-pool by vocab id (best prompt wins per topic)
  const bestByOwner = new Map();
  const bestPromptByOwner = new Map();
  for (let i = 0; i < perPrompt.length; i++) {
    const owner = state.promptOwners[i];
    const s = perPrompt[i];
    if (!bestByOwner.has(owner) || bestByOwner.get(owner) < s) {
      bestByOwner.set(owner, s);
      bestPromptByOwner.set(owner, state.promptTexts[i]);
    }
  }

  // Sort descending
  const ranked = [...bestByOwner.entries()]
    .map(([id, raw]) => ({ id, raw, matchedPrompt: bestPromptByOwner.get(id) }))
    .sort((a, b) => b.raw - a.raw);

  // Softmax over the top-K for a display-friendly 0..1 confidence.
  // Temperature 100 is the CLIP-standard scaling factor.
  const head = ranked.slice(0, Math.max(topK, 4));
  const TEMP = 100;
  const max = head[0].raw;
  let sumExp = 0;
  const exps = head.map(r => { const e = Math.exp((r.raw - max) * TEMP); sumExp += e; return e; });
  head.forEach((r, i) => { r.confidence = exps[i] / sumExp; });

  const results = head.slice(0, topK);
  return {
    top: results[0],
    confidence: results[0]?.confidence ?? 0,
    rawScore: results[0]?.raw ?? 0,
    results,
  };
}
