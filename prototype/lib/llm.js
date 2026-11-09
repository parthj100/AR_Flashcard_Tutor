// === Ollama client — on-demand flashcard generation ===
// Talks directly to the local Ollama server at http://localhost:11434.
// Uses Ollama's structured-output (JSON) mode so we get parseable results
// without prompt-engineering a JSON parser.
//
// Prereqs for the user (see SETUP.md):
//   1. ollama serve             (or run the Ollama app)
//   2. ollama pull phi4-mini
//   3. OLLAMA_ORIGINS='*'       so browser requests aren't blocked by CORS
//
// Public API:
//   await Llm.health()                    -> { ok, models: [...] }
//   await Llm.generateFlashcard({ topic, hintPrompt }) -> Flashcard

const DEFAULT_HOST = 'http://localhost:11434';
const DEFAULT_MODEL = 'phi4-mini';

const SYSTEM_PROMPT = `You write concise educational flashcards for a science/humanities study app called Lens.
For a given topic, output a single JSON object following the exact schema provided.
Be factually accurate, neutral, and specific. No marketing fluff.
If a subject area is ambiguous, pick the most common/educational interpretation.
Write in a calm, textbook tone. Use scientific notation where appropriate.
Do NOT include any commentary, code fences, or text outside the JSON object.`;

// Schema we constrain the model to. Matches what the flashcard view renders.
const FLASHCARD_SCHEMA = {
  type: 'object',
  properties: {
    name:     { type: 'string' },
    subject:  { type: 'string', description: 'Short taxonomy like "CHEMISTRY · INORGANIC · SALT"' },
    formula:  { type: 'string', description: 'Primary identifier: chemical formula, scientific name, date, coordinates, etc.' },
    mass:     { type: 'string', description: 'Secondary identifier: mass, dimensions, key stat. Keep short.' },
    oneline:  { type: 'string', description: 'One or two sentences capturing the essence.' },
    facts: {
      type: 'array',
      minItems: 4,
      maxItems: 4,
      items: {
        type: 'object',
        properties: {
          num:   { type: 'string', description: 'Two-digit number like "01", "02", "03", "04".' },
          label: { type: 'string', description: 'Short ALL-CAPS label, <= 3 words.' },
          body:  { type: 'string', description: 'One-sentence factual detail.' },
        },
        required: ['num', 'label', 'body'],
      },
    },
  },
  required: ['name', 'subject', 'formula', 'mass', 'oneline', 'facts'],
};

let config = {
  host: DEFAULT_HOST,
  model: DEFAULT_MODEL,
};

export function configure({ host, model }) {
  if (host) config.host = host;
  if (model) config.model = model;
}

export function getConfig() { return { ...config }; }

export async function health() {
  try {
    const res = await fetch(`${config.host}/api/tags`, { method: 'GET' });
    if (!res.ok) return { ok: false, error: `HTTP ${res.status}` };
    const data = await res.json();
    const models = (data.models || []).map(m => m.name);
    return {
      ok: true,
      models,
      hasConfiguredModel: models.some(m => m === config.model || m.startsWith(config.model.split(':')[0] + ':')),
    };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

/**
 * Generate a flashcard for a free-text topic.
 *   topic:      the recognized topic name, e.g. "Copper sulfate"
 *   hintPrompt: optional context, e.g. the OCR transcript so the LLM
 *               knows the visual context ("close-up of blue crystals...")
 *   signal:     optional AbortSignal so callers can cancel long generations
 */
export async function generateFlashcard({ topic, hintPrompt = '', signal } = {}) {
  if (!topic) throw new Error('topic is required');

  const userPrompt = `Topic: ${topic}${hintPrompt ? `\nVisual hint: ${hintPrompt}` : ''}

Write a flashcard for this topic as JSON matching the provided schema.
- "name" must be a clean human-readable title (capitalize normally).
- "subject" is 2-3 short ALL-CAPS tokens separated by " · " (e.g. "BIOLOGY · ORGANELLE · EUKARYOTIC").
- "formula" is the most canonical identifier (chemical formula, scientific name, year + location, etc.).
- "mass" is a short quantitative note (mass, size, count, dimensions).
- "oneline" is exactly 1-2 sentences, ~25-40 words.
- "facts" is exactly 4 items with num "01" through "04", each with a short ALL-CAPS label and a one-sentence body.`;

  const body = {
    model: config.model,
    stream: false,
    format: FLASHCARD_SCHEMA,
    options: { temperature: 0.2, num_predict: 512 },
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user',   content: userPrompt },
    ],
  };

  const res = await fetch(`${config.host}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Ollama error ${res.status}: ${text.slice(0, 200)}`);
  }

  const data = await res.json();
  const content = data.message?.content?.trim();
  if (!content) throw new Error('Empty response from Ollama');

  let parsed;
  try {
    parsed = JSON.parse(content);
  } catch (e) {
    // Some small models still wrap JSON in prose despite format=schema. Salvage.
    const m = content.match(/\{[\s\S]*\}/);
    if (!m) throw new Error(`Non-JSON response: ${content.slice(0, 200)}`);
    parsed = JSON.parse(m[0]);
  }

  return normalizeFlashcard(parsed, topic);
}

function normalizeFlashcard(obj, topic) {
  // Defensive normalization — small models occasionally drop fields.
  const facts = Array.isArray(obj.facts) ? obj.facts.slice(0, 4) : [];
  while (facts.length < 4) {
    facts.push({ num: String(facts.length + 1).padStart(2, '0'), label: 'DETAIL', body: '—' });
  }
  facts.forEach((f, i) => {
    f.num = f.num || String(i + 1).padStart(2, '0');
    f.label = (f.label || 'DETAIL').toUpperCase();
    f.body = f.body || '—';
  });

  return {
    name: obj.name || topic,
    subject: (obj.subject || 'GENERATED').toUpperCase(),
    formula: obj.formula || '—',
    mass: obj.mass || '',
    oneline: obj.oneline || '',
    facts,
    generated: true,
  };
}
