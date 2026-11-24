// === Lens prototype — router + views ===
const D = window.LENS_DATA;
const $ = (sel, root = document) => root.querySelector(sel);
const el = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------
// Three RT collections survive page reloads via localStorage:
//   captures        — recent scan feed
//   generatedCards  — LLM-produced cards (the "action" in offline RL terms)
//   quizSessions    — quiz outcomes (the "reward")
// Together these form the (state, action, reward) triples ALGORITHMS.md
// describes. Persistence here is the unblocker referenced as the only
// thing standing between the algorithmic framing and being able to train.
//
// Bounded to keep localStorage from growing unbounded. Versioned key prefix
// so we can change the schema later without resurrecting stale data.
const LENS_STORAGE_KEY    = 'lens.runtime.v1';
const LENS_QUIZ_CAP       = 500;
const LENS_CAPTURE_CAP    = 200;
const LENS_GEN_CARD_CAP   = 200;

function loadPersistedRT() {
  try {
    const raw = localStorage.getItem(LENS_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return {
      captures:       Array.isArray(parsed.captures)       ? parsed.captures       : [],
      generatedCards: (parsed.generatedCards && typeof parsed.generatedCards === 'object')
                        ? parsed.generatedCards : {},
      quizSessions:   Array.isArray(parsed.quizSessions)   ? parsed.quizSessions   : [],
    };
  } catch (e) {
    console.warn('lens: failed to read persisted runtime', e);
    return null;
  }
}

let _persistTimer = null;
function persistRT() {
  // Debounce — many writes per second during a quiz.
  if (_persistTimer) return;
  _persistTimer = setTimeout(() => {
    _persistTimer = null;
    try {
      const snapshot = {
        captures:       (RT.captures || []).slice(0, LENS_CAPTURE_CAP),
        generatedCards: trimGeneratedCards(RT.generatedCards || {}, LENS_GEN_CARD_CAP),
        quizSessions:   (RT.quizSessions || []).slice(0, LENS_QUIZ_CAP),
      };
      localStorage.setItem(LENS_STORAGE_KEY, JSON.stringify(snapshot));
    } catch (e) {
      console.warn('lens: failed to persist runtime', e);
    }
  }, 250);
}

function trimGeneratedCards(obj, cap) {
  const ids = Object.keys(obj);
  if (ids.length <= cap) return obj;
  // Drop the oldest entries (insertion order is preserved in modern JS).
  return Object.fromEntries(ids.slice(ids.length - cap).map(k => [k, obj[k]]));
}

const _persisted = loadPersistedRT();

// Runtime state (outlives view mounts)
const RT = window.LENS_RUNTIME = window.LENS_RUNTIME || {
  ollamaOk: false,      // true after a successful Ollama health check
  ocrOk: false,         // true after a successful OCR service health check
  ocrModel: (function () {   // which OCR model reads captures: 'easyocr' | 'pix2text'
    try { return localStorage.getItem('lens.ocrModel') || 'easyocr'; } catch { return 'easyocr'; }
  })(),
  generatedCards: (_persisted && _persisted.generatedCards) || {},
  scan: null,           // active scan controller (camera + loop)
  captures:       (_persisted && _persisted.captures)       || [],
  quizSessions:   (_persisted && _persisted.quizSessions)   || [],
  sessionStart: Date.now(),
};

// JSONL trajectory export — matches the schema in ALGORITHMS.md.
// Run window.lensExportTrajectories() in the browser console to download
// a .jsonl file containing one (s, a, r) triple per quiz session.
window.lensExportTrajectories = function (filename = 'lens-trajectories.jsonl') {
  const lines = (RT.quizSessions || []).map(sess => {
    // A quiz session may aggregate over multiple cards; emit one trajectory
    // per cardId with the aggregate reward divided by the queue length.
    return (sess.cardIds || []).map(cardId => {
      const card = (D.flashcards && D.flashcards[cardId]) || RT.generatedCards[cardId] || null;
      const isGenerated = !!(RT.generatedCards[cardId]);
      return JSON.stringify({
        t: new Date(sess.at).toISOString(),
        state: {
          topic: cardId,
          generated: isGenerated,
        },
        action: card,
        reward: {
          correct: sess.correct,
          total:   sess.total,
        },
      });
    }).join('\n');
  }).filter(Boolean).join('\n');
  const blob = new Blob([lines + '\n'], { type: 'application/jsonl' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
  return RT.quizSessions.length;
};

// ---------------------------------------------------------------------------
// Live derivations — build decks + due-today from the *real* corpus
// (authored cards in data.js + LLM-generated cards captured at runtime).
// These replace the hardcoded D.decks / D.myDecks / D.dueToday seed lists.
// ---------------------------------------------------------------------------

// Coarse deck key from a card's subject string. Subjects look like
// "BIOLOGY · ORGANELLE · EUKARYOTIC" — we use the first segment.
function deckKey(subject) {
  if (!subject) return 'misc';
  const w = String(subject).split('·')[0].trim().toLowerCase();
  return w || 'misc';
}

const SUBJECT_GRADIENT = {
  chemistry:    'var(--grad-blue)',
  biology:      'var(--grad-orange)',
  botany:       'var(--grad-green)',
  architecture: 'var(--grad-arch)',
  geology:      'var(--grad-violet)',
  art:          'var(--grad-art)',
  physics:      'var(--grad-physics)',
  astronomy:    'var(--grad-astro)',
  music:        'var(--grad-amber)',
  text:         'var(--grad-rust)',
};
const SUBJECT_DOT = {
  chemistry:    '#4a7fc6',
  biology:      '#d97a3a',
  botany:       '#4F8C5C',
  architecture: '#A88468',
  geology:      '#7E5BB7',
  art:          '#B85476',
  physics:      '#6F509E',
  astronomy:    '#3E8AB1',
  music:        '#C49A3F',
  text:         '#A05238',
  misc:         '#7A7A78',
};
function capWord(s) { return s ? s[0].toUpperCase() + s.slice(1) : s; }
function deckDescriptor(key, n, status) {
  if (status === 'GENERATED') return `${n} card${n===1?'':'s'} generated by Phi-4-mini from your scans.`;
  if (status === 'MIXED')     return `${n} card${n===1?'':'s'} — authored content plus your scans.`;
  return `${n} authored card${n===1?'':'s'} in ${key}.`;
}

// Group authored + generated cards by deck key. Returns deck objects with
// the same shape the renderDecks / renderSidebar views already use.
function deriveDecks() {
  const all = [];
  for (const [id, c] of Object.entries(D.flashcards || {})) {
    all.push({ id, subject: c.subject, name: c.name, grad: c.grad, src: 'authored' });
  }
  for (const [id, c] of Object.entries(RT.generatedCards || {})) {
    all.push({ id, subject: c.subject, name: c.name, grad: c.grad, src: 'generated' });
  }
  const groups = new Map();
  for (const c of all) {
    const k = deckKey(c.subject);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(c);
  }
  const dueIds = new Set(deriveDueToday().map(d => d.id));
  const out = [];
  for (const [key, cards] of groups) {
    const srcs = new Set(cards.map(c => c.src));
    let status;
    if (srcs.has('authored') && srcs.has('generated')) status = 'MIXED';
    else if (srcs.has('generated')) status = 'GENERATED';
    else status = 'AUTHORED';
    const due = cards.filter(c => dueIds.has(c.id)).length;
    out.push({
      id:      key,
      subject: key.toUpperCase(),
      name:    capWord(key),
      desc:    deckDescriptor(key, cards.length, status),
      cards:   cards.length,
      due,
      status,
      grad:    SUBJECT_GRADIENT[key] || cards[0]?.grad || 'var(--grad-physics)',
      dot:     SUBJECT_DOT[key] || '#7A7A78',
      _ids:    cards.map(c => c.id),
    });
  }
  out.sort((a, b) => b.cards - a.cards);
  return out;
}

// A card is "due" if it's been captured or generated but hasn't appeared
// in any RT.quizSessions entry yet. Simplest defensible rule — surfaces
// what the learner just scanned and hasn't reviewed.
function deriveDueToday() {
  const quizzed = new Set();
  for (const s of (RT.quizSessions || [])) {
    for (const id of (s.cardIds || [])) quizzed.add(id);
  }
  const captureWhen = new Map();
  for (const c of (RT.captures || [])) captureWhen.set(c.id, c.when);
  // Candidates: captures + generated cards. We exclude authored cards that
  // were never captured — they're "the library", not "your activity".
  const ids = new Set([
    ...Array.from(captureWhen.keys()),
    ...Object.keys(RT.generatedCards || {}),
  ]);
  const out = [];
  for (const id of ids) {
    if (quizzed.has(id)) continue;
    const card = (D.flashcards && D.flashcards[id]) || RT.generatedCards[id];
    if (!card) continue;
    const when = captureWhen.has(id) ? relativeTime(captureWhen.get(id)) : 'just generated';
    out.push({
      id,
      title: card.name || id,
      sub:   (card.subject || '').split('·').slice(0, 2).join('·').trim() || 'Card',
      when,
      now:   true,
    });
  }
  return out;
}

// First card id of a derived deck — used by sidebar deck pills that link
// the user into a representative flashcard.
function firstCardIdOfDeck(deck) {
  return (deck && deck._ids && deck._ids[0]) || null;
}

// Merge a fresh capture into the recent feed (newest first, dedupe by id).
function recordCapture({ id, meta, isGenerated }) {
  const now = Date.now();
  RT.captures = RT.captures.filter(c => c.id !== id);
  RT.captures.unshift({
    id,
    topic: meta.displayName,
    subject: meta.subject || '',
    grad: meta.grad || 'var(--grad-physics)',
    when: now,
    isGenerated: !!isGenerated,
  });
  persistRT();
}

// Human-readable "just now / 3m ago / 2h ago / yesterday / N days ago".
function relativeTime(ts) {
  const s = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (s < 30) return 'just now';
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d === 1) return 'yesterday';
  return `${d} days ago`;
}

// First 3 letters of display name for scan-card tag.
function shortTag(s) {
  return (s || '').replace(/[^A-Za-z]/g, '').slice(0, 4).toUpperCase() || 'NEW';
}

// Look up a vocabulary entry by id. Returns a unified shape regardless of
// whether it came from flashcards{} or extendedVocab[].
function vocabMeta(id) {
  if (D.flashcards[id]) {
    const c = D.flashcards[id];
    return { id, displayName: c.name, subject: c.subject, grad: c.grad, kind: 'authored' };
  }
  const v = (D.extendedVocab || []).find(x => x.id === id);
  if (v) return { id, displayName: v.displayName, subject: v.subject, grad: v.grad, kind: 'extended' };
  return null;
}

// Each deck opens to a representative flashcard. Historic mock IDs kept so
// the search corpus (which still references them) doesn't break; the live
// path resolves via deriveDecks() so newly-created subject decks work too.
const DECK_TO_CARD = {
  'chem-101':     'copper-sulfate',
  'biology-cell': 'mitochondria',
  'arch-survey':  'hagia-sophia',
  'botany-lab':   'maple-leaf',
  'mineral-id':   'quartz',
  'renaissance':  'mona-lisa',
  'mechanics':    'newtons-laws',
  'solar-system': 'jupiter',
};
function cardForDeck(deckId) {
  if (DECK_TO_CARD[deckId]) return DECK_TO_CARD[deckId];
  const live = deriveDecks().find(d => d.id === deckId);
  if (live) return firstCardIdOfDeck(live) || 'copper-sulfate';
  return 'copper-sulfate';
}

// ---------- Router ----------
const routes = {
  '': renderDashboard,
  '#dashboard': renderDashboard,
  '#decks': renderDecks,
  '#scan': renderScan,
  '#flashcard': renderFlashcard, // takes ?id=
  '#quiz': renderQuiz,            // takes ?id= (single card) or ?mode=review
};

function parseHash() {
  const raw = window.location.hash || '';
  const [path, query] = raw.split('?');
  const params = new URLSearchParams(query || '');
  return { path, params };
}

function navigate(hash) {
  if (window.location.hash === hash) return; // no-op
  window.location.hash = hash;
}


window.addEventListener('hashchange', () => {
  // Stop the camera if we're leaving the scan view
  if (window.location.hash !== '#scan' && RT.scan) {
    try { RT.scan.stop(); } catch (_) {}
    RT.scan = null;
  }
  mount();
});
window.addEventListener('DOMContentLoaded', mount);

function mount() {
  const { path, params } = parseHash();
  const view = routes[path] || renderDashboard;
  const root = $('#app');
  root.innerHTML = '';
  view(root, params);
  attachGlobalKeys();
}

// ---------- Sidebar (shared on light pages) ----------
function renderSidebar(active) {
  const decks = deriveDecks();
  const due   = deriveDueToday();
  const sidebarDecks = decks.slice(0, 4);
  const decksHtml = sidebarDecks.length ? sidebarDecks.map(d => `
    <button class="deck-pill" data-deck="${d.id}">
      <span class="deck-dot" style="background:${d.dot}"></span>
      <span>${d.name}</span>
      <span class="count">${d.cards}</span>
    </button>`).join('') : `
    <div class="deck-empty" style="padding:8px 10px;color:var(--muted);font-size:12px;">
      No decks yet — scan something to start one.
    </div>`;

  const reviewBadge = due.length
    ? `<span class="badge">${due.length}</span>` : '';

  // User defaults — D.user may be present as a soft default but we don't
  // bake any specific name into the chrome.
  const userName     = (D.user && D.user.name)     || 'Lens user';
  const userInitials = (D.user && D.user.initials) || 'LN';
  const userPlan     = (D.user && D.user.plan)     || 'Local prototype';

  return `
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark"></div>
      <div class="brand-name">LENS</div>
    </div>

    <button class="new-scan" data-action="new-scan">
      <span class="scan-icon"></span>
      <span class="label">New scan</span>
      <span class="kbd">⌘ S</span>
    </button>

    <div class="nav-section-label">Workspace</div>
    <nav class="nav">
      <button class="nav-item ${active==='dashboard'?'active':''}" data-route="#dashboard">
        ${ico('home')}<span>Home</span>
      </button>
      <button class="nav-item ${active==='decks'?'active':''}" data-route="#decks">
        ${ico('decks')}<span>Decks</span><span class="count">${decks.length}</span>
      </button>
      <button class="nav-item" data-action="start-review" ${due.length ? '' : 'disabled'}>
        ${ico('clock')}<span>Review</span>${reviewBadge}
      </button>
    </nav>

    <div class="nav-section-label">My decks</div>
    <div class="deck-list">${decksHtml}</div>

    <div class="sidebar-footer">
      <div class="avatar">${userInitials}</div>
      <div>
        <div class="user-name">${userName}</div>
        <div class="user-meta">${userPlan}</div>
      </div>
      <button class="dots-btn" title="Account">${ico('dots')}</button>
    </div>
  </aside>`;
}

// ---------- Topbar (shared) ----------
function renderTopbar({ withSearch = true, rightSlot = '' } = {}) {
  return `
  <header class="topbar">
    ${withSearch ? `
      <div class="search" data-action="open-search">
        ${ico('search')}
        <input placeholder="Search cards, decks, formulas…" readonly />
        <span class="kbd-shortcut"><kbd>⌘</kbd><kbd>K</kbd></span>
      </div>` : ''}
    <div class="spacer"></div>
    ${rightSlot}
  </header>`;
}

// ---------- Dashboard (W1) ----------
function renderDashboard(root) {
  const right = `
    <button class="btn btn-icon" title="Notifications">${ico('bell')}</button>
  `;
  const due = deriveDueToday();
  const firstName = ((D.user && D.user.name) || 'Lens user').split(' ')[0];

  // Build "recent scans" from live captures (fresh), falling back to seed data.
  const liveScans = RT.captures.map(c => {
    const authored = D.flashcards[c.id];
    const title = authored?.name || c.topic;
    const sub = authored?.formula || c.subject || (c.isGenerated ? 'Generated by Phi-4-mini' : '');
    return {
      id: c.id,
      title,
      sub,
      when: relativeTime(c.when),
      cards: authored?.facts?.length || 4,
      status: c.isGenerated ? 'new' : 'studying',
      tag: c.isGenerated ? 'AI' : shortTag(title),
      grad: c.grad,
      subject: subjectKeyForCard(c.id),
      isGenerated: c.isGenerated,
    };
  });
  const seedScans = D.recentScans.filter(s => !liveScans.some(l => l.id === s.id));
  const scans = [...liveScans, ...seedScans].slice(0, 9);

  const scansHtml = scans.map(s => `
    <button class="scan-card" data-scan="${s.id}">
      <div class="scan-cover" style="background:${s.grad}">
        <div class="blob"></div>
        <div class="tag-pill">${s.tag}</div>
      </div>
      <div class="scan-meta">
        <div class="scan-title">${s.title}</div>
        <div class="scan-sub">${s.sub}</div>
        <div class="scan-row">
          <span>${s.when} · ${s.cards} cards</span>
          <span class="scan-status status-${s.status}">${s.status.toUpperCase()}</span>
        </div>
      </div>
    </button>`).join('');

  const dueHtml = due.length ? due.map(d => `
    <div class="due-item ${d.now ? 'is-now' : ''}" data-scan="${d.id}">
      <div class="due-bar"></div>
      <div class="due-info">
        <div class="due-title">${d.title}</div>
        <div class="due-sub">${d.sub}</div>
      </div>
      <div class="due-when">${d.when}</div>
    </div>`).join('') : `
    <div class="due-empty" style="padding:20px;color:var(--muted);text-align:center;border:1px dashed var(--border);border-radius:10px;">
      <div style="font-weight:600;margin-bottom:4px;">Caught up</div>
      <div style="font-size:13px;">Nothing pending review — scan something new.</div>
    </div>`;

  // KPIs: derived from real activity when possible.
  const kpi = computeKpis();

  root.innerHTML = `
    <div class="app-shell">
      ${renderSidebar('dashboard')}
      <div class="main">
        ${renderTopbar({ rightSlot: right })}
        <div class="dash">
          <div class="dash-header">
            <div>
              <div class="eyebrow">${todayEyebrow()}</div>
              <h1 class="greeting">${greeting()}, ${firstName}.</h1>
            </div>
            <div class="dash-actions">
              <button class="btn btn-accent" data-action="start-review" ${due.length ? '' : 'disabled'}><span class="dot"></span>Start review · ${due.length} due</button>
            </div>
          </div>

          <section class="kpis">
            <div class="kpi">
              <div class="kpi-label">Cards in corpus</div>
              <div class="kpi-value">${kpi.captured} <span class="delta">${kpi.capturedDelta}</span></div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Retention rate</div>
              <div class="kpi-value">${kpi.retention} <span class="sub">${kpi.quizLabel}</span></div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Time studied</div>
              <div class="kpi-value">${kpi.timeStudied} <span class="sub">this session</span></div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Active days</div>
              <div class="kpi-value"><span class="hl">${kpi.streak}</span> day${kpi.streak===1?'':'s'} <span class="sub">scan or quiz history</span></div>
            </div>
          </section>

          <section class="dash-grid">
            <div>
              <div class="section-head">
                <div class="section-title">Recent scans <span class="sub">— this week</span></div>
                <div class="tabs">
                  <button class="tab active">All</button>
                  <button class="tab">Biology</button>
                  <button class="tab">Chemistry</button>
                  <button class="tab">Architecture</button>
                </div>
              </div>
              <div class="scan-grid">${scansHtml}</div>
            </div>
            <div>
              <div class="due-head">
                <div class="section-title">Due today</div>
                <div class="count">${due.length} card${due.length===1?'':'s'}</div>
              </div>
              <div class="due-list">${dueHtml}</div>
            </div>
          </section>
        </div>
      </div>
    </div>`;

  // Tabs (recent scans filter)
  root.querySelectorAll('.tabs .tab').forEach(t => t.addEventListener('click', () => {
    root.querySelectorAll('.tabs .tab').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    const sel = t.textContent.trim().toLowerCase();
    const map = { biology: 'bio', chemistry: 'chem', architecture: 'arch', botany: 'bot' };
    root.querySelectorAll('.scan-card').forEach(c => {
      const id = c.dataset.scan;
      const item = scans.find(s => s.id === id);
      if (!item) { c.style.display = 'none'; return; }
      c.style.display = (sel === 'all' || map[sel] === item.subject) ? '' : 'none';
    });
  }));
}

// Infer a coarse subject key (chem/bio/arch/bot/...) from a card id so the
// dashboard filter tabs keep working for newly captured cards. Falls back to
// the seed data's `subject` field when present.
function subjectKeyForCard(id) {
  const seed = D.recentScans.find(s => s.id === id);
  if (seed?.subject) return seed.subject;
  const meta = vocabMeta(id);
  const subject = (meta?.subject || '').toLowerCase();
  if (subject.includes('bio') || subject.includes('anatomy') || subject.includes('zool')) return 'bio';
  if (subject.includes('chem')) return 'chem';
  if (subject.includes('arch')) return 'arch';
  if (subject.includes('bot')) return 'bot';
  return '';
}

// Time-of-day greeting.
function greeting() {
  const h = new Date().getHours();
  if (h < 12) return 'Good morning';
  if (h < 18) return 'Good afternoon';
  return 'Good evening';
}
function todayEyebrow() {
  const d = new Date();
  const weekday = d.toLocaleDateString('en-US', { weekday: 'long' });
  const month = d.toLocaleDateString('en-US', { month: 'long' });
  return `${weekday} · ${month} ${d.getDate()}`;
}

// KPI math — derived from live runtime where possible.
function computeKpis() {
  // "Captured" — the corpus the user actually has: authored cards (the
  // library that ships with the app) plus generated cards plus unique
  // captures from the camera. No mock seed; this is real.
  const corpus = new Set([
    ...Object.keys(D.flashcards || {}),
    ...Object.keys(RT.generatedCards || {}),
    ...RT.captures.map(c => c.id),
  ]);
  const captured = corpus.size;
  const newToday = RT.captures.length;
  const capturedDelta = newToday > 0
    ? `+${newToday} scanned`
    : `${Object.keys(D.flashcards || {}).length} authored`;

  // Retention from real quiz sessions; only show a number if we have one.
  let retention = 0, quizCount = 0;
  if (RT.quizSessions.length) {
    const tot = RT.quizSessions.reduce((a, q) => a + q.total, 0);
    const cor = RT.quizSessions.reduce((a, q) => a + q.correct, 0);
    retention = tot ? Math.round((cor / tot) * 100) : 0;
    quizCount = RT.quizSessions.length;
  }
  const retentionText = quizCount > 0 ? `${retention}%` : '—';
  const quizLabel = quizCount > 0
    ? `last ${quizCount} quiz${quizCount===1?'':'zes'}`
    : 'no quizzes yet';

  // Time studied = elapsed session time.
  const minutes = Math.floor((Date.now() - RT.sessionStart) / 60000);
  const timeStudied = minutes >= 60
    ? `${(minutes / 60).toFixed(1)}h`
    : `${Math.max(minutes, 0)}m`;

  // Streak: distinct calendar days with at least one capture or quiz, in
  // the trailing window ending today. No more hardcoded "personal best: 14".
  const days = new Set();
  const dayKey = ts => new Date(ts).toDateString();
  for (const c of RT.captures)      days.add(dayKey(c.when));
  for (const q of RT.quizSessions)  days.add(dayKey(q.at));
  const streak = days.size;

  return {
    captured, capturedDelta,
    retention: retentionText, quizCount, quizLabel,
    timeStudied,
    streak,
  };
}

// ---------- Decks Browse (W4) ----------
function renderDecks(root) {
  const right = `
    <button class="btn">${ico('sort')} Sort: Most cards</button>
  `;

  const decks = deriveDecks();
  const totalCards = decks.reduce((s, d) => s + d.cards, 0);

  const cardsHtml = (filter) => {
    const filtered = decks.filter(d =>
      filter === 'all' ? true : d.status.toLowerCase() === filter
    );
    if (!filtered.length) {
      return `
        <div class="decks-empty" style="grid-column:1/-1;padding:60px 24px;text-align:center;color:var(--muted);border:1px dashed var(--border);border-radius:12px;">
          <div style="font-size:18px;font-weight:600;color:var(--text);margin-bottom:6px;">No decks ${filter==='all'?'yet':`in ${filter.toLowerCase()}`}.</div>
          <div style="font-size:13px;">${filter==='all' ? 'Scan an object or printed flashcard — the system will start a deck for the subject.' : 'Try a different tab.'}</div>
        </div>`;
    }
    return filtered.map(d => `
      <button class="deck-card" data-deck="${d.id}">
        <div class="deck-cover" style="background:${d.grad}">
          <div class="blob"></div>
          <div class="tag-pill">${d.status}</div>
        </div>
        <div class="deck-body">
          <div class="deck-eyebrow">${d.subject}</div>
          <div class="deck-name">${d.name}</div>
          <div class="deck-desc">${d.desc}</div>
        </div>
        <div class="deck-foot">
          <div class="deck-stat"><div class="v">${d.cards}</div><div class="l">cards</div></div>
          <div class="deck-stat"><div class="v">${d.due}</div><div class="l">due</div></div>
          <div class="study">Study ${ico('arrow')}</div>
        </div>
      </button>`).join('');
  };

  const counts = {
    all:       decks.length,
    authored:  decks.filter(d => d.status === 'AUTHORED').length,
    mixed:     decks.filter(d => d.status === 'MIXED').length,
    generated: decks.filter(d => d.status === 'GENERATED').length,
  };

  root.innerHTML = `
    <div class="app-shell">
      ${renderSidebar('decks')}
      <div class="main">
        ${renderTopbar({ rightSlot: right })}
        <div class="decks-page">
          <div class="decks-meta">${decks.length} deck${decks.length===1?'':'s'} · ${totalCards} card${totalCards===1?'':'s'}</div>
          <h1 class="decks-title">Your decks.</h1>
          <div class="decks-tabs">
            <button class="decks-tab active" data-filter="all">All <span class="num">${counts.all}</span></button>
            <button class="decks-tab" data-filter="authored">Authored <span class="num">${counts.authored}</span></button>
            <button class="decks-tab" data-filter="mixed">Mixed <span class="num">${counts.mixed}</span></button>
            <button class="decks-tab" data-filter="generated">Generated <span class="num">${counts.generated}</span></button>
          </div>
          <div class="decks-grid" id="decks-grid">${cardsHtml('all')}</div>
        </div>
      </div>
    </div>`;

  root.querySelectorAll('.decks-tab').forEach(t => t.addEventListener('click', () => {
    root.querySelectorAll('.decks-tab').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    $('#decks-grid', root).innerHTML = cardsHtml(t.dataset.filter);
  }));
}

// ---------- Scan (W2) — OCR pipeline: EasyOCR or Pix2Text reads the photo, LLM writes ----------
// YOLO (detection) and CLIP (recognition) were removed in Fall 2026 when the
// research question narrowed to reading math problems from photos. Their code
// lives in archive/spring-2026-multi-agent/.
function renderScan(root) {
  // Stop any prior scan controller before mounting fresh UI
  if (RT.scan) { try { RT.scan.stop(); } catch (_) {} RT.scan = null; }

  root.innerHTML = `
    <div class="scan-page">
      <section class="scan-stage">
        <div class="scan-backdrop"></div>

        <video id="scan-video" class="scan-video" autoplay muted playsinline></video>
        <div class="scan-vignette"></div>

        <div class="scan-top">
          <button class="back" data-route="#dashboard">${ico('chev-left')} Back</button>
          <div class="divider"></div>
          <div class="scoping">Scanning to</div>
          <div class="deck-pill-dark"><span class="dot"></span>Chem 101 ${ico('chev-down')}</div>
          <div class="mode-toggle" role="tablist" aria-label="OCR model">
            ${OCR_MODELS.map(m => `<button class="mode-opt ${RT.ocrModel===m.id?'is-active':''}" data-model="${m.id}" role="tab" aria-selected="${RT.ocrModel===m.id}">${m.label}</button>`).join('')}
          </div>
          <div class="right-tools">
            <span class="tool-pill" id="cam-status"><span class="live" id="cam-dot"></span><span id="cam-text">Starting…</span></span>
            <span class="tool-pill" id="ocr-status"><span class="live" id="ocr-dot" style="background:#8B8B86;box-shadow:none"></span><span id="ocr-text">OCR</span></span>
            <span class="tool-pill" id="llm-status"><span class="live" id="llm-dot" style="background:#8B8B86;box-shadow:none"></span><span id="llm-text">LLM</span></span>
            <button class="tool-pill close" data-route="#dashboard">${ico('x')}</button>
          </div>
        </div>

        <div class="reticle">
          <span class="corner tl"></span>
          <span class="corner tr"></span>
          <span class="corner bl"></span>
          <span class="corner br"></span>
        </div>

        <div class="scan-bottom">
          <div class="recognized" id="scan-hint">Frame the problem inside the corners and press space to capture</div>
          <div class="formula" id="scan-formula">—</div>
          <div class="capture-row">
            <div class="capture-aux">
              <button class="icon-btn" title="Upload">${ico('image')}</button>
              <span>Upload</span>
            </div>
            <div class="capture-stack">
              <button class="capture-btn" id="capture-btn" title="Capture"></button>
              <div class="capture-btn-label">CAPTURE · SPACE</div>
            </div>
            <div class="capture-aux">
              <button class="icon-btn" title="Settings">${ico('gear')}</button>
              <span>Settings</span>
            </div>
          </div>
        </div>
      </section>

      <aside class="detection-panel">
        <div class="live-label">Transcript</div>
        <div>
          <div class="detection-name" id="panel-name">Nothing read yet</div>
          <div class="detection-formula" id="panel-sub">
            <span>Capture a photo to run OCR</span>
          </div>
        </div>
        <div>
          <div class="confidence-row">
            <span class="label">Mean OCR confidence</span>
            <span class="v" id="confidence-pct">—</span>
          </div>
          <div class="confidence-bar"><span id="confidence-fill" style="width:0%"></span></div>
        </div>

        <div class="alt-section">
          <div class="alt-label">Lines read</div>
          <div id="line-list">
            <div class="alt-row">
              <div class="alt-mark">··</div>
              <div class="alt-info">
                <div class="alt-name" style="color:var(--muted)">No lines yet</div>
                <div class="alt-sub">Each line EasyOCR reads appears here with its confidence</div>
              </div>
            </div>
          </div>
        </div>

        <div class="tip-card">
          <div class="head">Pipeline</div>
          <div class="body">The selected OCR model (EasyOCR, or Pix2Text for math) reads the captured photo. The local LLM (Ollama) then writes the card from the transcript.</div>
          <div class="foot" id="pipeline-foot">Checking services…</div>
        </div>
      </aside>
    </div>`;

  startScanController(root).catch(err => {
    console.error('[scan] fatal:', err);
    showScanError(root, err);
  });
}

async function startScanController(root) {
  if (!window.LensML) {
    // If <script type="module"> hasn't loaded yet, wait for it
    await new Promise(resolve => window.addEventListener('lens-ml-ready', resolve, { once: true }));
  }
  const { Llm, Ocr, Camera } = window.LensML;

  const videoEl   = $('#scan-video', root);
  const panelName = $('#panel-name', root);
  const panelSub  = $('#panel-sub', root);
  const confPct   = $('#confidence-pct', root);
  const confFill  = $('#confidence-fill', root);
  const lineList  = $('#line-list', root);
  const scanHint  = $('#scan-hint', root);
  const scanFormula = $('#scan-formula', root);
  const llmText   = $('#llm-text', root);
  const llmDot    = $('#llm-dot', root);
  const pipeFoot  = $('#pipeline-foot', root);

  const setPill = (id, text, color) => {
    $(`#${id}-text`, root).textContent = text;
    const dot = $(`#${id}-dot`, root);
    dot.style.background = color;
    dot.style.boxShadow = color === '#2ecc71' ? '0 0 0 3px rgba(46,204,113,.18)' : 'none';
  };

  // 1. Start camera
  setPill('cam', 'Opening camera…', '#E5A23A');
  const cam = new Camera();
  try {
    await cam.start(videoEl);
    setPill('cam', 'Camera', '#2ecc71');
  } catch (err) {
    console.error('[camera]', err);
    setPill('cam', 'No camera', '#E97352');
    panelName.textContent = 'Camera unavailable';
    panelSub.innerHTML = `<span>${cameraErrorMessage(err)}</span>`;
    pipeFoot.textContent = 'Start the page from https:// or http://localhost so the browser grants camera access.';
    return;
  }

  // 2. Probe the OCR service and Ollama in parallel (non-blocking)
  // OCR status reflects the selected model: missing, still loading, or ready.
  let ocrModels = null;
  let stopped = false;   // set when the user leaves the scan view
  const refreshOcrPill = () => {
    const label = ocrModelLabel(RT.ocrModel);
    if (!ocrModels) { setPill('ocr', 'OCR offline', '#E97352'); return; }
    const m = ocrModels[RT.ocrModel];
    if (!m)             setPill('ocr', `${label} · not installed`, '#E97352');
    else if (!m.loaded) setPill('ocr', `${label} · loading…`, '#E5A23A');
    else                setPill('ocr', `${label} · ready`, '#2ecc71');
  };
  const checkOcr = async () => {
    const h = await Ocr.health();
    RT.ocrOk = !!h.ok;
    ocrModels = h.ok ? h.models : null;
    refreshOcrPill();
    // A model that is still loading flips to ready on its own; poll until then.
    if (!stopped && ocrModels && Object.values(ocrModels).some(m => !m.loaded)) setTimeout(checkOcr, 3000);
  };
  setPill('ocr', 'OCR · checking…', '#E5A23A');
  checkOcr();
  checkOllama(llmText, llmDot, pipeFoot);

  root.querySelectorAll('.mode-opt[data-model]').forEach(b => b.addEventListener('click', () => {
    setOcrModel(b.dataset.model);
    root.querySelectorAll('.mode-opt[data-model]').forEach(x => {
      const on = x.dataset.model === RT.ocrModel;
      x.classList.toggle('is-active', on);
      x.setAttribute('aria-selected', on);
    });
    refreshOcrPill();
  }));

  // 3. Capture: one frame -> EasyOCR -> show lines -> LLM writes the card
  const capture = async () => {
    const btn = $('#capture-btn', root);
    if (btn) { btn.style.transform = 'scale(.92)'; setTimeout(() => btn.style.transform = '', 120); }

    if (!RT.ocrOk) {
      scanHint.textContent = 'OCR server not reachable — start python OCR/ocr_serve.py';
      return;
    }
    if (ocrModels && !ocrModels[RT.ocrModel]) {
      scanHint.textContent = `${ocrModelLabel(RT.ocrModel)} is not installed on the OCR server — pip install -r OCR/requirements.txt`;
      return;
    }
    const full = cam.captureFrame(720);
    if (!full) { scanHint.textContent = 'Camera frame unavailable'; return; }

    scanHint.textContent = `Reading text with ${ocrModelLabel(RT.ocrModel)}…`;
    let result;
    try {
      result = await Ocr.extractText(full, { conf: 0.5, model: RT.ocrModel });
    } catch (err) {
      scanHint.textContent = `OCR error: ${(err.message || err).slice(0, 120)}`;
      return;
    }

    if (!result.rawText.trim()) {
      scanHint.textContent = 'No text found — try holding the camera steadier or closer';
      return;
    }

    renderTranscript(result);
    scanHint.textContent = `Found: "${result.rawText.slice(0, 60)}${result.rawText.length > 60 ? '…' : ''}"`;
    scanFormula.textContent = `${result.lines.length} line${result.lines.length === 1 ? '' : 's'} · ${result.inferenceMs.toFixed(0)} ms`;

    if (!RT.ollamaOk) {
      scanHint.textContent = 'Ollama unreachable — start "ollama serve" then try again';
      return;
    }

    const topic = result.rawText.slice(0, 200);
    const id    = `ocr-${Date.now()}`;
    const meta  = {
      id,
      displayName: topic.slice(0, 60),
      subject: 'TEXT · OCR',
      grad: 'var(--grad-physics)',
      kind: 'ocr',
    };

    showGenOverlay(meta.displayName);
    try {
      const flashcard = await Llm.generateFlashcard({
        topic,
        hintPrompt: `This text was extracted from a photo via OCR: "${topic}"`,
      });
      RT.generatedCards[id] = {
        id,
        crumbs: ['Scanned', 'Text · OCR'],
        subject: flashcard.subject || 'TEXT · OCR',
        name: flashcard.name,
        formula: flashcard.formula,
        mass: flashcard.mass,
        grad: 'var(--grad-physics)',
        scanned: 'JUST NOW · OCR SCAN',
        reviewWhen: 'Tomorrow',
        reviewAt: '9:00 AM',
        reviewProgress: { done: 0, total: 4 },
        oneline: flashcard.oneline,
        facts: flashcard.facts,
        ocrText: result.rawText,   // raw OCR output, kept for later scoring
        ocrModel: result.model,
        ocrModelVersion: result.modelVersion,
        generated: true,
      };
      recordCapture({ id, meta, isGenerated: true });
      hideGenOverlay();
      navigate(`#flashcard?id=${id}`);
    } catch (err) {
      hideGenOverlay();
      scanHint.textContent = `Generation failed: ${(err.message || err).slice(0, 100)}`;
    }
  };

  // Show what OCR read, line by line, with per-line confidence.
  const renderTranscript = (result) => {
    const lines = result.lines || [];
    const meanConf = lines.length ? lines.reduce((a, l) => a + (l.confidence || 0), 0) / lines.length : 0;
    const pct = Math.round(meanConf * 100);
    panelName.textContent = 'Text detected';
    panelSub.innerHTML = `<span class="strong">${escapeHtml(ocrModelLabel(result.model))}</span><span>·</span><span>${lines.length} line${lines.length === 1 ? '' : 's'}</span><span>·</span><span style="font-family:var(--font-mono)">${result.inferenceMs.toFixed(0)} ms</span>`;
    confPct.textContent = lines.length ? pct + '%' : '—';
    confFill.style.width = pct + '%';
    lineList.innerHTML = lines.map((l, i) => `
      <div class="alt-row">
        <div class="alt-mark">${String(i + 1).padStart(2, '0')}</div>
        <div class="alt-info">
          <div class="alt-name" ${l.kind && l.kind !== 'text' ? 'style="font-family:var(--font-mono)"' : ''}>${escapeHtml(l.text)}</div>
          ${l.kind && l.kind !== 'text' ? `<div class="alt-sub">${l.kind === 'math' ? 'Math (LaTeX)' : 'Text + math (LaTeX)'}</div>` : ''}
        </div>
        <div class="alt-pct">${Math.round((l.confidence || 0) * 100)}%</div>
      </div>`).join('');
  };

  $('#capture-btn', root).addEventListener('click', capture);

  // 4. Register a scan controller so we can tear down on navigate-away
  RT.scan = {
    stop: () => { stopped = true; cam.stop(); },
    capture,
  };
}

// OCR models the scan view can switch between (ids match OCR/engines.py).
const OCR_MODELS = [
  { id: 'easyocr',  label: 'EasyOCR' },
  { id: 'pix2text', label: 'Pix2Text' },
];
function ocrModelLabel(id) {
  return (OCR_MODELS.find(m => m.id === id) || {}).label || id || 'OCR';
}
function setOcrModel(id) {
  RT.ocrModel = OCR_MODELS.some(m => m.id === id) ? id : 'easyocr';
  try { localStorage.setItem('lens.ocrModel', RT.ocrModel); } catch {}
}

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c])); }

// ----- Flashcard tab panels (quiz history + sources) -----
function renderQuizHistoryPanel(card) {
  const sessions = RT.quizSessions.filter(s => (s.cardIds || []).includes(card.id));
  if (!sessions.length) {
    return `
      <div class="oneline">Quiz history</div>
      <p class="oneline-text">No sessions yet for this card. Click <strong>Quiz me</strong> above to run one — your score will show up here.</p>
    `;
  }
  const totQ = sessions.reduce((a, s) => a + s.total, 0);
  const totC = sessions.reduce((a, s) => a + s.correct, 0);
  const overallPct = totQ ? Math.round((totC / totQ) * 100) : 0;

  const rows = sessions.map(s => {
    const pct = s.total ? Math.round((s.correct / s.total) * 100) : 0;
    const when = relativeTime(s.at);
    return `
      <div class="history-row">
        <div class="history-when">${escapeHtml(when)}</div>
        <div class="history-bar"><span style="width:${pct}%"></span></div>
        <div class="history-score">${s.correct} / ${s.total}</div>
        <div class="history-pct">${pct}%</div>
      </div>`;
  }).join('');

  return `
    <div class="oneline">Quiz history</div>
    <div class="history-summary">
      <div class="history-hero">
        <span class="hl">${overallPct}%</span>
        <span class="history-hero-sub">across ${sessions.length} session${sessions.length===1?'':'s'}</span>
      </div>
      <div class="history-hero-meta">${totC} correct · ${totQ - totC} missed · ${totQ} question${totQ===1?'':'s'} total</div>
    </div>
    <div class="history-list">
      <div class="history-head">
        <div>When</div><div>Score</div><div></div><div class="t-right">%</div>
      </div>
      ${rows}
    </div>
    <div class="history-actions">
      <button class="btn btn-primary" data-action="quiz-me" data-card-id="${card.id}"><span class="dot"></span>Quiz again</button>
    </div>
  `;
}

function renderSourcesPanel(card) {
  const subject = (card.subject || '').split(' · ')[0].trim().toUpperCase();
  const sources = sourcesForSubject(subject, card);
  const linksHtml = sources.map(s => `
    <a class="source-row" href="${s.href}" target="_blank" rel="noopener noreferrer">
      <div class="source-mark" style="background:${s.swatch}">${s.tag}</div>
      <div class="source-info">
        <div class="source-title">${escapeHtml(s.title)}</div>
        <div class="source-sub">${escapeHtml(s.sub)}</div>
      </div>
      <div class="source-arrow">${ico('chev-right')}</div>
    </a>`).join('');

  const prov = card.generated
    ? 'Read from a photo by EasyOCR, then drafted by a local Phi-4-mini model via Ollama — review for accuracy.'
    : 'Hand-authored reference card in the Lens corpus.';

  return `
    <div class="oneline">Sources</div>
    <p class="oneline-text">${prov} References below.</p>
    <div class="source-list">${linksHtml}</div>
  `;
}

// Subject-driven list of public reference sources. Each card's queries are
// URL-encoded against its display name so links land on a relevant page.
function sourcesForSubject(subject, card) {
  const q = encodeURIComponent(card.name);
  const wiki = {
    title: 'Wikipedia',
    sub: `Search: ${card.name}`,
    tag: 'WP',
    swatch: 'var(--grad-arch)',
    href: `https://en.wikipedia.org/wiki/Special:Search?search=${q}`,
  };
  const wikidata = {
    title: 'Wikidata',
    sub: 'Structured entity data',
    tag: 'WD',
    swatch: 'var(--grad-physics)',
    href: `https://www.wikidata.org/w/index.php?search=${q}`,
  };
  const pubchem = {
    title: 'PubChem',
    sub: 'NIH chemistry database',
    tag: 'PC',
    swatch: 'var(--grad-blue)',
    href: `https://pubchem.ncbi.nlm.nih.gov/#query=${q}`,
  };
  const chemspider = {
    title: 'ChemSpider',
    sub: 'Royal Society of Chemistry',
    tag: 'CS',
    swatch: 'var(--grad-violet)',
    href: `https://www.chemspider.com/Search.aspx?q=${q}`,
  };
  const ncbi = {
    title: 'NCBI / MeSH',
    sub: 'Biomedical reference',
    tag: 'NC',
    swatch: 'var(--grad-orange)',
    href: `https://www.ncbi.nlm.nih.gov/pubmed/?term=${q}`,
  };
  const itis = {
    title: 'ITIS',
    sub: 'Taxonomic reference',
    tag: 'IT',
    swatch: 'var(--grad-green)',
    href: `https://www.itis.gov/servlet/SingleRpt/SingleRpt?search_topic=Scientific_Name&search_value=${q}`,
  };
  const mindat = {
    title: 'Mindat',
    sub: 'Mineralogy database',
    tag: 'MD',
    swatch: 'var(--grad-violet)',
    href: `https://www.mindat.org/search.php?search=${q}`,
  };
  const nasa = {
    title: 'NASA Science',
    sub: 'Planetary reference',
    tag: 'NA',
    swatch: 'var(--grad-astro)',
    href: `https://science.nasa.gov/?search=${q}`,
  };
  const archnet = {
    title: 'Archnet',
    sub: 'Architecture archive (MIT)',
    tag: 'AN',
    swatch: 'var(--grad-arch)',
    href: `https://www.archnet.org/search?q=${q}`,
  };
  const wikiart = {
    title: 'WikiArt',
    sub: 'Art history index',
    tag: 'WA',
    swatch: 'var(--grad-art)',
    href: `https://www.wikiart.org/en/search/${q}`,
  };

  switch (subject) {
    case 'CHEMISTRY':   return [pubchem, chemspider, wiki, wikidata];
    case 'BIOLOGY':
    case 'ANATOMY':     return [ncbi, wiki, wikidata];
    case 'BOTANY':
    case 'ZOOLOGY':     return [itis, ncbi, wiki];
    case 'GEOLOGY':     return [mindat, wiki, wikidata];
    case 'ASTRONOMY':   return [nasa, wiki, wikidata];
    case 'ARCHITECTURE': return [archnet, wiki, wikidata];
    case 'ART HISTORY': return [wikiart, wiki, wikidata];
    case 'PHYSICS':     return [wiki, wikidata];
    default:            return [wiki, wikidata];
  }
}

function cameraErrorMessage(err) {
  switch (err.code) {
    case 'denied':      return 'Camera permission denied — grant access and reload.';
    case 'no-device':   return 'No camera detected on this device.';
    case 'unsupported': return 'Camera requires HTTPS or http://localhost.';
    default:            return err.message || 'Camera unavailable.';
  }
}

async function checkOllama(llmText, llmDot, pipeFoot) {
  const { Llm } = window.LensML;
  try {
    const h = await Llm.health();
    if (h.ok) {
      RT.ollamaOk = true;
      llmText.textContent = h.hasConfiguredModel ? `${Llm.getConfig().model} · ready` : `LLM · pull ${Llm.getConfig().model}`;
      llmDot.style.background = h.hasConfiguredModel ? '#2ecc71' : '#E5A23A';
      llmDot.style.boxShadow = h.hasConfiguredModel ? '0 0 0 3px rgba(46,204,113,.18)' : 'none';
      if (h.hasConfiguredModel) {
        if (pipeFoot) pipeFoot.textContent = `${Llm.getConfig().model} will write a flashcard on capture.`;
      } else {
        if (pipeFoot) pipeFoot.textContent = `Ollama is running, but \`${Llm.getConfig().model}\` isn't installed. Run: ollama pull ${Llm.getConfig().model}`;
      }
    } else {
      RT.ollamaOk = false;
      llmText.textContent = 'LLM offline';
      llmDot.style.background = '#8B8B86';
      llmDot.style.boxShadow = 'none';
      if (pipeFoot) pipeFoot.textContent = 'Ollama not reachable at localhost:11434. Known flashcards still work.';
    }
  } catch (err) {
    RT.ollamaOk = false;
    llmText.textContent = 'LLM offline';
  }
}

function showGenOverlay(topic) {
  const ov = document.getElementById('gen-overlay');
  document.getElementById('gen-topic').textContent = topic;
  document.getElementById('gen-sub').textContent = 'Phi-4-mini is composing the explanation and key facts.';
  ov.hidden = false;
}
function hideGenOverlay() {
  const ov = document.getElementById('gen-overlay');
  if (ov) ov.hidden = true;
}

function showScanError(root, err) {
  const hint = $('#scan-hint', root);
  if (hint) hint.textContent = `Error: ${err.message || err}`;
}

// ---------- Quiz (new) ----------
// Generates a self-graded quiz from a card's facts. Each fact becomes a question:
// the LABEL is the prompt, the BODY is the "answer". The user reads the question,
// hits "Show answer", then self-grades via "I knew it" / "Try again".
function renderQuiz(root, params) {
  const mode = params.get('mode');
  const id = params.get('id');

  // Decide the card set. Single-card mode = just that card. Review mode = all
  // currently-due cards (live computation), falling back to copper-sulfate so
  // the user has *something* to study even on a fresh install.
  let cards = [];
  if (mode === 'review') {
    const ids = deriveDueToday().map(d => d.id).filter(i => D.flashcards[i] || RT.generatedCards[i]);
    cards = ids.map(i => D.flashcards[i] || RT.generatedCards[i]);
    if (!cards.length) cards = [D.flashcards['copper-sulfate']];
  } else {
    const card = D.flashcards[id] || RT.generatedCards[id] || D.flashcards['copper-sulfate'];
    cards = [card];
  }

  // Flatten every fact across all cards into one quiz queue.
  const queue = [];
  for (const c of cards) {
    for (const f of (c.facts || [])) {
      queue.push({ cardId: c.id, cardName: c.name, grad: c.grad, label: f.label, body: f.body });
    }
  }
  if (!queue.length) {
    root.innerHTML = `<div class="app-shell">${renderSidebar('')}<div class="main">${renderTopbar({ withSearch: false })}<div class="quiz-page"><div class="quiz-empty">No questions available.</div></div></div></div>`;
    return;
  }

  const state = { i: 0, correct: 0, revealed: false };

  const mount = () => {
    const q = queue[state.i];
    const progressPct = Math.round((state.i / queue.length) * 100);

    root.innerHTML = `
      <div class="app-shell">
        ${renderSidebar('')}
        <div class="main">
          ${renderTopbar({ withSearch: false, rightSlot: `
            <div class="crumbs" style="margin-right:auto">
              <span>Quiz</span><span class="sep">›</span>
              <span class="now">${escapeHtml(q.cardName)}</span>
            </div>
            <button class="btn" data-route="#flashcard?id=${q.cardId}">${ico('x')} Exit quiz</button>
          ` })}
          <div class="quiz-page">
            <div class="quiz-shell">
              <div class="quiz-meta">
                <div class="quiz-count">Question ${state.i + 1} of ${queue.length}</div>
                <div class="quiz-score">Score ${state.correct} / ${state.i}</div>
              </div>
              <div class="quiz-progress"><span style="width:${progressPct}%"></span></div>

              <div class="quiz-card-wrap">
                <button class="quiz-card-flip" id="quiz-flip" aria-label="Reveal answer">
                  <div class="quiz-card-face front" style="background:${q.grad}">
                    <div class="quiz-eyebrow">${escapeHtml(q.cardName)}</div>
                    <div class="quiz-question">${escapeHtml(q.label)}</div>
                    <div class="quiz-face-foot">Tap card or press <kbd>space</kbd> to reveal</div>
                  </div>
                  <div class="quiz-card-face back" style="background:${q.grad}">
                    <div class="quiz-eyebrow">Answer</div>
                    <div class="quiz-answer-text">${escapeHtml(q.body)}</div>
                    <div class="quiz-face-foot">Grade yourself below · <kbd>1</kbd> wrong · <kbd>2</kbd> right</div>
                  </div>
                </button>
              </div>

              <div class="quiz-actions" id="quiz-actions">
                <button class="btn btn-primary" data-quiz-reveal><span class="dot"></span>Show answer</button>
                <span class="quiz-hint">Click the card or press <kbd>space</kbd> to flip</span>
              </div>
            </div>
          </div>
        </div>
      </div>`;

    const flip = root.querySelector('#quiz-flip');
    const actions = root.querySelector('#quiz-actions');

    const reveal = () => {
      if (state.revealed) return;
      state.revealed = true;
      flip.classList.add('is-flipped');
      actions.innerHTML = `
        <button class="btn" data-quiz-grade="wrong">Try again</button>
        <button class="btn btn-accent" data-quiz-grade="right"><span class="dot"></span>I knew it</button>
        <span class="quiz-hint">Self-grade · <kbd>1</kbd> wrong · <kbd>2</kbd> right</span>
      `;
      actions.querySelectorAll('[data-quiz-grade]').forEach(b =>
        b.addEventListener('click', (e) => { e.stopPropagation(); grade(b.dataset.quizGrade === 'right'); })
      );
    };

    flip.addEventListener('click', reveal);
    actions.querySelector('[data-quiz-reveal]')?.addEventListener('click', (e) => { e.stopPropagation(); reveal(); });
  };

  const grade = (correct) => {
    if (correct) state.correct++;
    state.i++;
    state.revealed = false;
    if (state.i >= queue.length) {
      finish();
    } else {
      mount();
    }
  };

  const finish = () => {
    RT.quizSessions.unshift({
      cardIds: [...new Set(queue.map(q => q.cardId))],
      correct: state.correct,
      total: queue.length,
      at: Date.now(),
    });
    persistRT();
    const pct = Math.round((state.correct / queue.length) * 100);
    root.innerHTML = `
      <div class="app-shell">
        ${renderSidebar('')}
        <div class="main">
          ${renderTopbar({ withSearch: false, rightSlot: `
            <button class="btn" data-route="#dashboard">Done</button>
          ` })}
          <div class="quiz-page">
            <div class="quiz-result">
              <div class="eyebrow">Session complete</div>
              <h1 class="quiz-result-title"><span class="hl">${state.correct} / ${queue.length}</span></h1>
              <p class="quiz-result-sub">${pct}% retention · ${queue.length} question${queue.length===1?'':'s'} across ${new Set(queue.map(q=>q.cardId)).size} card${new Set(queue.map(q=>q.cardId)).size===1?'':'s'}.</p>
              <div class="quiz-actions">
                <button class="btn" data-route="#dashboard">Back to dashboard</button>
                <button class="btn btn-primary" data-action="start-review"><span class="dot"></span>Quiz again</button>
              </div>
            </div>
          </div>
        </div>
      </div>`;
  };

  // Keyboard: space reveals (flip), 1 = wrong, 2 = right, esc exits.
  const onKey = (e) => {
    if (window.location.hash.split('?')[0] !== '#quiz') { document.removeEventListener('keydown', onKey, true); return; }
    if ($('#search-overlay')?.hidden === false) return;
    if (e.code === 'Space' && !state.revealed) {
      e.preventDefault();
      const flipEl = root.querySelector('#quiz-flip');
      flipEl && flipEl.click();
    } else if (state.revealed && (e.key === '1' || e.key === '2')) {
      e.preventDefault();
      grade(e.key === '2');
    }
  };
  document.addEventListener('keydown', onKey, true);

  mount();
}

// ---------- Flashcard detail (W3) ----------
function renderFlashcard(root, params) {
  const id = params.get('id') || 'copper-sulfate';
  const card = D.flashcards[id] || RT.generatedCards[id] || D.flashcards['copper-sulfate'];

  const right = `
    <button class="btn">${ico('share')} Share</button>
    <button class="btn btn-primary" data-action="quiz-me" data-card-id="${id}"><span class="dot"></span>Quiz me</button>
  `;

  const factsHtml = card.facts.map(f => `
    <div class="fact">
      <div><span class="fact-num">${f.num}</span><span class="fact-label">${f.label}</span></div>
      <div class="fact-body">${f.body}</div>
    </div>`).join('');

  const progressPct = (card.reviewProgress.done / card.reviewProgress.total) * 100;

  root.innerHTML = `
    <div class="app-shell">
      ${renderSidebar('')}
      <div class="main">
        ${renderTopbar({ withSearch: false, rightSlot: `
          <div class="crumbs" style="margin-right:auto">
            <span>${card.crumbs[0]}</span><span class="sep">›</span>
            <span>${card.crumbs[1]}</span><span class="sep">›</span>
            <span class="now">${card.name}</span>
          </div>
          ${right}
        ` })}
        <div class="detail-page">
          <div class="detail-grid">
            <div>
              <div class="card-cover" style="background:${card.grad}">
                <div class="scan-tag">${card.scanned}</div>
                <div class="blob"></div>
                <div class="pager">
                  <span class="pip active"></span>
                  <span class="pip"></span>
                  <span class="pip"></span>
                </div>
              </div>
              <div class="repetition">
                <div class="label">Spaced repetition</div>
                <div class="when">${card.reviewWhen} <span class="at">${card.reviewAt}</span></div>
                <div class="progress"><span style="width:${progressPct}%"></span></div>
                <div class="progress-meta">
                  <span>Reviews</span>
                  <span>${card.reviewProgress.done} / ${card.reviewProgress.total}</span>
                </div>
              </div>
            </div>

            <div>
              <div class="subject-eyebrow">
                ${card.subject.split(' · ').map((s,i,a) => `<span>${s}</span>${i<a.length-1?'<span class="dot">·</span>':''}`).join('')}
              </div>
              <h1 class="subject-name"><span class="hl">${card.name}</span></h1>
              <div class="subject-formula"><span>${card.formula}</span><span class="mass">${card.mass}</span></div>
              <div class="detail-tabs">
                <button class="detail-tab active" data-tab="overview">Overview</button>
                <button class="detail-tab" data-tab="notes">Notes</button>
                <button class="detail-tab" data-tab="quiz">Quiz history</button>
                <button class="detail-tab" data-tab="sources">Sources</button>
              </div>
              <div class="tab-panel" id="panel">
                <div class="oneline">In one line</div>
                <p class="oneline-text">${card.oneline}</p>
                <div class="fact-grid">${factsHtml}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>`;

  // Tab switching — rebuilt per click so live data (quiz history) stays fresh.
  root.querySelectorAll('.detail-tab').forEach(t => t.addEventListener('click', () => {
    root.querySelectorAll('.detail-tab').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    const tab = t.dataset.tab;
    if (tab === 'overview') {
      $('#panel', root).innerHTML = `
        <div class="oneline">In one line</div>
        <p class="oneline-text">${card.oneline}</p>
        <div class="fact-grid">${factsHtml}</div>`;
    } else if (tab === 'notes') {
      $('#panel', root).innerHTML = `
        <div class="oneline">Your notes</div>
        <p class="oneline-text">No notes yet — press <kbd>N</kbd> to start writing.</p>`;
    } else if (tab === 'quiz') {
      $('#panel', root).innerHTML = renderQuizHistoryPanel(card);
    } else {
      $('#panel', root).innerHTML = renderSourcesPanel(card);
    }
  }));

  // Pager pips - cycle on click
  const pips = root.querySelectorAll('.pager .pip');
  pips.forEach((p, i) => p.addEventListener('click', () => {
    pips.forEach(x => x.classList.remove('active'));
    p.classList.add('active');
  }));
}

// ---------- Global delegation: routing, search, deck clicks ----------
document.addEventListener('click', (e) => {
  const route = e.target.closest('[data-route]');
  if (route) { navigate(route.dataset.route); return; }
  const newScan = e.target.closest('[data-action="new-scan"]');
  if (newScan) { navigate('#scan'); return; }
  const startReview = e.target.closest('[data-action="start-review"]');
  if (startReview) { navigate('#quiz?mode=review'); return; }
  const quizMe = e.target.closest('[data-action="quiz-me"]');
  if (quizMe) { navigate(`#quiz?id=${quizMe.dataset.cardId}`); return; }
  const scanCard = e.target.closest('[data-scan]');
  if (scanCard) { navigate(`#flashcard?id=${scanCard.dataset.scan}`); return; }
  const deckCard = e.target.closest('[data-deck]');
  if (deckCard) {
    navigate(`#flashcard?id=${cardForDeck(deckCard.dataset.deck)}`);
    return;
  }
  const openSearch = e.target.closest('[data-action="open-search"]');
  if (openSearch) { openSearchOverlay(); }
});

// ---------- Search overlay ----------
function openSearchOverlay() {
  const overlay = $('#search-overlay');
  const input = $('#search-input');
  overlay.hidden = false;
  input.value = '';
  renderSearchResults('');
  setTimeout(() => input.focus(), 0);
}
function closeSearchOverlay() { $('#search-overlay').hidden = true; }

function searchCorpus(q) {
  q = q.trim().toLowerCase();
  // Live captures first, mock recent-scans as cold-start filler, then
  // derived decks (no hardcoded D.decks).
  const liveCards = RT.captures.map(c => ({
    kind: 'card', id: c.id, title: c.topic, sub: c.subject || '', swatch: c.grad,
  }));
  const liveIds = new Set(liveCards.map(c => c.id));
  const seedCards = D.recentScans
    .filter(s => !liveIds.has(s.id))
    .map(s => ({ kind: 'card', id: s.id, title: s.title, sub: s.sub, swatch: s.grad }));
  const liveDecks = deriveDecks().map(d => ({
    kind: 'deck', id: d.id, title: d.name, sub: d.subject, swatch: d.grad,
  }));
  const items = [...liveCards, ...seedCards, ...liveDecks];
  if (!q) return items.slice(0, 6);
  return items.filter(i => i.title.toLowerCase().includes(q) || (i.sub||'').toLowerCase().includes(q)).slice(0, 8);
}

function renderSearchResults(q) {
  const list = $('#search-results');
  const items = searchCorpus(q);
  list.innerHTML = items.map((i, idx) => `
    <button class="search-result ${idx===0?'is-active':''}" data-kind="${i.kind}" data-id="${i.id}">
      <div class="swatch" style="background:${i.swatch}"></div>
      <div>
        <div class="title">${i.title}</div>
      </div>
      <span class="sub">${i.sub || ''}</span>
    </button>`).join('') || `<div class="search-result"><span class="sub">No matches.</span></div>`;
}

document.addEventListener('input', (e) => {
  if (e.target.id === 'search-input') renderSearchResults(e.target.value);
});
document.addEventListener('click', (e) => {
  const r = e.target.closest('.search-result');
  if (!r) return;
  const kind = r.dataset.kind;
  const id = r.dataset.id;
  closeSearchOverlay();
  if (kind === 'card') navigate(`#flashcard?id=${id}`);
  if (kind === 'deck') navigate(`#flashcard?id=${cardForDeck(id)}`);
});

// ---------- Global keys ----------
function attachGlobalKeys() {
  document.onkeydown = (e) => {
    const overlay = $('#search-overlay');
    const overlayOpen = !overlay.hidden;

    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openSearchOverlay(); return; }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') { e.preventDefault(); navigate('#scan'); return; }
    if (e.key === 'Escape') { if (overlayOpen) { closeSearchOverlay(); return; } }
    if (e.code === 'Space' && window.location.hash === '#scan' && !overlayOpen) {
      e.preventDefault();
      const btn = $('#capture-btn');
      btn && btn.click();
      return;
    }
    if (overlayOpen && (e.key === 'Enter' || e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
      const items = Array.from(document.querySelectorAll('.search-result'));
      let idx = items.findIndex(x => x.classList.contains('is-active'));
      if (e.key === 'Enter') { items[idx]?.click(); return; }
      idx = Math.max(0, Math.min(items.length - 1, idx + (e.key === 'ArrowDown' ? 1 : -1)));
      items.forEach(x => x.classList.remove('is-active'));
      items[idx]?.classList.add('is-active');
      e.preventDefault();
    }
  };
}

// click outside overlay closes it
$('#search-overlay').addEventListener('click', (e) => { if (e.target.id === 'search-overlay') closeSearchOverlay(); });

// ---------- Icons ----------
function ico(name) {
  const m = {
    home: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="m3 11 9-8 9 8"/><path d="M5 10v9a1 1 0 0 0 1 1h4v-7h4v7h4a1 1 0 0 0 1-1v-9"/></svg>`,
    decks: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="8" height="18" rx="1.5"/><rect x="13" y="3" width="8" height="18" rx="1.5"/></svg>`,
    clock: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>`,
    chart: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17 9 11l4 4 8-8"/><path d="M21 7v4h-4"/></svg>`,
    chat: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a8 8 0 1 1-3.5-6.6L21 4l-1.4 3.5A8 8 0 0 1 21 12Z"/></svg>`,
    dots: `<svg width="16" height="4" viewBox="0 0 16 4" fill="currentColor"><circle cx="2" cy="2" r="1.5"/><circle cx="8" cy="2" r="1.5"/><circle cx="14" cy="2" r="1.5"/></svg>`,
    search: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>`,
    plus: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>`,
    bell: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M6 8a6 6 0 1 1 12 0c0 7 3 7 3 9H3c0-2 3-2 3-9Z"/><path d="M10 21a2 2 0 0 0 4 0"/></svg>`,
    import: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8" opacity=".4"/></svg>`,
    sort: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M6 12h12M9 18h6"/></svg>`,
    arrow: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg>`,
    'chev-left': `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m15 6-6 6 6 6"/></svg>`,
    'chev-right': `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m9 6 6 6-6 6"/></svg>`,
    'chev-down': `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>`,
    bolt: `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 4 14h7l-1 8 9-12h-7l1-8z"/></svg>`,
    x: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M6 6l12 12M18 6 6 18"/></svg>`,
    image: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.5"/><path d="m4 18 5-5 4 4 3-3 4 4"/></svg>`,
    gear: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 13.5a7.4 7.4 0 0 0 0-3l2.1-1.6-2-3.5-2.5.9a7.6 7.6 0 0 0-2.6-1.5L14 2h-4l-.4 2.7a7.6 7.6 0 0 0-2.6 1.5l-2.5-.9-2 3.5L4.6 10.5a7.4 7.4 0 0 0 0 3l-2.1 1.6 2 3.5 2.5-.9a7.6 7.6 0 0 0 2.6 1.5L10 22h4l.4-2.7a7.6 7.6 0 0 0 2.6-1.5l2.5.9 2-3.5z"/></svg>`,
    share: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M5 14v5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-5"/></svg>`,
    check: `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 5 5L20 7"/></svg>`
  };
  return m[name] || '';
}
