// Pure logic behind the AI generator page (ai-page.js): what a lane, the
// patterns column and the header show, the model problems, the leave
// question and what keep says. No DOM.

export const CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8];
export const LANE_FIELDS = ['type', 'candidates', 'candidate', 'name', 'trying', 'generating', 'error'];
const KINDS = ['patch', 'pattern'];

/** What the standalone app (no pip, ai.install_command empty) says instead of the install command. */
export const AI_EDITION_TEXT = 'Download the Pythonic + AI edition from the releases page: '
  + 'https://github.com/jeromeboivin/pythonic/releases';

/** The address of a lane field: ai.ch<N>.<field>. */
export const laneAddress = (channel, field) => `ai.ch${channel}.${field}`;

/** Every address the page reads. */
export function aiAddresses() {
  return [
    'ai.available', 'ai.install_command', 'ai.installing', 'ai.state', 'ai.models', 'ai.tried',
    'ai.bank', 'ai.bank_chains', 'ai.preview', 'pref.ai.patch_temperature', 'pref.ai.pattern_temperature',
    ...CHANNELS.flatMap((n) => LANE_FIELDS.map((f) => laneAddress(n, f))),
  ];
}

/**
 * What a lane shows, from its fields ({candidates, name, trying, generating,
 * error}): state 'generating' | 'ready' | 'error' | 'empty', the new sound's
 * name, whether old / new can switch, which one plays and a note.
 */
export function laneView(lane = {}) {
  const ready = Number(lane.candidates) > 0;
  const trying = !!lane.trying;
  let state = ready ? 'ready' : 'empty';
  if (lane.generating) state = 'generating';
  else if (lane.error && !ready) state = 'error';
  let note = '';
  if (state === 'generating') note = 'generating…';
  else if (state === 'error') note = `error: ${lane.error}`;
  else if (ready) note = trying ? 'the new sound plays' : 'the old sound plays';
  return {
    state,
    name: ready ? String(lane.name || '') : '',
    canSwitch: ready && !lane.generating,
    trying,
    note,
  };
}

/** The chains of a bank (ai.bank_chains, "a-b c-e") as the page shows them: "A–B, C–E". */
export function chainsText(chains) {
  const runs = String(chains || '').trim().split(/\s+/).filter(Boolean);
  return runs.map((run) => run.toUpperCase().replace('-', '–')).join(', ');
}

/**
 * What the patterns column shows, from ai.bank ('none' | 'generating' |
 * 'ready'), ai.preview and ai.bank_chains: whether old / new can switch,
 * whether the new patterns play, and a note (which names the chains).
 */
export function patternView(bank, preview, chains = '') {
  const trying = preview === 'try';
  const chained = chainsText(chains);
  let note = 'make 12 patterns (A–L) for the sounds on the face';
  if (bank === 'generating') note = 'generating patterns…';
  else if (trying) {
    note = `the new patterns play${chained ? ` (chains ${chained})` : ''}: pick any of A–L on the face`;
  } else if (bank === 'ready') note = 'the old patterns play';
  return { canSwitch: bank === 'ready', trying, generating: bank === 'generating', note };
}

/** Whether anything new plays on the face (keep and revert have work). */
export const hasTrial = (tried, preview) => (tried || []).length > 0 || preview === 'try';

/** The last part of a path ('' for none). */
export function fileName(path) {
  if (!path) return '';
  return String(path).split(/[\\/]/).filter(Boolean).pop() || '';
}

/** What is wrong with the models ('' when nothing is), from ai.models. */
export function modelProblem(models) {
  const label = { patch: 'sound model', pattern: 'pattern model' };
  for (const kind of KINDS) {
    const m = (models || {})[kind];
    if (!m) continue;
    if (m.status === 'missing') return `no ${label[kind]}: set one in models…`;
    if (m.status === 'error') return `${label[kind]}: ${m.error || fileName(m.path)}`;
  }
  return '';
}

/** What the header says for ai.state ('' when idle). */
export function stateText(state) {
  return {
    generating: 'generating…', loading: 'loading a model…', installing: 'installing the ML extras…',
    unavailable: 'ML extras missing',
  }[state] || '';
}

/** The header note: what is new on the face ('' when nothing). */
export function trialText(tried, preview) {
  const parts = [];
  if ((tried || []).length) parts.push(`new sounds on ch ${tried.join(', ')}`);
  if (preview === 'try') parts.push('new patterns');
  return parts.join(' + ');
}

/**
 * The keep-or-revert question when leaving with new sounds or patterns:
 * tried is ai.tried (channels 1..8), names maps a channel to its new sound's
 * name, patterns is whether the new patterns play.
 */
export function leaveQuestion(tried, names = {}, patterns = false) {
  const list = (tried || []).map((ch) => (names[ch] ? `CH${ch} ${names[ch]}` : `CH${ch}`));
  if (patterns) list.push('the 12 new patterns');
  const which = list.length > 1 ? `${list.slice(0, -1).join(', ')} and ${list[list.length - 1]}` : list[0];
  const plural = list.length > 1 || (patterns && !list[0].startsWith('CH'));
  return {
    title: 'Keep what is new?',
    text: `${which[0].toUpperCase()}${which.slice(1)} ${plural ? 'play' : 'plays'} on the face. `
      + 'Keep (one undo step), or revert to what was there.',
  };
}

/** The display's line after a keep (ai.keep / ai.replace_patterns result). */
export function keptText(result) {
  const kept = (result && result.kept) || [];
  const parts = [];
  if (kept.length === 1) parts.push(`ch ${kept[0]}`);
  else if (kept.length) parts.push(`${kept.length} sounds`);
  if (result && result.patterns) parts.push(`${result.patterns} patterns`);
  return parts.length ? `kept ${parts.join(' + ')}` : 'nothing new';
}
