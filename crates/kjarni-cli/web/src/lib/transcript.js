// A conversation as the page shows it: a list of blocks built from the saved
// messages, then extended by the events of the turn in progress.
//
// Blocks:
//   { type: 'user', text }
//   { type: 'assistant', raw, reasoning, text, thinking }
//   { type: 'tool', call, name, summary, access, arguments, status, output, approval }
//       status: running | waiting | done | failed | denied
//   { type: 'notice', text } · { type: 'error', text } · { type: 'stopped' }

const THINK_OPEN = '<think>';
const THINK_CLOSE = '</think>';
const TOOL_CALL = '<tool_call>';

/** `text` without a trailing partial `tag`, so a tag arriving a few
 * characters at a time never flashes on screen. */
function withoutPartial(text, tag) {
  for (let n = tag.length - 1; n > 0; n--) {
    if (text.endsWith(tag.slice(0, n))) return text.slice(0, -n);
  }
  return text;
}

/** A reply's reasoning and answer. A reply still inside its `<think>` block
 * is `thinking`. Templates that open the block in the prompt leave only the
 * close in saved replies. */
export function splitReasoning(raw) {
  const trimmed = raw.trimStart();
  const opened = trimmed.startsWith(THINK_OPEN);
  const body = opened ? trimmed.slice(THINK_OPEN.length) : raw;
  const close = body.indexOf(THINK_CLOSE);
  if (close >= 0) {
    return {
      reasoning: body.slice(0, close).trim(),
      text: body.slice(close + THINK_CLOSE.length).trimStart(),
      thinking: false,
    };
  }
  if (opened) {
    return { reasoning: withoutPartial(body, THINK_CLOSE).trim(), text: '', thinking: true };
  }
  return { reasoning: '', text: raw, thinking: false };
}

/** Reply text up to the first tool call: the calls show as steps instead. */
export function visibleText(text) {
  const at = text.indexOf(TOOL_CALL);
  const before = at >= 0 ? text.slice(0, at) : withoutPartial(text, TOOL_CALL);
  return before.trimEnd();
}

/** The argument that says what a call is about: the file, the search, the
 * command or the address, else the first one. */
export function mainArgument(args) {
  if (!args) return undefined;
  for (const key of ['path', 'query', 'command', 'url', 'text']) {
    if (key in args) return args[key];
  }
  return Object.values(args)[0];
}

/** `value`'s first line, shortened to `max` characters. */
export function firstLine(value, max = 80) {
  const text = typeof value === 'string' ? value : JSON.stringify(value);
  const lines = text.split('\n');
  let short = [...lines[0]].slice(0, max).join('');
  if (short.length < lines[0].length || lines.length > 1) short += '…';
  return short;
}

/** What a call is about, in a few words; paths inside `workdir` are shown
 * relative to it. */
export function callDetail(args, workdir) {
  let value = mainArgument(args);
  if (value === undefined) return '';
  if (typeof value === 'string' && workdir) {
    const base = workdir.replace(/[\\/]$/, '');
    if (value.startsWith(base + '/') || value.startsWith(base + '\\')) value = value.slice(base.length + 1);
  }
  return firstLine(value);
}

/** A call in a few words. */
export function summarize(name, args) {
  const detail = callDetail(args);
  return detail ? `${name} ${detail}` : name;
}

const DECLINED = 'The user declined this call';

/** Blocks for the saved messages of a conversation. */
export function fromHistory(messages) {
  const blocks = [];
  const unanswered = [];
  for (const m of messages ?? []) {
    if (m.role === 'user') {
      blocks.push({ type: 'user', text: m.text });
    } else if (m.role === 'assistant') {
      const text = visibleText(m.text ?? '');
      if (text || m.reasoning) {
        blocks.push({ type: 'assistant', raw: '', reasoning: m.reasoning ?? '', text, thinking: false });
      }
      for (const c of m.tool_calls ?? []) {
        const block = {
          type: 'tool',
          call: null,
          name: c.name,
          summary: summarize(c.name, c.arguments),
          access: null,
          arguments: c.arguments,
          status: 'done',
          output: null,
          approval: null,
        };
        blocks.push(block);
        unanswered.push(block);
      }
    } else if (m.role === 'tool') {
      const block = unanswered.shift();
      if (block) {
        block.output = m.text;
        if (m.text?.startsWith(DECLINED)) block.status = 'denied';
      }
    }
  }
  return blocks;
}

/** A conversation's blocks, ready for events. */
export function newTurnState(blocks) {
  return { blocks, open: null, calls: {}, ended: null, lastSeq: 0 };
}

/** Appends `block` and returns it as the list holds it. In a Svelte `$state`
 * that is a reactive proxy, which later changes must go through. */
function append(state, block) {
  state.blocks.push(block);
  return state.blocks[state.blocks.length - 1];
}

/** Adds one event of the turn in progress to `state`. */
export function applyEvent(state, e) {
  if (e.seq) state.lastSeq = Math.max(state.lastSeq, e.seq);
  switch (e.type) {
    case 'user':
      state.blocks.push({ type: 'user', text: e.text });
      state.open = null;
      state.ended = null;
      break;
    case 'text': {
      if (!state.open) {
        state.open = append(state, { type: 'assistant', raw: '', reasoning: '', text: '', thinking: false });
      }
      const open = state.open;
      open.raw += e.delta;
      const parts = splitReasoning(open.raw);
      open.reasoning = parts.reasoning;
      open.thinking = parts.thinking;
      open.text = visibleText(parts.text);
      break;
    }
    case 'tool_call': {
      state.open = null;
      state.calls[e.call] = append(state, {
        type: 'tool',
        call: e.call,
        name: e.name,
        summary: e.summary,
        access: e.access,
        arguments: e.arguments,
        status: 'running',
        output: null,
        approval: null,
      });
      break;
    }
    case 'approval': {
      const block = state.calls[e.call];
      if (block) {
        block.status = 'waiting';
        block.approval = e.approval;
      }
      break;
    }
    case 'approved': {
      const block = Object.values(state.calls).find((b) => b.approval === e.approval);
      if (block) {
        block.status = e.decision === 'deny' ? 'denied' : 'running';
        block.approval = null;
      }
      break;
    }
    case 'tool_result': {
      const block = state.calls[e.call];
      if (block) {
        block.status = e.denied ? 'denied' : e.ok ? 'done' : 'failed';
        block.output = e.output;
      }
      break;
    }
    case 'notice':
      state.open = null;
      state.blocks.push({ type: 'notice', text: e.message });
      break;
    case 'done':
      state.open = null;
      state.ended = 'done';
      break;
    case 'error':
      state.open = null;
      state.blocks.push({ type: 'error', text: e.message });
      state.ended = 'error';
      break;
    case 'stopped':
      state.open = null;
      for (const b of Object.values(state.calls)) {
        if (b.status === 'running' || b.status === 'waiting') {
          b.status = 'failed';
          b.approval = null;
        }
      }
      state.blocks.push({ type: 'stopped' });
      state.ended = 'stopped';
      break;
  }
  return state;
}

/** Whether an assistant block has anything to show yet. */
export function hasContent(block) {
  return Boolean(block.text || block.reasoning || block.thinking);
}
