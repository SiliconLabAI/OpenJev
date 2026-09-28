/**
 * Logprob readout: score all options of a question in ONE chat completion
 * by asking the model to answer with a single option letter (A, B, C, ...)
 * and reading the top-logprobs of that first output token.
 *
 * Replaces N parallel micro-calls with one call per question for providers
 * that support `logprobs` (OpenAI, vLLM, most OpenAI-compatible servers).
 * Options not present in the returned top-k window get a floor logit
 * (min seen logit minus `floorGap`), so distributions stay well-formed.
 */

export const OPTION_LETTERS: string[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
export const DEFAULT_LOGPROB_TOP_K = 20;

export interface OptionSpec {
  key: string;
  description?: string | null;
}

export interface TopLogprob {
  token: string;
  logprob: number;
}

export interface LetterPrompt {
  system: string;
  user: string;
}

export function buildOptionLetterPrompt(
  state: string,
  questionText: string,
  options: OptionSpec[]
): LetterPrompt {
  const lines = options.map((o, i) => {
    const letter = OPTION_LETTERS[i];
    const desc =
      o.description && o.description.trim() && o.description.trim() !== o.key
        ? ` — ${o.description.trim()}`
        : "";
    return `${letter}. ${o.key}${desc}`;
  });
  const last = OPTION_LETTERS[options.length - 1] ?? OPTION_LETTERS[0];
  const system =
    "You are a calibrated decision model. Read the STATE and the QUESTION, then pick exactly one option letter. " +
    "Reply with a single letter and nothing else.";
  const user =
    `STATE:\n${state}\n\n` +
    `QUESTION: ${questionText}\n\n` +
    `OPTIONS:\n${lines.join("\n")}\n\n` +
    `Answer with exactly one letter (A–${last}) and nothing else.`;
  return { system, user };
}

function firstLetter(token: string): string | null {
  const m = token.trim().match(/^[A-Za-z]/);
  return m ? m[0].toUpperCase() : null;
}

/**
 * Map the first-token top-logprobs onto option-letter logits.
 * Token normalization tolerates leading whitespace and trailing punctuation
 * (" A", "A)", "A."). The first (highest-probability) occurrence of each
 * letter wins. Letters missing from the window get `minSeen - floorGap`.
 *
 * `optionKeys` (optional) validates that the letters actually correspond to
 * the requested option count on top of the raw letter match.
 */
export function parseLetterLogprobs(
  topLogprobs: TopLogprob[],
  optionCount: number,
  optionKeys?: string[],
  floorGap = 5
): { logits: number[]; matched: number } {
  const byLetter = new Map<string, number>();
  for (const tl of topLogprobs) {
    if (!tl || typeof tl.logprob !== "number") continue;
    const letter = firstLetter(String(tl.token));
    if (!letter) continue;
    if (!byLetter.has(letter)) byLetter.set(letter, tl.logprob);
  }
  const letters = OPTION_LETTERS.slice(0, Math.max(0, optionCount));
  const found = letters
    .map((l) => byLetter.get(l))
    .filter((v): v is number => typeof v === "number");
  const floor = (found.length ? Math.min(...found) : -20) - floorGap;
  const logits = letters.map((l) => {
    const v = byLetter.get(l);
    return typeof v === "number" ? v : floor;
  });
  // If the caller supplied the real option keys, require at least one letter
  // to be present (same check as `found.length`); keys themselves are not
  // letters, so this only guards the count sanity.
  const matched = optionKeys && optionKeys.length !== optionCount ? 0 : found.length;
  return { logits, matched };
}

/**
 * Extract Yes/No logits from a first-token top-logprobs window for noul
 * questions. Tolerates "Yes"/"No", " yes"/" no", "Yes."/... forms. Either
 * side may be missing from the window; callers decide how to floor it.
 */
export function parseYesNoLogprobs(
  topLogprobs: TopLogprob[]
): { yesLogit: number | null; noLogit: number | null } {
  let yesLogit: number | null = null;
  let noLogit: number | null = null;
  for (const tl of topLogprobs) {
    if (!tl || typeof tl.logprob !== "number") continue;
    const norm = String(tl.token).trim().toLowerCase().replace(/[^a-z]/g, "");
    if (yesLogit === null && (norm === "yes" || norm === "y")) yesLogit = tl.logprob;
    else if (noLogit === null && (norm === "no" || norm === "n")) noLogit = tl.logprob;
  }
  return { yesLogit, noLogit };
}
