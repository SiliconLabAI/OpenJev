/**
 * Bounded-concurrency task runner.
 *
 * Keeps at most `limit` worker promises in flight while preserving the input
 * order of results. Workers that throw reject the whole run — callers that
 * need per-item isolation should catch inside `worker`.
 */
export async function limitConcurrency<TIn, TOut>(
  items: readonly TIn[],
  limit: number,
  worker: (item: TIn, index: number) => Promise<TOut>
): Promise<TOut[]> {
  const results = new Array<TOut>(items.length);
  if (items.length === 0) return results;
  const n = Math.max(1, Math.min(Math.floor(limit), items.length));
  let next = 0;
  const runners = Array.from({ length: n }, async () => {
    for (;;) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await worker(items[i], i);
    }
  });
  await Promise.all(runners);
  return results;
}
