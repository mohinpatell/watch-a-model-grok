export type AblationRecord = {
  step: number;
  train_loss: number;
  test_loss: number;
  train_acc: number;
  test_acc: number;
};

export type AblationConfig = {
  weight_decay: number;
  init_mode: string;
  use_layer_norm: boolean;
  seed: number;
  num_steps: number;
};

export type AblationRun = {
  name: string;
  config: AblationConfig;
  records: AblationRecord[];
  final: AblationRecord;
};

export type AblationBundle = {
  order: string[];
  runs: Record<string, AblationRun>;
};

const PREFIX = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export async function loadAblations(
  basePath = `${PREFIX}/ablations`,
): Promise<AblationBundle | null> {
  const indexRes = await fetch(`${basePath}/index.json`);
  if (!indexRes.ok) return null;
  const index: { names: string[] } = await indexRes.json();

  const runs: Record<string, AblationRun> = {};
  await Promise.all(
    index.names.map(async (name) => {
      const res = await fetch(`${basePath}/${name}.json`);
      if (!res.ok) return;
      const run: AblationRun = await res.json();
      runs[name] = run;
    }),
  );

  return { order: index.names, runs };
}

export function grokked(run: AblationRun): boolean {
  return run.final.test_acc > 0.9;
}

// First step at which test_acc crosses THRESHOLD, or null if it never does.
export function grokStep(
  run: AblationRun,
  threshold = 0.9,
): number | null {
  for (const r of run.records) {
    if (r.test_acc >= threshold) return r.step;
  }
  return null;
}
