"use client";

import { useEffect, useState } from "react";
import {
  loadAblations,
  grokked,
  grokStep,
  type AblationBundle,
} from "@/lib/ablations";

type Row = {
  name: string;
  label: string;
  change: string;
  trainAcc: number;
  testAcc: number;
  grokked: boolean;
  grokStep: number | null;
};

const LABELS: Record<string, { label: string; change: string }> = {
  wd_low: { label: "wd_low", change: "weight_decay = 0.01" },
  wd_zero: { label: "wd_zero", change: "weight_decay = 0" },
  init_pytorch: { label: "init_default", change: "embeddings ~ N(0, 1)" },
  layer_norm: { label: "layer_norm", change: "pre-norm LayerNorm on" },
  seed_0: { label: "seed_0", change: "seed = 0 (default config)" },
  seed_7: { label: "seed_7", change: "seed = 7 (default config)" },
};

export default function AblationTable() {
  const [bundle, setBundle] = useState<AblationBundle | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    loadAblations()
      .then((b) => {
        if (b) setBundle(b);
        else setMissing(true);
      })
      .catch(() => setMissing(true));
  }, []);

  if (missing) {
    return (
      <div className="border border-[var(--rule)] bg-white rounded px-4 py-3 font-mono text-xs text-[var(--muted)]">
        ablation data not yet exported — run{" "}
        <code>uv run python -m training.ablate</code> then{" "}
        <code>uv run python -m training.export_ablations</code>.
      </div>
    );
  }

  if (!bundle) {
    return (
      <div className="flex items-center gap-2 font-mono text-sm text-[var(--muted)]">
        <span className="inline-block h-1.5 w-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
        loading ablations…
      </div>
    );
  }

  const rows: Row[] = bundle.order.map((name) => {
    const run = bundle.runs[name];
    const meta = LABELS[name] ?? { label: name, change: name };
    return {
      name,
      label: meta.label,
      change: meta.change,
      trainAcc: run.final.train_acc,
      testAcc: run.final.test_acc,
      grokked: grokked(run),
      grokStep: grokStep(run),
    };
  });

  return (
    <figure className="my-6">
      <table className="w-full border-collapse font-mono text-[13px]">
        <thead>
          <tr className="text-left text-[var(--muted)] border-b border-[var(--rule)]">
            <th className="py-2 pr-4 font-medium">run</th>
            <th className="py-2 pr-4 font-medium">change vs. baseline</th>
            <th className="py-2 pr-3 font-medium text-right">test acc</th>
            <th className="py-2 pr-3 font-medium text-right">grok step</th>
            <th className="py-2 font-medium text-right">grokked?</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-b border-[var(--rule)]">
            <td className="py-2 pr-4">baseline</td>
            <td className="py-2 pr-4 text-[var(--muted)]">
              wd=1.0, 1/√d init, no LN, seed=42
            </td>
            <td className="py-2 pr-3 text-right tabular-nums">1.00</td>
            <td className="py-2 pr-3 text-right tabular-nums">~8,000</td>
            <td className="py-2 text-right" style={{ color: "var(--accent)" }}>
              yes
            </td>
          </tr>
          {rows.map((r) => (
            <tr key={r.name} className="border-b border-[var(--rule)]">
              <td className="py-2 pr-4">{r.label}</td>
              <td className="py-2 pr-4 text-[var(--muted)]">{r.change}</td>
              <td className="py-2 pr-3 text-right tabular-nums">
                {r.testAcc.toFixed(2)}
              </td>
              <td className="py-2 pr-3 text-right tabular-nums text-[var(--muted)]">
                {r.grokStep !== null
                  ? r.grokStep.toLocaleString()
                  : "—"}
              </td>
              <td
                className="py-2 text-right"
                style={{ color: r.grokked ? "var(--accent)" : "var(--train)" }}
              >
                {r.grokked ? "yes" : "no"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <figcaption className="mt-2">
        Ablations trained for 15,000 steps each (baseline runs to 40,000 but
        groks near step 8,000). <em>Grok step</em> is the first checkpoint at
        which test accuracy crosses 0.9. Each row changes exactly one knob
        relative to the baseline; train accuracy reaches 1.00 in every run.
      </figcaption>
    </figure>
  );
}
