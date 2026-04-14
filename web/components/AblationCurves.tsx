"use client";

import { useEffect, useState } from "react";
import * as Plot from "@observablehq/plot";
import { loadAblations, type AblationBundle } from "@/lib/ablations";
import { useResponsiveWidth } from "@/lib/useResponsiveWidth";

const LABEL: Record<string, string> = {
  wd_low: "wd=0.01",
  wd_zero: "wd=0",
  init_pytorch: "default init",
  layer_norm: "LayerNorm",
  seed_0: "seed=0 (baseline)",
  seed_7: "seed=7 (baseline)",
};

const COLOR: Record<string, string> = {
  wd_low: "#cf222e",
  wd_zero: "#9a3412",
  init_pytorch: "#9333ea",
  layer_norm: "#ea580c",
  seed_0: "#1f6feb",
  seed_7: "#0ea5e9",
};

export default function AblationCurves() {
  const [ref, width] = useResponsiveWidth(660);
  const [bundle, setBundle] = useState<AblationBundle | null>(null);

  useEffect(() => {
    loadAblations().then((b) => setBundle(b)).catch(() => setBundle(null));
  }, []);

  useEffect(() => {
    if (!bundle || !ref.current) return;
    const rows: { step: number; test_loss: number; run: string }[] = [];
    for (const name of bundle.order) {
      const run = bundle.runs[name];
      for (const r of run.records) {
        rows.push({
          step: Math.max(r.step, 1),
          test_loss: r.test_loss,
          run: LABEL[name] ?? name,
        });
      }
    }
    const domain = bundle.order.map((n) => LABEL[n] ?? n);
    const range = bundle.order.map((n) => COLOR[n] ?? "#555");

    const legendWidth = width < 560 ? 0 : 160;
    const plot = Plot.plot({
      marginLeft: 56,
      marginRight: legendWidth || 18,
      marginBottom: 36,
      marginTop: 16,
      width,
      height: 280,
      style: {
        background: "transparent",
        color: "#444",
        fontSize: "12px",
        fontFamily:
          'ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, monospace',
      },
      x: { type: "log", label: "training step", grid: true },
      y: {
        type: "log",
        label: "test loss",
        grid: true,
      },
      color: { domain, range, legend: legendWidth > 0 },
      marks: [
        Plot.line(rows, {
          x: "step",
          y: "test_loss",
          stroke: "run",
          strokeWidth: 1.6,
          curve: "monotone-x",
        }),
      ],
    });
    ref.current.replaceChildren(plot);
    return () => plot.remove();
  }, [bundle, width, ref]);

  return <div ref={ref} className="w-full my-6" />;
}
