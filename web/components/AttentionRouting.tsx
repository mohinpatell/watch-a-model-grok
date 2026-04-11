"use client";

import { useEffect } from "react";
import * as Plot from "@observablehq/plot";
import { useStore } from "@/lib/store";
import { attnFrame } from "@/lib/data";
import { useResponsiveWidth } from "@/lib/useResponsiveWidth";

const POS_LABEL = ["a", "b", "="];

export default function AttentionRouting() {
  const [ref, width] = useResponsiveWidth(660);
  const dataset = useStore((s) => s.dataset);
  const frame = useStore((s) => s.frame);

  useEffect(() => {
    if (!dataset || !ref.current) return;
    const nH = dataset.meta.n_heads;
    const [, nProbe, , , nSeq] = dataset.meta.shapes.attn;
    const att = attnFrame(dataset, frame);
    // att layout: (probe, head, query, key), strides = (nH*nSeq*nSeq, nSeq*nSeq, nSeq, 1).
    // We want attention at query=2 (the = token), averaged across probes.
    const q = nSeq - 1;
    const rows: { head: string; pos: string; weight: number }[] = [];
    for (let h = 0; h < nH; h++) {
      for (let k = 0; k < nSeq; k++) {
        let sum = 0;
        for (let p = 0; p < nProbe; p++) {
          const idx = p * nH * nSeq * nSeq + h * nSeq * nSeq + q * nSeq + k;
          sum += att[idx];
        }
        rows.push({
          head: `head ${h}`,
          pos: POS_LABEL[k] ?? String(k),
          weight: sum / nProbe,
        });
      }
    }

    const plot = Plot.plot({
      marginLeft: 72,
      marginRight: 18,
      marginBottom: 30,
      marginTop: 12,
      width,
      height: 140,
      style: {
        background: "transparent",
        color: "#444",
        fontSize: "12px",
        fontFamily:
          'ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, monospace',
      },
      x: {
        label: `attention weight (averaged over ${nProbe} probe pairs)`,
        domain: [0, 1],
        grid: true,
      },
      y: {
        label: null,
        domain: Array.from({ length: nH }, (_, h) => `head ${h}`).reverse(),
      },
      color: {
        domain: POS_LABEL,
        range: ["#cf222e", "#1f6feb", "#9a9a9a"],
        legend: true,
        label: "attends to",
      },
      marks: [
        Plot.barX(rows, {
          x: "weight",
          y: "head",
          fill: "pos",
          order: POS_LABEL,
        }),
      ],
    });

    ref.current.replaceChildren(plot);
    return () => plot.remove();
  }, [dataset, frame, width, ref]);

  return <div ref={ref} className="w-full" />;
}
