"use client";

import { useEffect } from "react";
import * as Plot from "@observablehq/plot";
import { useStore } from "@/lib/store";
import { fftShareFrame } from "@/lib/data";
import { useResponsiveWidth } from "@/lib/useResponsiveWidth";

export default function FftSpectrum() {
  const [ref, width] = useResponsiveWidth(660);
  const dataset = useStore((s) => s.dataset);
  const frame = useStore((s) => s.frame);

  useEffect(() => {
    if (!dataset || !ref.current) return;
    const kStar = dataset.meta.fourier_k;
    const share = fftShareFrame(dataset, frame);

    const rows = Array.from(share, (v, i) => ({
      k: i + 1,
      share: v,
      isKStar: i + 1 === kStar,
    }));

    const plot = Plot.plot({
      ariaLabel: "embedding power share per Fourier frequency at current checkpoint",
      ariaDescription: `Bar chart of the token embedding's L2 power share at each frequency k from 1 to ${share.length}. The bar at k=${kStar} is highlighted.`,
      marginLeft: 56,
      marginRight: 18,
      marginBottom: 36,
      marginTop: 16,
      width,
      height: 220,
      style: {
        background: "transparent",
        color: "#444",
        fontSize: "12px",
        fontFamily:
          'ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, monospace',
      },
      x: {
        type: "linear",
        label: "frequency k",
        domain: [0.5, share.length + 0.5],
        ticks: [1, 10, 20, 30, 40, 46, 56],
        grid: false,
      },
      y: {
        label: "embedding power share",
        domain: [0, 0.3],
        grid: true,
      },
      marks: [
        Plot.ruleY([0], { stroke: "#ccc" }),
        Plot.ruleX([kStar], {
          stroke: "#9a3412",
          strokeWidth: 1,
          strokeDasharray: "2,3",
          opacity: 0.5,
        }),
        Plot.rectY(rows, {
          x1: (d: { k: number }) => d.k - 0.45,
          x2: (d: { k: number }) => d.k + 0.45,
          y: "share",
          fill: (d: { isKStar: boolean }) =>
            d.isKStar ? "#9a3412" : "#94a3b8",
        }),
      ],
    });

    ref.current.replaceChildren(plot);
    return () => plot.remove();
  }, [dataset, frame, width, ref]);

  return <div ref={ref} className="w-full" />;
}
