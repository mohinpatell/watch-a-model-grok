"use client";

import { useEffect, useRef } from "react";
import * as Plot from "@observablehq/plot";
import { useStore } from "@/lib/store";
import { useResponsiveWidth } from "@/lib/useResponsiveWidth";

type PlotWithScales = SVGElement & {
  scale?: (name: string) => { apply: (x: number) => number } | undefined;
};

type PlotGeometry = {
  applyX: (x: number) => number;
  applyY: (y: number) => number;
  top: number;
  height: number;
};

export default function ProgressMeasure() {
  const [wrapRef, width] = useResponsiveWidth(660);
  const plotHostRef = useRef<HTMLDivElement | null>(null);
  const ruleRef = useRef<HTMLDivElement | null>(null);
  const dotRef = useRef<HTMLDivElement | null>(null);
  const geomRef = useRef<PlotGeometry | null>(null);

  const dataset = useStore((s) => s.dataset);
  const frame = useStore((s) => s.frame);

  // Build the curve once per dataset/width change.
  useEffect(() => {
    if (!dataset || !plotHostRef.current) return;
    const { steps, share_at_kstar, fourier_k, p } = dataset.meta;
    if (!share_at_kstar) return;
    const baseline = 1 / Math.floor(p / 2);

    const rows = steps.map((s, i) => ({
      step: Math.max(s, 1),
      share: share_at_kstar[i] ?? 0,
    }));

    const marginTop = 16;
    const marginBottom = 36;

    const plot = Plot.plot({
      ariaLabel: `embedding power share at k=${fourier_k} across training`,
      ariaDescription: `Line chart of the share of token-embedding L2 power concentrated at frequency k=${fourier_k}, per checkpoint. The dashed horizontal line marks the uniform baseline 1/floor(p/2).`,
      marginLeft: 56,
      marginRight: 56,
      marginBottom,
      marginTop,
      width,
      height: 260,
      style: {
        background: "transparent",
        color: "#444",
        fontSize: "12px",
        fontFamily:
          'ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, monospace',
      },
      x: {
        type: "log",
        label: "training step",
        grid: true,
      },
      y: {
        label: `embedding power at k=${fourier_k}  /  total`,
        domain: [0, Math.max(0.3, ...rows.map((r) => r.share)) * 1.05],
        grid: true,
      },
      marks: [
        Plot.ruleY([baseline], {
          stroke: "#bbb",
          strokeDasharray: "3,3",
        }),
        Plot.line(rows, {
          x: "step",
          y: "share",
          stroke: "#9a3412",
          strokeWidth: 1.8,
          curve: "monotone-x",
        }),
      ],
    });

    plotHostRef.current.replaceChildren(plot);

    const p2 = plot as unknown as PlotWithScales;
    const xScale = p2.scale?.("x");
    const yScale = p2.scale?.("y");
    const wrap = plotHostRef.current.parentElement;
    const svg = plotHostRef.current.querySelector("svg");
    if (xScale && yScale && wrap && svg) {
      const wrapBox = wrap.getBoundingClientRect();
      const svgBox = svg.getBoundingClientRect();
      geomRef.current = {
        applyX: (x: number) => xScale.apply(x),
        applyY: (y: number) => yScale.apply(y),
        top: svgBox.top - wrapBox.top + marginTop,
        height: svgBox.height - marginTop - marginBottom,
      };
    } else {
      geomRef.current = null;
    }

    return () => plot.remove();
  }, [dataset, width]);

  // Cheap per-scrub update: move rule + dot without rebuilding the plot.
  useEffect(() => {
    const rule = ruleRef.current;
    const dot = dotRef.current;
    const geom = geomRef.current;
    if (!rule || !dot) return;
    if (!dataset || !geom) {
      rule.style.display = "none";
      dot.style.display = "none";
      return;
    }
    const { steps, share_at_kstar } = dataset.meta;
    const step = Math.max(steps[frame], 1);
    const share = share_at_kstar?.[frame] ?? 0;
    const x = geom.applyX(step);
    const y = geom.applyY(share);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      rule.style.display = "none";
      dot.style.display = "none";
      return;
    }
    rule.style.display = "block";
    rule.style.left = `${x}px`;
    rule.style.top = `${geom.top}px`;
    rule.style.height = `${geom.height}px`;
    dot.style.display = "block";
    dot.style.left = `${x - 4}px`;
    dot.style.top = `${y - 4}px`;
  }, [dataset, frame, width]);

  return (
    <div ref={wrapRef} className="w-full relative">
      <div ref={plotHostRef} />
      <div
        ref={ruleRef}
        aria-hidden
        className="pointer-events-none absolute w-px"
        style={{
          borderLeft: "1px dashed #0969da",
          display: "none",
        }}
      />
      <div
        ref={dotRef}
        aria-hidden
        className="pointer-events-none absolute"
        style={{
          width: 8,
          height: 8,
          borderRadius: 9999,
          background: "#9a3412",
          boxShadow: "0 0 0 1.5px white",
          display: "none",
        }}
      />
    </div>
  );
}
