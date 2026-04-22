"use client";

import { useEffect, useRef } from "react";
import * as Plot from "@observablehq/plot";
import { useStore } from "@/lib/store";
import { useResponsiveWidth } from "@/lib/useResponsiveWidth";

const PLOT_HEIGHT = 280;

type PlotWithScales = SVGElement & {
  scale?: (name: string) => { apply: (x: number) => number } | undefined;
};

type PlotGeometry = {
  applyX: (x: number) => number;
  top: number;
  height: number;
};

export default function FftSpectrogram() {
  const [wrapRef, width] = useResponsiveWidth(660);
  const plotHostRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const geomRef = useRef<PlotGeometry | null>(null);

  const dataset = useStore((s) => s.dataset);
  const frame = useStore((s) => s.frame);

  // Rebuild the heavy heatmap only when dataset/width change, not on scrub.
  useEffect(() => {
    if (!dataset || !plotHostRef.current) return;
    const { steps, fourier_k, shapes } = dataset.meta;
    const [, nK] = shapes.fft_share;

    const logMids: number[] = steps.map((s) => Math.log(Math.max(s, 1)));
    const bounds: number[] = new Array(steps.length + 1);
    bounds[0] = Math.max(logMids[0] - 0.3, 0);
    for (let i = 1; i < steps.length; i++) {
      bounds[i] = 0.5 * (logMids[i - 1] + logMids[i]);
    }
    bounds[steps.length] = logMids[steps.length - 1] + 0.3;

    const rows: { x1: number; x2: number; k: number; share: number }[] = [];
    for (let f = 0; f < steps.length; f++) {
      const x1 = Math.exp(bounds[f]);
      const x2 = Math.exp(bounds[f + 1]);
      for (let k = 0; k < nK; k++) {
        rows.push({
          x1,
          x2,
          k: k + 1,
          share: dataset.fftShare[f * nK + k],
        });
      }
    }

    const legendWidth = width < 560 ? 0 : 100;

    const plot = Plot.plot({
      ariaLabel: "embedding power share per Fourier frequency across training",
      ariaDescription: `Heatmap of embedding power share at each frequency k from 1 to ${nK}, across all checkpoints. Training step on the log x-axis, frequency on the y-axis. The dashed horizontal line marks k=${fourier_k}.`,
      marginLeft: 48,
      marginRight: legendWidth || 12,
      marginBottom: 36,
      marginTop: 16,
      width,
      height: PLOT_HEIGHT,
      style: {
        background: "transparent",
        color: "#444",
        fontSize: "12px",
        fontFamily:
          'ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, monospace',
      },
      x: { type: "log", label: "training step", grid: false },
      y: {
        label: "frequency k",
        domain: [0.5, nK + 0.5],
        ticks: [1, 10, 20, 30, 40, 46, 56],
      },
      color: {
        type: "sqrt",
        scheme: "blues",
        domain: [0.02, 0.12],
        clamp: true,
        legend: legendWidth > 0,
        label: "share",
      },
      marks: [
        Plot.rect(rows, {
          x1: "x1",
          x2: "x2",
          y1: (d: { k: number }) => d.k - 0.5,
          y2: (d: { k: number }) => d.k + 0.5,
          fill: "share",
          stroke: null,
        }),
        Plot.ruleY([fourier_k], {
          stroke: "#9a3412",
          strokeWidth: 1,
          strokeDasharray: "3,3",
          opacity: 0.7,
        }),
      ],
    });

    plotHostRef.current.replaceChildren(plot);

    const p = plot as unknown as PlotWithScales;
    const xScale = p.scale?.("x");

    // Compute plot-area offset relative to the wrapper so the overlay
    // line sits exactly over the plot's data rect (excluding legend).
    const wrap = plotHostRef.current.parentElement;
    const svg = plotHostRef.current.querySelector("svg");
    if (xScale && wrap && svg) {
      const wrapBox = wrap.getBoundingClientRect();
      const svgBox = svg.getBoundingClientRect();
      geomRef.current = {
        applyX: (x: number) => xScale.apply(x),
        top: svgBox.top - wrapBox.top + 16, // marginTop
        height: svgBox.height - 16 - 36, // minus marginTop/Bottom
      };
    } else {
      geomRef.current = null;
    }

    return () => plot.remove();
  }, [dataset, width]);

  // Cheap per-scrub update: only move the overlay line.
  useEffect(() => {
    const el = overlayRef.current;
    if (!el) return;
    const geom = geomRef.current;
    if (!dataset || !geom) {
      el.style.display = "none";
      return;
    }
    const step = Math.max(dataset.meta.steps[frame], 1);
    const x = geom.applyX(step);
    if (!Number.isFinite(x)) {
      el.style.display = "none";
      return;
    }
    el.style.display = "block";
    el.style.left = `${x}px`;
    el.style.top = `${geom.top}px`;
    el.style.height = `${geom.height}px`;
  }, [dataset, frame, width]);

  return (
    <div ref={wrapRef} className="w-full relative">
      <div ref={plotHostRef} />
      <div
        ref={overlayRef}
        aria-hidden
        className="pointer-events-none absolute w-px"
        style={{
          borderLeft: "1px dashed #0969da",
          display: "none",
        }}
      />
    </div>
  );
}
