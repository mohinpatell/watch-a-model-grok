"use client";

import { useEffect, useRef } from "react";
import * as Plot from "@observablehq/plot";
import { useStore } from "@/lib/store";
import { useResponsiveWidth } from "@/lib/useResponsiveWidth";

type Mode = "loss" | "acc";

type Props = {
  mode?: Mode;
  width?: number;
  height?: number;
  compact?: boolean;
};

type PlotWithScales = SVGElement & {
  scale?: (name: string) => { apply: (x: number) => number } | undefined;
};

type PlotGeometry = {
  applyX: (x: number) => number;
  top: number;
  height: number;
};

export default function LossChart({
  mode = "loss",
  width: maxWidth = 660,
  height = 240,
  compact = false,
}: Props) {
  const [wrapRef, width] = useResponsiveWidth(maxWidth);
  const plotHostRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const geomRef = useRef<PlotGeometry | null>(null);

  const dataset = useStore((s) => s.dataset);
  const frame = useStore((s) => s.frame);

  // Rebuild the chart only when inputs other than frame change.
  useEffect(() => {
    if (!dataset || !plotHostRef.current) return;
    const { steps, train_loss, test_loss, train_acc, test_acc } = dataset.meta;

    const train = mode === "loss" ? train_loss : train_acc;
    const test = mode === "loss" ? test_loss : test_acc;
    const label = mode === "loss" ? "loss" : "accuracy";

    const rows = steps.flatMap((s, i) => [
      { step: Math.max(s, 1), value: train[i], split: "train" },
      { step: Math.max(s, 1), value: test[i], split: "test" },
    ]);

    const marginTop = 16;
    const marginBottom = compact ? 28 : 36;

    const plot = Plot.plot({
      marginLeft: compact ? 40 : 52,
      marginBottom,
      marginTop,
      marginRight: compact ? 12 : 18,
      width,
      height,
      style: {
        background: "transparent",
        color: "#444",
        fontSize: compact ? "11px" : "12px",
        fontFamily:
          'ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, monospace',
      },
      x: {
        type: "log",
        label: "training step",
        labelAnchor: "center",
        grid: true,
      },
      y: {
        label,
        labelAnchor: "top",
        grid: true,
        domain: mode === "acc" ? [0, 1] : undefined,
        type: mode === "loss" ? "log" : "linear",
      },
      color: {
        domain: ["train", "test"],
        range: ["#cf222e", "#1f6feb"],
        legend: !compact,
      },
      marks: [
        Plot.ruleY(mode === "acc" ? [0, 1] : [], { stroke: "#ddd" }),
        Plot.line(rows, {
          x: "step",
          y: "value",
          stroke: "split",
          strokeWidth: 1.8,
          curve: "monotone-x",
        }),
      ],
    });

    plotHostRef.current.replaceChildren(plot);

    const p = plot as unknown as PlotWithScales;
    const xScale = p.scale?.("x");
    const wrap = plotHostRef.current.parentElement;
    const svg = plotHostRef.current.querySelector("svg");
    if (xScale && wrap && svg) {
      const wrapBox = wrap.getBoundingClientRect();
      const svgBox = svg.getBoundingClientRect();
      geomRef.current = {
        applyX: (x: number) => xScale.apply(x),
        top: svgBox.top - wrapBox.top + marginTop,
        height: svgBox.height - marginTop - marginBottom,
      };
    } else {
      geomRef.current = null;
    }

    return () => plot.remove();
  }, [dataset, mode, width, height, compact]);

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
