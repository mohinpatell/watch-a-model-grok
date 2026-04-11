"use client";

import { useEffect, useRef } from "react";
import { interpolateSinebow } from "d3-scale-chromatic";
import { useStore } from "@/lib/store";
import { embedsFrame } from "@/lib/data";

const SIZE = 320;
const PAD = 22;

export default function EmbeddingScatter() {
  const ref = useRef<HTMLCanvasElement>(null);
  const dataset = useStore((s) => s.dataset);
  const frame = useStore((s) => s.frame);

  // One-time canvas setup: DPR scaling is expensive (context reset + layout)
  // and the size never changes after mount.
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;
    canvas.style.width = `${SIZE}px`;
    canvas.style.height = `${SIZE}px`;
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }, []);

  useEffect(() => {
    if (!dataset || !ref.current) return;
    const ctx = ref.current.getContext("2d");
    if (!ctx) return;

    const pts = embedsFrame(dataset, frame);
    const p = dataset.meta.p;

    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (let i = 0; i < p; i++) {
      const x = pts[i * 2];
      const y = pts[i * 2 + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const span = Math.max(maxX - minX, maxY - minY) || 1;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const scale = (SIZE - 2 * PAD) / span;

    ctx.clearRect(0, 0, SIZE, SIZE);

    ctx.strokeStyle = "#e8e8e8";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD, SIZE / 2);
    ctx.lineTo(SIZE - PAD, SIZE / 2);
    ctx.moveTo(SIZE / 2, PAD);
    ctx.lineTo(SIZE / 2, SIZE - PAD);
    ctx.stroke();

    for (let i = 0; i < p; i++) {
      const x = (pts[i * 2] - cx) * scale + SIZE / 2;
      const y = (pts[i * 2 + 1] - cy) * scale + SIZE / 2;
      const color = interpolateSinebow(i / p);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.25)";
      ctx.lineWidth = 0.6;
      ctx.stroke();
    }
  }, [dataset, frame]);

  return <canvas ref={ref} className="rounded border border-[var(--rule)] bg-white" />;
}
