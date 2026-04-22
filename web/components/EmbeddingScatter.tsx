"use client";

import { useEffect, useRef } from "react";
import { interpolateSinebow } from "d3-scale-chromatic";
import { useStore } from "@/lib/store";
import { embedsFrame } from "@/lib/data";
import { useResponsiveWidth } from "@/lib/useResponsiveWidth";

const MAX_SIZE = 320;
const PAD = 22;

export default function EmbeddingScatter() {
  const [wrapRef, size] = useResponsiveWidth(MAX_SIZE);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dataset = useStore((s) => s.dataset);
  const frame = useStore((s) => s.frame);

  // Resize canvas pixel buffer (DPR-aware) whenever rendered size changes.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }, [size]);

  useEffect(() => {
    if (!dataset || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
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
    const scale = (size - 2 * PAD) / span;
    const dotR = Math.max(2.4, (size / MAX_SIZE) * 3.2);

    ctx.clearRect(0, 0, size, size);

    ctx.strokeStyle = "#e8e8e8";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD, size / 2);
    ctx.lineTo(size - PAD, size / 2);
    ctx.moveTo(size / 2, PAD);
    ctx.lineTo(size / 2, size - PAD);
    ctx.stroke();

    for (let i = 0; i < p; i++) {
      const x = (pts[i * 2] - cx) * scale + size / 2;
      const y = (pts[i * 2 + 1] - cy) * scale + size / 2;
      const color = interpolateSinebow(i / p);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, dotR, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.25)";
      ctx.lineWidth = 0.6;
      ctx.stroke();
    }
  }, [dataset, frame, size]);

  return (
    <div ref={wrapRef} className="w-full flex justify-center">
      <canvas
        ref={canvasRef}
        role="img"
        aria-label="Token embeddings projected onto the 2D cosine/sine Fourier basis at the dominant frequency"
        className="rounded border border-[var(--rule)] bg-white"
      />
    </div>
  );
}
