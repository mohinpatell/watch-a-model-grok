"use client";

import { useEffect } from "react";
import { useStore } from "@/lib/store";
import { waypoints, findFrameForStep } from "@/lib/waypoints";

const FRAME_MS = 60;

export default function Scrubber() {
  const dataset = useStore((s) => s.dataset);
  const frame = useStore((s) => s.frame);
  const playing = useStore((s) => s.playing);
  const setFrame = useStore((s) => s.setFrame);
  const setPlaying = useStore((s) => s.setPlaying);

  useEffect(() => {
    if (!playing || !dataset) return;
    const last = dataset.meta.n_frames - 1;
    const id = window.setInterval(() => {
      const cur = useStore.getState().frame;
      if (cur >= last) {
        setPlaying(false);
        return;
      }
      setFrame(cur + 1);
    }, FRAME_MS);
    return () => window.clearInterval(id);
  }, [playing, dataset, setFrame, setPlaying]);

  if (!dataset) return null;

  const step = dataset.meta.steps[frame];
  const trainAcc = dataset.meta.train_acc[frame];
  const testAcc = dataset.meta.test_acc[frame];
  const nFrames = dataset.meta.n_frames;
  const atEnd = frame >= nFrames - 1;

  const togglePlay = () => {
    if (atEnd) setFrame(0);
    setPlaying(!playing);
  };

  return (
    <div className="flex flex-col gap-2 font-mono text-sm">
      <div className="flex items-center gap-3 text-[var(--muted)]">
        <button
          type="button"
          onClick={togglePlay}
          className="shrink-0 inline-flex items-center gap-1.5 px-2.5 py-1 rounded border border-[var(--rule)] bg-white hover:border-[var(--accent)] hover:text-[var(--accent)] transition-colors"
          aria-label={playing ? "pause" : atEnd ? "replay" : "play"}
        >
          <span aria-hidden>{playing ? "❙❙" : atEnd ? "↻" : "▶"}</span>
          <span>{playing ? "pause" : atEnd ? "replay" : "play"}</span>
        </button>
        <span className="truncate">
          step{" "}
          <span className="text-[var(--foreground)] tabular-nums">
            {step.toLocaleString()}
          </span>
        </span>
        <span className="ml-auto text-xs shrink-0">
          train{" "}
          <span className="tabular-nums" style={{ color: "var(--train)" }}>
            {trainAcc.toFixed(2)}
          </span>
          <span className="text-[var(--faint)]"> · </span>
          test{" "}
          <span className="tabular-nums" style={{ color: "var(--test)" }}>
            {testAcc.toFixed(2)}
          </span>
        </span>
      </div>
      <div className="relative">
        <input
          type="range"
          min={0}
          max={nFrames - 1}
          value={frame}
          onChange={(e) => {
            setFrame(Number(e.target.value));
            if (playing) setPlaying(false);
          }}
          className="w-full accent-[var(--accent)] relative z-10"
          aria-label="training step"
          aria-valuetext={`step ${step.toLocaleString()}, test accuracy ${testAcc.toFixed(2)}`}
        />
        <div className="pointer-events-none absolute inset-x-0 -bottom-3 h-3">
          {waypoints.map((wp) => {
            const wpFrame = findFrameForStep(dataset.meta.steps, wp.step);
            const pct = (wpFrame / (nFrames - 1)) * 100;
            const active = Math.abs(wpFrame - frame) < 2;
            return (
              <div
                key={wp.id}
                title={`${wp.title} · step ${wp.step.toLocaleString()}`}
                className={`absolute top-0 h-2 w-[2px] -translate-x-1/2 ${
                  active ? "bg-[var(--accent)]" : "bg-[var(--faint)]"
                }`}
                style={{ left: `${pct}%` }}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
