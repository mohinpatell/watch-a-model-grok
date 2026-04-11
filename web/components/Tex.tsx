"use client";

import { useEffect, useRef } from "react";
import katex from "katex";

type Props = {
  expr: string;
  display?: boolean;
};

export default function Tex({ expr, display = false }: Props) {
  const ref = useRef<HTMLSpanElement | HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    katex.render(expr, ref.current, {
      displayMode: display,
      throwOnError: false,
      strict: "ignore",
    });
  }, [expr, display]);

  // Render the raw expression as a text fallback so the slot reserves
  // layout space pre-hydration — katex.render then replaces it.
  if (display) {
    return (
      <div
        ref={ref as React.RefObject<HTMLDivElement>}
        className="my-5 overflow-x-auto text-center font-mono text-[var(--muted)]"
      >
        {expr}
      </div>
    );
  }
  return (
    <span
      ref={ref as React.RefObject<HTMLSpanElement>}
      className="inline-block font-mono text-[var(--muted)]"
    >
      {expr}
    </span>
  );
}
