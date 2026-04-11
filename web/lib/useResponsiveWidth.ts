"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Observe the container's clientWidth and clamp to a max. Returns the
 * ref to attach to the container and the current width in px.
 */
export function useResponsiveWidth(
  maxWidth: number,
): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement | null>(null);
  const [w, setW] = useState(maxWidth);
  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const width = e.contentRect.width;
        if (width > 0) setW(Math.min(maxWidth, width));
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [maxWidth]);
  return [ref, w];
}
