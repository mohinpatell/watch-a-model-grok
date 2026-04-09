"use client";

import { useEffect } from "react";
import { loadDataset } from "@/lib/data";
import { useStore } from "@/lib/store";

// Fires the dataset load on mount. Renders nothing — chart components
// handle their own pending-data state.
export default function DataLoader() {
  const setDataset = useStore((s) => s.setDataset);

  useEffect(() => {
    let cancelled = false;
    loadDataset()
      .then((d) => {
        if (!cancelled) setDataset(d);
      })
      .catch((err) => {
        console.error("Failed to load dataset:", err);
        // Chart components will stay blank; no essay-wide blocker.
      });
    return () => {
      cancelled = true;
    };
  }, [setDataset]);

  return null;
}
