import { create } from "zustand";
import type { Dataset } from "./data";

type State = {
  dataset: Dataset | null;
  frame: number;
  playing: boolean;
  setDataset: (d: Dataset) => void;
  setFrame: (f: number) => void;
  setPlaying: (p: boolean) => void;
};

export const useStore = create<State>((set) => ({
  dataset: null,
  frame: 0,
  playing: false,
  setDataset: (dataset) => set({ dataset, frame: 0, playing: false }),
  setFrame: (frame) => set({ frame }),
  setPlaying: (playing) => set({ playing }),
}));
