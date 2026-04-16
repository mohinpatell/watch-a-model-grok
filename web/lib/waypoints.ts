export type Waypoint = {
  id: string;
  step: number;
  title: string;
};

export const waypoints: Waypoint[] = [
  { id: "init", step: 0, title: "at init, the model knows nothing" },
  { id: "memorize", step: 500, title: "memorization is fast" },
  { id: "plateau", step: 3000, title: "the long plateau" },
  { id: "onset", step: 6000, title: "something starts to give" },
  { id: "grok", step: 7800, title: "the snap — grokking" },
  { id: "generalized", step: 40000, title: "a learned algorithm" },
];

export function findFrameForStep(steps: number[], target: number): number {
  let best = 0;
  let bestDiff = Infinity;
  for (let i = 0; i < steps.length; i++) {
    const d = Math.abs(steps[i] - target);
    if (d < bestDiff) {
      bestDiff = d;
      best = i;
    }
  }
  return best;
}
