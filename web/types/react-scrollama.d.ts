declare module "react-scrollama" {
  import type { ReactNode } from "react";

  export type CallbackResponse<T = unknown> = {
    data: T;
    direction: "up" | "down";
    element: HTMLElement;
    entry: IntersectionObserverEntry;
  };

  export type ScrollamaProps = {
    children?: ReactNode;
    offset?: number;
    onStepEnter?: (response: CallbackResponse) => void;
    onStepExit?: (response: CallbackResponse) => void;
    onStepProgress?: (response: CallbackResponse & { progress: number }) => void;
    debug?: boolean;
    threshold?: number;
    rootMargin?: string;
  };

  export type StepProps = {
    children?: ReactNode;
    data?: unknown;
  };

  export const Scrollama: React.FC<ScrollamaProps>;
  export const Step: React.FC<StepProps>;
}
