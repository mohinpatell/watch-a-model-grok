import { ImageResponse } from "next/og";

export const dynamic = "force-static";
export const alt =
  "watch a model grok — scroll-driven visualization of a 1-layer transformer learning modular addition";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function Image() {
  const p = 113;
  const k = 46;
  const cx = 900;
  const cy = 315;
  const r = 180;
  const dots = Array.from({ length: p }, (_, i) => {
    const theta = (2 * Math.PI * k * i) / p;
    return {
      x: cx + r * Math.cos(theta),
      y: cy + r * Math.sin(theta),
      hue: (i / p) * 360,
    };
  });

  return new ImageResponse(
    (
      <div
        style={{
          width: 1200,
          height: 630,
          background: "#09090b",
          color: "#fafafa",
          display: "flex",
          fontFamily: "ui-sans-serif, system-ui, sans-serif",
          position: "relative",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: 1200,
            height: 630,
            display: "flex",
          }}
        >
          {dots.map((d, i) => (
            <div
              key={i}
              style={{
                position: "absolute",
                left: d.x - 7,
                top: d.y - 7,
                width: 14,
                height: 14,
                borderRadius: 7,
                background: `hsl(${d.hue}, 80%, 60%)`,
                opacity: 0.95,
              }}
            />
          ))}
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            padding: "80px",
            maxWidth: 640,
            marginTop: "auto",
          }}
        >
          <div
            style={{
              fontSize: 22,
              fontFamily: "ui-monospace, monospace",
              color: "#fbbf24",
              letterSpacing: 2,
              marginBottom: 24,
            }}
          >
            step 40,000 · test acc 1.00
          </div>
          <div
            style={{
              fontSize: 72,
              fontWeight: 600,
              letterSpacing: "-0.02em",
              lineHeight: 1.05,
              marginBottom: 24,
              display: "flex",
              flexDirection: "column",
              color: "#fafafa",
            }}
          >
            <div style={{ display: "flex" }}>watch a model</div>
            <div style={{ display: "flex", color: "#fbbf24" }}>grok</div>
          </div>
          <div
            style={{
              fontSize: 26,
              color: "#a1a1aa",
              lineHeight: 1.4,
              marginBottom: 24,
            }}
          >
            a 1-layer transformer learns modular addition mod 113, memorizes,
            plateaus, then snaps into a Fourier circuit
          </div>
          <div
            style={{
              fontSize: 18,
              fontFamily: "ui-monospace, monospace",
              color: "#71717a",
            }}
          >
            after Nanda et al. 2023
          </div>
        </div>
      </div>
    ),
    size,
  );
}
