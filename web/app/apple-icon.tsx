import { ImageResponse } from "next/og";

export const dynamic = "force-static";
export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default function AppleIcon() {
  const n = 32;
  const cx = 90;
  const cy = 90;
  const r = 64;
  const dots = Array.from({ length: n }, (_, i) => {
    const theta = (2 * Math.PI * i) / n;
    return {
      x: cx + r * Math.cos(theta),
      y: cy + r * Math.sin(theta),
      hue: (i / n) * 360,
    };
  });

  return new ImageResponse(
    (
      <div
        style={{
          width: 180,
          height: 180,
          background: "#09090b",
          display: "flex",
          position: "relative",
          borderRadius: 40,
        }}
      >
        {dots.map((d, i) => (
          <div
            key={i}
            style={{
              position: "absolute",
              left: d.x - 6,
              top: d.y - 6,
              width: 12,
              height: 12,
              borderRadius: 6,
              background: `hsl(${d.hue}, 80%, 62%)`,
            }}
          />
        ))}
      </div>
    ),
    size,
  );
}
