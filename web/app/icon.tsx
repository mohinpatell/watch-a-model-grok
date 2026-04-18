import { ImageResponse } from "next/og";

export const dynamic = "force-static";
export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  const n = 16;
  const cx = 16;
  const cy = 16;
  const r = 11;
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
          width: 32,
          height: 32,
          background: "#09090b",
          display: "flex",
          position: "relative",
        }}
      >
        {dots.map((d, i) => (
          <div
            key={i}
            style={{
              position: "absolute",
              left: d.x - 2,
              top: d.y - 2,
              width: 4,
              height: 4,
              borderRadius: 2,
              background: `hsl(${d.hue}, 80%, 60%)`,
            }}
          />
        ))}
      </div>
    ),
    size,
  );
}
