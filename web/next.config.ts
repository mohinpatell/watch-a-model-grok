import type { NextConfig } from "next";

const basePath =
  process.env.GITHUB_PAGES === "true" ? "/watch-a-model-grok" : "";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  basePath,
};

export default nextConfig;
