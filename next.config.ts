import type { NextConfig } from "next";

// Root-level config used when Vercel builds from the repo root.
// Mirrors frontend/next.config.ts exactly.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: false,
};

export default nextConfig;
