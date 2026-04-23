import type { Metadata } from "next";
import { Source_Serif_4, JetBrains_Mono } from "next/font/google";
import "katex/dist/katex.min.css";
import "./globals.css";

const serif = Source_Serif_4({
  variable: "--font-serif",
  subsets: ["latin"],
  weight: ["400", "600"],
  display: "swap",
});

const mono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
});

const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL ??
  "https://mohinpatell.github.io/watch-a-model-grok";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: "Watch a model grok",
  description:
    "A 1-layer transformer learns modular addition mod 113 — memorization, plateau, and the Fourier circuit that emerges at grokking.",
  openGraph: {
    title: "Watch a model grok",
    description:
      "A 1-layer transformer learns modular addition mod 113. It memorizes fast, plateaus, then snaps into a Fourier circuit.",
    type: "article",
    url: siteUrl,
  },
  twitter: {
    card: "summary_large_image",
    title: "Watch a model grok",
    description:
      "Memorization, plateau, and the Fourier circuit that emerges at grokking.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${serif.variable} ${mono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
