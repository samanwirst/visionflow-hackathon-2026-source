import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VisionFlow AI",
  description: "Visual AI automation for automotive operations",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
