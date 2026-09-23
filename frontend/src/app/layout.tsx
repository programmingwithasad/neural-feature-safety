import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Neural Feature Safety",
  description: "Neural safety classification and feature analysis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}