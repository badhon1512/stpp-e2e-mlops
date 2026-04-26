import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Support Ticket Priority",
  description: "Predict support ticket priority with the local MLOps backend."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
