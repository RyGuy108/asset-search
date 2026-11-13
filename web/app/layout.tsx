import "./globals.css";

export const metadata = { title: "Asset search", description: "Search your past Magic Hour generations in plain language" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
