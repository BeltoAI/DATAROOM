import "./globals.css";

export const metadata = {
  title: "DATAROOM",
  description: "Create datasets, analyze, and ask questions.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-screen bg-slate-900 text-white">
        {children}
      </body>
    </html>
  );
}
