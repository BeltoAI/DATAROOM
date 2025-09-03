"use client";
import { useEffect, useState } from "react";
export default function ClientOnly({ children }) {
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);
  if (!ready) return null;
  return children;
}
