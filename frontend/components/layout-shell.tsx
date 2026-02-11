"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/sidebar";
import { cn } from "@/lib/utils";

export function LayoutShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);

  // Optional: Persist state
  useEffect(() => {
    const saved = localStorage.getItem("sidebar-collapsed");
    if (saved) setCollapsed(saved === "true");
  }, []);

  const toggle = () => {
    const next = !collapsed;
    setCollapsed(next);
    localStorage.setItem("sidebar-collapsed", String(next));
  };

  return (
    <>
      <Sidebar collapsed={collapsed} toggle={toggle} />
      <main
        className={cn(
          "min-h-screen transition-[margin] duration-300 ease-in-out",
          collapsed ? "ml-16" : "ml-64"
        )}
      >
        {children}
      </main>
    </>
  );
}
