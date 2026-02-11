"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  ScanSearch,
  FolderOutput,
  Video,
  Cpu,
  Activity,
  Mic,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

const navItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/analysis", label: "Analysis", icon: ScanSearch },
  { href: "/live", label: "Live Preview", icon: Video },
  { href: "/voice-live", label: "Voice Live", icon: Mic },
  { href: "/outputs", label: "Server Outputs", icon: FolderOutput },
  { href: "/models", label: "Models", icon: Cpu },
];

export function Sidebar() {
  const pathname = usePathname();
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(
          (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/"
        );
        setBackendOnline(res.ok);
      } catch {
        setBackendOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-64 flex-col border-r border-border bg-sidebar">
      {/* Logo */}
      <div className="flex h-16 items-center gap-3 border-b border-border px-6">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
          <Activity className="h-5 w-5 text-primary-foreground" />
        </div>
        <div>
          <h1 className="text-base font-semibold tracking-tight text-foreground">
            SurgVision AI
          </h1>
          <p className="text-[11px] font-medium text-muted-foreground">
            Surgical Analysis Platform
          </p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navItems.map((item) => {
          const isActive =
            pathname === item.href ||
            (item.href !== "/" && pathname.startsWith(item.href));
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                isActive
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
              )}
            >
              <item.icon className="h-[18px] w-[18px]" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Backend Status */}
      <div className="border-t border-border px-4 py-3">
        <div className="flex items-center gap-2 text-xs">
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              backendOnline === null
                ? "bg-muted-foreground animate-pulse"
                : backendOnline
                ? "bg-green-500"
                : "bg-red-500"
            )}
          />
          <span className="text-muted-foreground">
            Backend{" "}
            {backendOnline === null
              ? "checking..."
              : backendOnline
              ? "connected"
              : "offline"}
          </span>
        </div>
      </div>
    </aside>
  );
}
