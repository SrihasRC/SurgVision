"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  ScanSearch,
  FolderOutput,
  Video,
  Cpu,
  Mic,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { Button } from "./ui/button";

const navItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/analysis", label: "Analysis", icon: ScanSearch },
  { href: "/live", label: "Live Preview", icon: Video },
  { href: "/voice-live", label: "Voice Live", icon: Mic },
  { href: "/outputs", label: "Server Outputs", icon: FolderOutput },
  { href: "/models", label: "Models", icon: Cpu },
];

interface SidebarProps {
  collapsed: boolean;
  toggle: () => void;
}

export function Sidebar({ collapsed, toggle }: SidebarProps) {
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
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 flex h-screen flex-col border-r border-border bg-sidebar transition-[width] duration-300 ease-in-out",
        collapsed ? "w-16" : "w-64"
      )}
    >
      {/* Header with Toggle */}
      <div
        className={cn(
          "flex h-16 items-center border-b border-border transition-all duration-300",
          collapsed ? "justify-center px-0" : "gap-3 px-4"
        )}
      >
        <Button
          variant="ghost"
          size="icon"
          onClick={toggle}
          className="h-9 w-9 shrink-0 text-muted-foreground hover:text-foreground hover:bg-sidebar-accent"
        >
          {collapsed ? (
            <PanelLeftOpen className="h-5 w-5" />
          ) : (
            <PanelLeftClose className="h-5 w-5" />
          )}
        </Button>
        {!collapsed && (
          <div className="overflow-hidden whitespace-nowrap">
            <h1 className="text-sm font-semibold tracking-tight text-foreground leading-tight">
              SurgVision AI
            </h1>
            <p className="text-[10px] font-medium text-muted-foreground leading-tight">
              Surgical Analysis Platform
            </p>
          </div>
        )}
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
              title={collapsed ? item.label : undefined}
              className={cn(
                "flex items-center gap-3 rounded-lg px-2.5 py-2.5 text-sm font-medium transition-colors",
                isActive
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground",
                collapsed ? "justify-center" : ""
              )}
            >
              <item.icon className="h-[18px] w-[18px] shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Backend Status footer */}
      <div className="border-t border-border p-3">
        <div
          className={cn(
            "flex items-center gap-2 text-[10px]",
            collapsed ? "justify-center" : "px-2"
          )}
          title={
            collapsed
              ? `Backend: ${
                  backendOnline === null
                    ? "checking"
                    : backendOnline
                    ? "online"
                    : "offline"
                }`
              : undefined
          }
        >
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full shrink-0",
              backendOnline === null
                ? "bg-muted-foreground animate-pulse"
                : backendOnline
                ? "bg-green-500"
                : "bg-red-500"
            )}
          />
          {!collapsed && (
            <span className="text-muted-foreground overflow-hidden text-ellipsis whitespace-nowrap font-medium">
              Backend{" "}
              {backendOnline === null
                ? "checking..."
                : backendOnline
                ? "connected"
                : "offline"}
            </span>
          )}
        </div>
      </div>
    </aside>
  );
}
