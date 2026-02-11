"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  ScanSearch,
  Video,
  FolderOutput,
  Cpu,
  Activity,
  CheckCircle2,
  XCircle,
  ArrowRight,
} from "lucide-react";
import { getHealth, listModels, type HealthResponse, type ModelInfo } from "@/lib/api-client";

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [h, m] = await Promise.all([getHealth(), listModels()]);
        setHealth(h);
        setModels(m);
        setError(null);
      } catch {
        setError("Cannot connect to backend. Make sure the API is running on port 8000.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Overview of your surgical analysis platform
        </p>
      </div>

      {/* Status Banner */}
      {!loading && (
        <div className="mb-6">
          {error ? (
            <div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              <XCircle className="h-4 w-4 shrink-0" />
              {error}
            </div>
          ) : (
            <div className="flex items-center gap-3 rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">
              <CheckCircle2 className="h-4 w-4 shrink-0" />
              Backend connected — {health?.models_available} models available
              {health?.default_model && (
                <Badge variant="secondary" className="ml-auto">
                  Default: {health.default_model}
                </Badge>
              )}
            </div>
          )}
        </div>
      )}

      {/* Stats Cards */}
      <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Status
            </CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  health ? "bg-green-500" : loading ? "bg-yellow-400 animate-pulse" : "bg-red-500"
                }`}
              />
              <span className="text-xl font-semibold">
                {loading ? "Connecting..." : health ? "Online" : "Offline"}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Models Available
            </CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <p className="text-xl font-semibold">
              {health?.models_available ?? "—"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Default Model
            </CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <p className="text-xl font-semibold">
              {health?.default_model ?? "—"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Size
            </CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <p className="text-xl font-semibold">
              {models.length > 0
                ? `${models.reduce((s, m) => s + m.size_mb, 0).toFixed(1)} MB`
                : "—"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <h2 className="mb-4 text-lg font-semibold">Quick Actions</h2>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Link href="/analysis">
          <Card className="group cursor-pointer transition-shadow hover:shadow-md">
            <CardHeader>
              <div className="mb-2 flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <ScanSearch className="h-5 w-5 text-primary" />
              </div>
              <CardTitle className="text-base">Analyze Image</CardTitle>
              <CardDescription>
                Upload a surgical image for AI segmentation and distance
                measurement
              </CardDescription>
            </CardHeader>
            <CardContent>
              <span className="inline-flex items-center gap-1 text-sm font-medium text-primary group-hover:gap-2 transition-all">
                Start Analysis <ArrowRight className="h-3.5 w-3.5" />
              </span>
            </CardContent>
          </Card>
        </Link>

        <Link href="/live">
          <Card className="group cursor-pointer transition-shadow hover:shadow-md">
            <CardHeader>
              <div className="mb-2 flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <Video className="h-5 w-5 text-primary" />
              </div>
              <CardTitle className="text-base">Live Preview</CardTitle>
              <CardDescription>
                Stream real-time video with AI overlay for live surgical
                monitoring
              </CardDescription>
            </CardHeader>
            <CardContent>
              <span className="inline-flex items-center gap-1 text-sm font-medium text-primary group-hover:gap-2 transition-all">
                Open Stream <ArrowRight className="h-3.5 w-3.5" />
              </span>
            </CardContent>
          </Card>
        </Link>

        <Link href="/outputs">
          <Card className="group cursor-pointer transition-shadow hover:shadow-md">
            <CardHeader>
              <div className="mb-2 flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <FolderOutput className="h-5 w-5 text-primary" />
              </div>
              <CardTitle className="text-base">Server Outputs</CardTitle>
              <CardDescription>
                Browse and play processed videos from the GPU server
              </CardDescription>
            </CardHeader>
            <CardContent>
              <span className="inline-flex items-center gap-1 text-sm font-medium text-primary group-hover:gap-2 transition-all">
                View Outputs <ArrowRight className="h-3.5 w-3.5" />
              </span>
            </CardContent>
          </Card>
        </Link>
      </div>

      {/* Models Quick View */}
      {models.length > 0 && (
        <>
          <h2 className="mb-4 mt-8 text-lg font-semibold">Available Models</h2>
          <div className="rounded-lg border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                    Name
                  </th>
                  <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                    Size
                  </th>
                  <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                    Location
                  </th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.name} className="border-b last:border-0">
                    <td className="px-4 py-3 font-medium">
                      {m.name}
                      {m.name === health?.default_model && (
                        <Badge variant="outline" className="ml-2 text-[10px]">
                          default
                        </Badge>
                      )}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {m.size_mb} MB
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                      {m.directory.split("/").slice(-2).join("/")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
