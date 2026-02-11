"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Play,
  Download,
  FolderOutput,
  Folder,
  Film,
  RefreshCw,
  Loader2,
  FileVideo,
  HardDrive,
  ChevronRight,
  Home,
} from "lucide-react";
import {
  listOutputs,
  getOutputVideoUrl,
  type OutputFile,
  type OutputFolder,
  type OutputListing,
} from "@/lib/api-client";
import Image from "next/image";

export default function OutputsPage() {
  const [listing, setListing] = useState<OutputListing | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<OutputFile | null>(null);
  const [currentPath, setCurrentPath] = useState("");

  const loadOutputs = useCallback(async (path: string = "") => {
    setLoading(true);
    try {
      const data = await listOutputs(path);
      setListing(data);
      setCurrentPath(path);
      setError(null);
      setSelectedVideo(null);
    } catch {
      setError(
        "Cannot load server outputs. Make sure the backend is running and the server_output folder exists."
      );
      setListing(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadOutputs("");
  }, [loadOutputs]);

  const navigateTo = (path: string) => {
    loadOutputs(path);
  };

  const goUp = () => {
    if (!currentPath) return;
    const parts = currentPath.split("/").filter(Boolean);
    parts.pop();
    navigateTo(parts.join("/"));
  };

  // Breadcrumb segments
  const pathSegments = currentPath
    ? currentPath.split("/").filter(Boolean)
    : [];

  return (
    <div className="p-8">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Server Outputs
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Browse processed videos from the GPU server
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => loadOutputs(currentPath)}
        >
          <RefreshCw
            className={`mr-2 h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </div>

      {/* Breadcrumbs */}
      <div className="mb-4 flex items-center gap-1 text-sm">
        <button
          onClick={() => navigateTo("")}
          className={`flex items-center gap-1 rounded px-2 py-1 transition-colors hover:bg-accent ${
            !currentPath
              ? "font-medium text-foreground"
              : "text-muted-foreground"
          }`}
        >
          <Home className="h-3.5 w-3.5" />
          server_output
        </button>
        {pathSegments.map((seg, i) => {
          const segPath = pathSegments.slice(0, i + 1).join("/");
          const isLast = i === pathSegments.length - 1;
          return (
            <span key={segPath} className="flex items-center gap-1">
              <ChevronRight className="h-3 w-3 text-muted-foreground" />
              <button
                onClick={() => navigateTo(segPath)}
                className={`rounded px-2 py-1 transition-colors hover:bg-accent ${
                  isLast
                    ? "font-medium text-foreground"
                    : "text-muted-foreground"
                }`}
              >
                {seg}
              </button>
            </span>
          );
        })}
      </div>

      {error && (
        <div className="mb-6 flex items-center gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700">
          <FolderOutput className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Video Player */}
      {selectedVideo && (
        <Card className="mb-6 overflow-hidden">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Film className="h-4 w-4 text-primary" />
                <CardTitle className="text-sm">{selectedVideo.name}</CardTitle>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs">
                  {selectedVideo.size_mb} MB
                </Badge>
                <a
                  href={getOutputVideoUrl(selectedVideo.path)}
                  download
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Button variant="ghost" size="sm">
                    <Download className="mr-1 h-3.5 w-3.5" />
                    Download
                  </Button>
                </a>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <video
              controls
              autoPlay
              className="w-full rounded-md bg-black"
              src={getOutputVideoUrl(selectedVideo.path)}
            >
              Your browser does not support the video tag.
            </video>
          </CardContent>
        </Card>
      )}

      {/* Content */}
      {loading ? (
        <div className="flex h-48 items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : !listing ||
        (listing.folders.length === 0 && listing.files.length === 0) ? (
        <div className="flex h-48 flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed text-muted-foreground">
          <HardDrive className="h-8 w-8" />
          <p className="text-sm font-medium">
            {currentPath ? "This folder is empty" : "No output files found"}
          </p>
          <p className="text-xs">
            Place processed videos in server_output/ subfolders
          </p>
          {currentPath && (
            <Button variant="outline" size="sm" className="mt-2" onClick={goUp}>
              Go Back
            </Button>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          {/* Folders */}
          {listing.folders.length > 0 && (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {listing.folders.map((f) => (
                <Card
                  key={f.path}
                  className="group cursor-pointer transition-shadow hover:shadow-md"
                  onClick={() => navigateTo(f.path)}
                >
                  <CardContent className="flex items-center gap-3 p-4">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                      <Folder className="h-5 w-5 text-primary" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">{f.name}</p>
                      <p className="text-[11px] text-muted-foreground">
                        {f.video_count} video{f.video_count !== 1 ? "s" : ""}
                        {f.subfolder_count > 0 &&
                          ` · ${f.subfolder_count} folder${
                            f.subfolder_count !== 1 ? "s" : ""
                          }`}
                      </p>
                    </div>
                    <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          {/* Files */}
          {listing.files.length > 0 && (
            <>
              {listing.folders.length > 0 && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <FileVideo className="h-3.5 w-3.5" />
                  <span>
                    {listing.files.length} video
                    {listing.files.length !== 1 ? "s" : ""}
                  </span>
                </div>
              )}
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {listing.files.map((f) => (
                  <Card
                    key={f.path}
                    className={`group cursor-pointer transition-shadow hover:shadow-md ${
                      selectedVideo?.path === f.path
                        ? "ring-2 ring-primary"
                        : ""
                    }`}
                    onClick={() => setSelectedVideo(f)}
                  >
                    <CardContent className="p-4">
                      <div className="relative mb-3 flex aspect-video items-center justify-center overflow-hidden rounded-md bg-muted">
                        {f.thumbnail_url ? (
                          <Image
                            src={
                              (process.env.NEXT_PUBLIC_API_URL ||
                                "http://localhost:8000") + f.thumbnail_url
                            }
                            alt={f.name}
                            className="h-full w-full object-cover"
                          />
                        ) : (
                          <FileVideo className="h-10 w-10 text-muted-foreground/50" />
                        )}
                        <div className="absolute inset-0 flex items-center justify-center rounded-md bg-black/0 transition-colors group-hover:bg-black/20">
                          <Play className="h-8 w-8 text-white opacity-0 transition-opacity group-hover:opacity-100" />
                        </div>
                      </div>
                      <p className="truncate text-sm font-medium">{f.name}</p>
                      <div className="mt-1 flex items-center justify-between text-xs text-muted-foreground">
                        <span>{f.size_mb} MB</span>
                        <span>{f.created}</span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
