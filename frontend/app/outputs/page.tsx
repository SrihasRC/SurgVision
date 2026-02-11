"use client";

import { useEffect, useState } from "react";
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
  Play,
  Download,
  FolderOutput,
  Film,
  RefreshCw,
  Loader2,
  FileVideo,
  HardDrive,
} from "lucide-react";
import { listOutputs, getOutputVideoUrl, type OutputFile } from "@/lib/api-client";

export default function OutputsPage() {
  const [outputs, setOutputs] = useState<OutputFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<OutputFile | null>(null);

  const loadOutputs = async () => {
    setLoading(true);
    try {
      const files = await listOutputs();
      setOutputs(files);
      setError(null);
    } catch {
      setError(
        "Cannot load server outputs. Make sure the backend is running and the server_output folder exists."
      );
      setOutputs([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadOutputs();
  }, []);

  return (
    <div className="p-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Server Outputs
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Browse processed videos from the GPU server
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={loadOutputs}>
          <RefreshCw className={`mr-2 h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
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
                  href={getOutputVideoUrl(selectedVideo.name)}
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
              src={getOutputVideoUrl(selectedVideo.name)}
            >
              Your browser does not support the video tag.
            </video>
          </CardContent>
        </Card>
      )}

      {/* Video Grid */}
      {loading ? (
        <div className="flex h-48 items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : outputs.length === 0 && !error ? (
        <div className="flex h-48 flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed text-muted-foreground">
          <HardDrive className="h-8 w-8" />
          <p className="text-sm font-medium">No output files found</p>
          <p className="text-xs">
            Place processed videos in the server_output folder
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {outputs.map((f) => (
            <Card
              key={f.name}
              className={`group cursor-pointer transition-shadow hover:shadow-md ${
                selectedVideo?.name === f.name ? "ring-2 ring-primary" : ""
              }`}
              onClick={() => setSelectedVideo(f)}
            >
              <CardContent className="p-4">
                <div className="mb-3 flex aspect-video items-center justify-center overflow-hidden rounded-md bg-muted">
                  {f.thumbnail_url ? (
                    <img
                      src={f.thumbnail_url}
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
      )}
    </div>
  );
}
