"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  Video,
  Play,
  Square,
  Upload,
  Wifi,
  WifiOff,
  FileVideo,
} from "lucide-react";
import {
  listModels,
  getStreamStartUrl,
  stopStream,
  type ModelInfo,
} from "@/lib/api-client";

export default function LivePage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [conf, setConf] = useState(0.25);

  const [file, setFile] = useState<File | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m);
        const def = m.find((x) => x.name === "26n") || m[0];
        if (def) setSelectedModel(def.name);
      })
      .catch(() => {});
  }, []);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    // Reset any previous stream
    setStreamUrl(null);
    setStreaming(false);
    setSessionId(null);
  }, []);

  const startStream = async () => {
    if (!file || !selectedModel) return;

    // Build the form and POST to /stream/start
    const form = new FormData();
    form.append("file", file);
    const url = getStreamStartUrl(selectedModel, conf);

    try {
      const res = await fetch(url, { method: "POST", body: form });
      // Get session ID from response header
      const sid = res.headers.get("X-Session-Id");
      if (sid) setSessionId(sid);

      // The response body is the MJPEG stream.
      // We need to create a blob URL from the stream for the <img> tag.
      // Since MJPEG is multipart, the simplest approach is to use the
      // fetch URL directly — but fetch already consumed it.
      // Instead, we'll use an <img> with a form-based POST trick via an iframe,
      // or we can re-POST. Let's use a simpler approach:
      // We send the file to server, server saves it, and we GET a stream URL.
      // Actually the best approach for MJPEG in browser is to use the URL directly.

      // We'll adjust: upload file first, get a session back, then stream via GET.
      // But our current backend is POST-based stream. Let's use a workaround:
      // Read the multipart stream and render frames on a canvas.

      if (!res.ok || !res.body) {
        throw new Error("Failed to start stream");
      }

      setStreaming(true);
      renderMjpegStream(res.body);
    } catch (e) {
      console.error("Stream error:", e);
      setStreaming(false);
    }
  };

  const renderMjpegStream = async (body: ReadableStream<Uint8Array>) => {
    const reader = body.getReader();
    let buffer = new Uint8Array(0);

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Append chunk
        const newBuf = new Uint8Array(buffer.length + value.length);
        newBuf.set(buffer);
        newBuf.set(value, buffer.length);
        buffer = newBuf;

        // Find JPEG boundaries (SOI = 0xFFD8, EOI = 0xFFD9)
        while (true) {
          const soiIdx = findBytes(buffer, [0xff, 0xd8]);
          if (soiIdx === -1) break;
          const eoiIdx = findBytes(buffer, [0xff, 0xd9], soiIdx + 2);
          if (eoiIdx === -1) break;

          const jpegEnd = eoiIdx + 2;
          const jpeg = buffer.slice(soiIdx, jpegEnd);
          buffer = buffer.slice(jpegEnd);

          // Display frame
          const blob = new Blob([jpeg], { type: "image/jpeg" });
          const url = URL.createObjectURL(blob);
          if (imgRef.current) {
            const prevSrc = imgRef.current.src;
            imgRef.current.src = url;
            if (prevSrc.startsWith("blob:")) URL.revokeObjectURL(prevSrc);
          }
        }
      }
    } catch {
      // Stream ended or cancelled
    } finally {
      reader.releaseLock();
      setStreaming(false);
    }
  };

  const findBytes = (
    haystack: Uint8Array,
    needle: number[],
    startFrom = 0
  ): number => {
    for (let i = startFrom; i <= haystack.length - needle.length; i++) {
      let found = true;
      for (let j = 0; j < needle.length; j++) {
        if (haystack[i + j] !== needle[j]) {
          found = false;
          break;
        }
      }
      if (found) return i;
    }
    return -1;
  };

  const handleStop = async () => {
    if (sessionId) {
      try {
        await stopStream(sessionId);
      } catch {
        // Session may have already ended
      }
    }
    setStreaming(false);
    setStreamUrl(null);
    setSessionId(null);
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">
          Live Inference
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Upload a video for real-time AI inference frame by frame
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-4">
        {/* Controls */}
        <div className="space-y-4">
          {/* Video Upload */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Video Input</CardTitle>
            </CardHeader>
            <CardContent>
              <div
                onClick={() => fileInputRef.current?.click()}
                className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border py-6 transition-colors hover:border-primary/50 hover:bg-accent/30"
              >
                {file ? (
                  <>
                    <FileVideo className="mb-2 h-7 w-7 text-primary" />
                    <p className="text-sm font-medium truncate max-w-full px-2">
                      {file.name}
                    </p>
                    <p className="mt-1 text-[11px] text-muted-foreground">
                      {(file.size / (1024 * 1024)).toFixed(1)} MB — click to
                      change
                    </p>
                  </>
                ) : (
                  <>
                    <Upload className="mb-2 h-7 w-7 text-muted-foreground" />
                    <p className="text-sm font-medium">Upload video file</p>
                    <p className="mt-1 text-[11px] text-muted-foreground">
                      MP4, AVI, MOV supported
                    </p>
                  </>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp4,.avi,.mov,.mkv,.webm,video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,video/webm"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFile(f);
                }}
              />
            </CardContent>
          </Card>

          {/* Model & Confidence */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                  Model
                </label>
                <Select
                  value={selectedModel}
                  onValueChange={setSelectedModel}
                  disabled={streaming}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map((m) => (
                      <SelectItem key={m.name} value={m.name}>
                        {m.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="mb-1.5 flex items-center justify-between text-xs font-medium text-muted-foreground">
                  Confidence{" "}
                  <span className="font-mono">{conf.toFixed(2)}</span>
                </label>
                <Slider
                  value={[conf]}
                  onValueChange={([v]) => setConf(v)}
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  disabled={streaming}
                />
              </div>
            </CardContent>
          </Card>

          {/* Start / Stop */}
          <div className="flex gap-2">
            {!streaming ? (
              <Button
                onClick={startStream}
                className="flex-1"
                size="lg"
                disabled={!file || !selectedModel}
              >
                <Play className="mr-2 h-4 w-4" />
                Start Inference
              </Button>
            ) : (
              <Button
                onClick={handleStop}
                variant="destructive"
                className="flex-1"
                size="lg"
              >
                <Square className="mr-2 h-4 w-4" />
                Stop
              </Button>
            )}
          </div>

          {/* Status */}
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-xs">
                {streaming ? (
                  <>
                    <Wifi className="h-3.5 w-3.5 text-green-500" />
                    <span className="text-green-600 font-medium">
                      Inference running
                    </span>
                  </>
                ) : (
                  <>
                    <WifiOff className="h-3.5 w-3.5 text-muted-foreground" />
                    <span className="text-muted-foreground">Idle</span>
                  </>
                )}
              </div>
              {sessionId && (
                <p className="mt-1.5 truncate font-mono text-[10px] text-muted-foreground">
                  Session: {sessionId.slice(0, 8)}…
                </p>
              )}
              <p className="mt-2 text-[11px] text-muted-foreground">
                Upload a video and click Start to process each frame through the
                YOLO model. Stop cancels the backend inference.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Stream View */}
        <div className="xl:col-span-3">
          <Card className="overflow-hidden">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Video className="h-4 w-4 text-primary" />
                  <CardTitle className="text-sm">Inference Output</CardTitle>
                </div>
                {streaming && (
                  <Badge className="bg-red-500 text-white animate-pulse">
                    ● PROCESSING
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {streaming ? (
                <div className="relative aspect-video overflow-hidden rounded-md bg-black">
                  <img
                    ref={imgRef}
                    alt="Inference stream"
                    className="h-full w-full object-contain"
                  />
                </div>
              ) : (
                <div className="flex aspect-video flex-col items-center justify-center rounded-lg bg-muted/30 text-muted-foreground">
                  <Video className="mb-3 h-12 w-12 opacity-30" />
                  <p className="text-sm font-medium">No active inference</p>
                  <p className="mt-1 text-xs">
                    Upload a video and click &ldquo;Start Inference&rdquo;
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
