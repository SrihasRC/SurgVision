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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Video, Play, Square, RefreshCw, Wifi, WifiOff } from "lucide-react";
import { listModels, getLiveStreamUrl, type ModelInfo } from "@/lib/api-client";

export default function LivePage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [conf, setConf] = useState(0.25);
  const [streaming, setStreaming] = useState(false);
  const [streamUrl, setStreamUrl] = useState<string | null>(null);

  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m);
        const def = m.find((x) => x.name === "26n") || m[0];
        if (def) setSelectedModel(def.name);
      })
      .catch(() => {});
  }, []);

  const startStream = () => {
    if (!selectedModel) return;
    setStreamUrl(getLiveStreamUrl(selectedModel, conf));
    setStreaming(true);
  };

  const stopStream = () => {
    setStreamUrl(null);
    setStreaming(false);
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">Live Preview</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Real-time video stream with AI segmentation overlay
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-4">
        {/* Controls */}
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Stream Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                  Model
                </label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
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

          <div className="flex gap-2">
            {!streaming ? (
              <Button onClick={startStream} className="flex-1" size="lg">
                <Play className="mr-2 h-4 w-4" />
                Start Stream
              </Button>
            ) : (
              <Button
                onClick={stopStream}
                variant="destructive"
                className="flex-1"
                size="lg"
              >
                <Square className="mr-2 h-4 w-4" />
                Stop
              </Button>
            )}
          </div>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2 text-xs">
                {streaming ? (
                  <>
                    <Wifi className="h-3.5 w-3.5 text-green-500" />
                    <span className="text-green-600 font-medium">
                      Stream active
                    </span>
                  </>
                ) : (
                  <>
                    <WifiOff className="h-3.5 w-3.5 text-muted-foreground" />
                    <span className="text-muted-foreground">
                      Stream inactive
                    </span>
                  </>
                )}
              </div>
              <p className="mt-2 text-[11px] text-muted-foreground">
                Streams MJPEG from backend. Requires the backend to have a video
                source (webcam or video file).
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
                  <CardTitle className="text-sm">Live Feed</CardTitle>
                </div>
                {streaming && (
                  <Badge className="bg-red-500 text-white animate-pulse">
                    ● LIVE
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {streaming && streamUrl ? (
                <div className="relative aspect-video overflow-hidden rounded-md bg-black">
                  <img
                    src={streamUrl}
                    alt="Live stream"
                    className="h-full w-full object-contain"
                  />
                </div>
              ) : (
                <div className="flex aspect-video flex-col items-center justify-center rounded-lg bg-muted/30 text-muted-foreground">
                  <Video className="mb-3 h-12 w-12 opacity-30" />
                  <p className="text-sm font-medium">No active stream</p>
                  <p className="mt-1 text-xs">
                    Select a model and click &ldquo;Start Stream&rdquo;
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
