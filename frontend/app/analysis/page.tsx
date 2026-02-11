"use client";

import { useCallback, useState, useRef, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import {
  Upload,
  Loader2,
  Download,
  Ruler,
  ScanSearch,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";
import {
  listModels,
  predictWithDistances,
  type ModelInfo,
  type DistancePredictionResponse,
} from "@/lib/api-client";

export default function AnalysisPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [topk, setTopk] = useState(7);

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<DistancePredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m);
        // default to 26n if available
        const def = m.find((x) => x.name === "26n") || m[0];
        if (def) setSelectedModel(def.name);
      })
      .catch(() => {});
  }, []);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f && f.type.startsWith("image/")) handleFile(f);
    },
    [handleFile]
  );

  const handleAnalyze = async () => {
    if (!file || !selectedModel) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictWithDistances(file, selectedModel, conf, iou, topk);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const downloadAnnotated = () => {
    if (!result?.annotated_image_base64) return;
    const link = document.createElement("a");
    link.href = `data:image/jpeg;base64,${result.annotated_image_base64}`;
    link.download = `annotated_${file?.name || "image"}.jpg`;
    link.click();
  };

  const statusIcon = (status: string) => {
    switch (status) {
      case "SAFE":
        return <ShieldCheck className="h-3.5 w-3.5 text-green-600" />;
      case "CAUTION":
        return <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />;
      case "DANGER":
        return <ShieldAlert className="h-3.5 w-3.5 text-red-500" />;
    }
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">Analysis</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Upload a surgical image for AI segmentation and distance measurement
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        {/* Left: Controls */}
        <div className="space-y-4">
          {/* Upload Zone */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Upload Image</CardTitle>
            </CardHeader>
            <CardContent>
              <div
                ref={dropRef}
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => fileInputRef.current?.click()}
                className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border py-8 transition-colors hover:border-primary/50 hover:bg-accent/30"
              >
                <Upload className="mb-2 h-8 w-8 text-muted-foreground" />
                <p className="text-sm font-medium">
                  {file ? file.name : "Drop image or click to upload"}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  JPEG, PNG supported
                </p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFile(f);
                }}
              />
            </CardContent>
          </Card>

          {/* Model & Parameters */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                  Model
                </label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map((m) => (
                      <SelectItem key={m.name} value={m.name}>
                        {m.name} ({m.size_mb} MB)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="mb-1.5 flex items-center justify-between text-xs font-medium text-muted-foreground">
                  Confidence <span className="font-mono">{conf.toFixed(2)}</span>
                </label>
                <Slider
                  value={[conf]}
                  onValueChange={([v]) => setConf(v)}
                  min={0.05}
                  max={0.95}
                  step={0.05}
                />
              </div>

              <div>
                <label className="mb-1.5 flex items-center justify-between text-xs font-medium text-muted-foreground">
                  IOU Threshold <span className="font-mono">{iou.toFixed(2)}</span>
                </label>
                <Slider
                  value={[iou]}
                  onValueChange={([v]) => setIou(v)}
                  min={0.1}
                  max={0.9}
                  step={0.05}
                />
              </div>

              <div>
                <label className="mb-1.5 flex items-center justify-between text-xs font-medium text-muted-foreground">
                  Top-K Distances <span className="font-mono">{topk}</span>
                </label>
                <Slider
                  value={[topk]}
                  onValueChange={([v]) => setTopk(v)}
                  min={1}
                  max={15}
                  step={1}
                />
              </div>
            </CardContent>
          </Card>

          {/* Analyze Button */}
          <Button
            onClick={handleAnalyze}
            disabled={!file || !selectedModel || loading}
            className="w-full"
            size="lg"
          >
            {loading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <ScanSearch className="mr-2 h-4 w-4" />
            )}
            {loading ? "Analyzing..." : "Run Analysis"}
          </Button>

          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600">
              {error}
            </div>
          )}
        </div>

        {/* Center + Right: Results */}
        <div className="xl:col-span-2 space-y-4">
          {/* Image View */}
          <Card className="overflow-hidden">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">
                  {result ? "Annotated Result" : "Preview"}
                </CardTitle>
                {result && (
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      {result.inference_time_ms.toFixed(0)}ms
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      {result.detections.length} detections
                    </Badge>
                    <Button variant="ghost" size="sm" onClick={downloadAnnotated}>
                      <Download className="mr-1 h-3.5 w-3.5" />
                      Save
                    </Button>
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {result?.annotated_image_base64 ? (
                <img
                  src={`data:image/jpeg;base64,${result.annotated_image_base64}`}
                  alt="Annotated result"
                  className="w-full rounded-md"
                />
              ) : previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="w-full rounded-md"
                />
              ) : (
                <div className="flex h-64 items-center justify-center rounded-lg bg-muted/30 text-sm text-muted-foreground">
                  Upload an image to begin
                </div>
              )}
            </CardContent>
          </Card>

          {/* Distance Results */}
          {result && result.distances.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <Ruler className="h-4 w-4 text-primary" />
                  <CardTitle className="text-sm">Distance Measurements</CardTitle>
                </div>
                <CardDescription className="text-xs">
                  Instrument to anatomical structure distances
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-lg border">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="px-3 py-2 text-left font-medium text-muted-foreground">
                          Structure
                        </th>
                        <th className="px-3 py-2 text-right font-medium text-muted-foreground">
                          Distance (px)
                        </th>
                        <th className="px-3 py-2 text-right font-medium text-muted-foreground">
                          Confidence
                        </th>
                        <th className="px-3 py-2 text-center font-medium text-muted-foreground">
                          Status
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.distances.map((d, i) => (
                        <tr key={i} className="border-b last:border-0">
                          <td className="px-3 py-2.5 font-medium">{d.organ}</td>
                          <td className="px-3 py-2.5 text-right font-mono">
                            {d.distance_px.toFixed(1)}
                          </td>
                          <td className="px-3 py-2.5 text-right font-mono text-muted-foreground">
                            {(d.organ_confidence * 100).toFixed(1)}%
                          </td>
                          <td className="px-3 py-2.5 text-center">
                            <Badge
                              className={
                                d.status === "SAFE"
                                  ? "status-safe border-0"
                                  : d.status === "CAUTION"
                                  ? "status-caution border-0"
                                  : "status-danger border-0"
                              }
                            >
                              <span className="flex items-center gap-1">
                                {statusIcon(d.status)}
                                {d.status}
                              </span>
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Detections Summary */}
          {result && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Detections Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {result.detections.map((d, i) => (
                    <Badge key={i} variant="secondary" className="text-xs">
                      {d.class_name}{" "}
                      <span className="ml-1 text-muted-foreground">
                        {(d.confidence * 100).toFixed(0)}%
                      </span>
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
