const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ModelInfo {
  name: string;
  path: string;
  size_mb: number;
  directory: string;
  class_names?: Record<number, string>;
  task?: string;
}

export interface Detection {
  class_name: string;
  class_id: number;
  confidence: number;
  bbox: number[];
  mask_polygon?: number[][];
}

export interface DistanceMeasurement {
  organ: string;
  distance_px: number;
  status: "SAFE" | "CAUTION" | "DANGER";
  instrument_point: number[];
  organ_point: number[];
  organ_confidence: number;
}

export interface PredictionResponse {
  model_name: string;
  image_width: number;
  image_height: number;
  detections: Detection[];
  annotated_image_base64?: string;
  inference_time_ms: number;
}

export interface DistancePredictionResponse extends PredictionResponse {
  distances: DistanceMeasurement[];
}

export interface VideoResponse {
  model_name: string;
  total_frames: number;
  output_path: string;
  processing_time_s: number;
}

export interface OutputFolder {
  name: string;
  path: string;
  video_count: number;
  subfolder_count: number;
}

export interface OutputFile {
  name: string;
  path: string;
  size_mb: number;
  created: string;
  thumbnail_url: string;
  video_url: string;
}

export interface OutputListing {
  current_path: string;
  folders: OutputFolder[];
  files: OutputFile[];
}

export interface HealthResponse {
  status: string;
  models_available: number;
  default_model: string | null;
}

// ─────── Health ───────
export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/`);
  if (!res.ok) throw new Error("Backend unreachable");
  return res.json();
}

// ─────── Models ───────
export async function listModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE}/models/`);
  if (!res.ok) throw new Error("Failed to list models");
  return res.json();
}

export async function getModelInfo(name: string): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/models/${name}/info`);
  if (!res.ok) throw new Error(`Model '${name}' not found`);
  return res.json();
}

// ─────── Prediction ───────
export async function predictImage(
  file: File,
  modelName: string,
  conf: number = 0.25,
  iou: number = 0.45,
  annotate: boolean = true
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams({
    model_name: modelName,
    conf: conf.toString(),
    iou: iou.toString(),
    annotate: annotate.toString(),
  });
  const res = await fetch(`${API_BASE}/predict/image?${params}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("Prediction failed");
  return res.json();
}

export async function predictWithDistances(
  file: File,
  modelName: string,
  conf: number = 0.25,
  iou: number = 0.45,
  topk: number = 7
): Promise<DistancePredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams({
    model_name: modelName,
    conf: conf.toString(),
    iou: iou.toString(),
    topk: topk.toString(),
  });
  const res = await fetch(`${API_BASE}/predict/image/distances?${params}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("Distance prediction failed");
  return res.json();
}

// ─────── Outputs (subfolder browsing) ───────
export async function listOutputs(path: string = ""): Promise<OutputListing> {
  const params = path ? `?path=${encodeURIComponent(path)}` : "";
  const res = await fetch(`${API_BASE}/outputs/${params}`);
  if (!res.ok) throw new Error("Failed to list outputs");
  return res.json();
}

export function getOutputVideoUrl(filepath: string): string {
  return `${API_BASE}/outputs/file/${filepath}`;
}

export function getOutputThumbnailUrl(filepath: string): string {
  return `${API_BASE}/outputs/thumbnail/${filepath}`;
}

// ─────── Live Stream (video upload) ───────
export function getStreamStartUrl(
  modelName: string,
  conf: number = 0.25
): string {
  return `${API_BASE}/stream/start?model_name=${modelName}&conf=${conf}`;
}

// Two-step flow for voice-live (avoids CORS header issue)
export async function createStreamSession(
  file: File,
  modelName: string,
  conf: number = 0.25
): Promise<{ session_id: string; feed_url: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(
    `${API_BASE}/stream/create?model_name=${modelName}&conf=${conf}`,
    { method: "POST", body: form }
  );
  if (!res.ok) throw new Error("Failed to create stream session");
  return res.json();
}

export function getStreamFeedUrl(sessionId: string): string {
  return `${API_BASE}/stream/feed/${sessionId}`;
}

export async function stopStream(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/stream/stop/${sessionId}`, { method: "POST" });
}

export interface CommandResponse {
  status: string;
  confirmation: string;
  config: {
    show_masks: boolean;
    show_labels: boolean;
    hidden_classes: string[];
  };
}

export async function sendStreamCommand(
  sessionId: string,
  command: string,
  className?: string
): Promise<CommandResponse> {
  const res = await fetch(`${API_BASE}/stream/command/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command, class_name: className }),
  });
  if (!res.ok) throw new Error("Command failed");
  return res.json();
}

export async function speakTts(text: string): Promise<ArrayBuffer> {
  const res = await fetch(`${API_BASE}/tts/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error("TTS failed");
  return res.arrayBuffer();
}
