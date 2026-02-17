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
  Mic,
  MicOff,
  FileVideo,
  MessageSquare,
  Eye,
  EyeOff,
  Tag,
  Layers,
} from "lucide-react";
import {
  listModels,
  createStreamSession,
  getStreamFeedUrl,
  stopStream,
  sendStreamCommand,
  speakTts,
  type ModelInfo,
} from "@/lib/api-client";

// ─── Voice command parser ───────────────────────────────────────

interface ParsedCommand {
  command: string;
  className?: string;
}

function parseVoiceCommand(transcript: string): ParsedCommand | null {
  const t = transcript.toLowerCase().trim();

  if (t.includes("show all")) return { command: "show_all" };
  if (t.includes("show mask")) return { command: "show_masks" };
  if (t.includes("hide mask")) return { command: "hide_masks" };
  if (t.includes("show label")) return { command: "show_labels" };
  if (t.includes("hide label")) return { command: "hide_labels" };
  if (t === "stop" || t.includes("stop inference") || t.includes("stop stream"))
    return { command: "stop" };

  const showMatch = t.match(/^show\s+(.+)$/);
  if (showMatch) return { command: "show_class", className: showMatch[1] };

  const hideMatch = t.match(/^hide\s+(.+)$/);
  if (hideMatch) return { command: "hide_class", className: hideMatch[1] };

  return null;
}

// ─── Types ──────────────────────────────────────────────────────

interface LogEntry {
  id: number;
  time: string;
  transcript: string;
  confirmation: string;
  success: boolean;
}

// ─── Component ──────────────────────────────────────────────────

export default function VoiceLivePage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [conf, setConf] = useState(0.25);

  const [file, setFile] = useState<File | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [commandLog, setCommandLog] = useState<LogEntry[]>([]);
  const logIdRef = useRef(0);

  const [displayConfig, setDisplayConfig] = useState({
    show_masks: true,
    show_labels: true,
    hidden_classes: [] as string[],
  });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const recognitionRef = useRef<any>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

  // *** Use refs for values that callbacks need current access to ***
  const sessionIdRef = useRef<string | null>(null);
  const streamingRef = useRef(false);
  const listeningRef = useRef(false);

  // Keep refs in sync with state
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);
  useEffect(() => { streamingRef.current = streaming; }, [streaming]);
  useEffect(() => { listeningRef.current = listening; }, [listening]);

  // Load models
  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m);
        const def = m.find((x) => x.name === "26n") || m[0];
        if (def) setSelectedModel(def.name);
      })
      .catch(() => {});
  }, []);

  // ─── Audio playback ─────────────────────────────────────────

  const playTts = useCallback(async (text: string) => {
    try {
      const audioData = await speakTts(text);
      if (audioData.byteLength === 0) throw new Error("empty");

      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContext();
      }
      const ctx = audioCtxRef.current;
      const decoded = await ctx.decodeAudioData(audioData);
      const source = ctx.createBufferSource();
      source.buffer = decoded;
      source.connect(ctx.destination);
      source.start();
    } catch {
      // Fallback: browser speech synthesis
      if ("speechSynthesis" in window) {
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 1.1;
        window.speechSynthesis.speak(u);
      }
    }
  }, []);

  // ─── Command execution (uses refs to avoid stale closures) ──

  const addLogEntry = useCallback(
    (transcript: string, confirmation: string, success: boolean) => {
      logIdRef.current += 1;
      const entry: LogEntry = {
        id: logIdRef.current,
        time: new Date().toLocaleTimeString(),
        transcript,
        confirmation,
        success,
      };
      setCommandLog((prev) => [entry, ...prev].slice(0, 20));
    },
    []
  );

  const executeCommand = useCallback(
    async (raw: string) => {
      const sid = sessionIdRef.current; // ← use ref, not stale state
      const parsed = parseVoiceCommand(raw);

      if (!parsed) {
        addLogEntry(raw, "Command not recognized", false);
        return;
      }
      if (!sid) {
        addLogEntry(raw, "No active session — start inference first", false);
        return;
      }

      try {
        const res = await sendStreamCommand(
          sid,
          parsed.command,
          parsed.className
        );
        setDisplayConfig(res.config);
        addLogEntry(raw, res.confirmation, true);
        playTts(res.confirmation);

        if (parsed.command === "stop") {
          setStreaming(false);
          setSessionId(null);
          doStopListening();
        }
      } catch (err) {
        addLogEntry(raw, `Failed: ${err}`, false);
      }
    },
    [addLogEntry, playTts]
  );

  // ─── Speech Recognition ─────────────────────────────────────

  const startListening = useCallback(() => {
    if (
      !("webkitSpeechRecognition" in window || "SpeechRecognition" in window)
    ) {
      addLogEntry(
        "",
        "Speech recognition not supported — use Chrome",
        false
      );
      return;
    }

    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event: any) => {
      let final = "";
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          final += t;
        } else {
          interim = t;
        }
      }
      setTranscript(interim);
      if (final.trim()) {
        setTranscript("");
        executeCommand(final.trim());
      }
    };

    recognition.onerror = (e: any) => {
      console.error("SpeechRecognition error:", e.error);
      if (e.error === "not-allowed") {
        addLogEntry("", "Microphone access denied", false);
        doStopListening();
        return;
      }
      // Auto-restart on transient errors
      setTimeout(() => {
        if (listeningRef.current && recognitionRef.current) {
          try {
            recognitionRef.current.start();
          } catch {}
        }
      }, 500);
    };

    recognition.onend = () => {
      // Auto-restart if still listening
      if (listeningRef.current) {
        try {
          recognition.start();
        } catch {}
      }
    };

    recognition.start();
    recognitionRef.current = recognition;
    setListening(true);
  }, [addLogEntry, executeCommand]);

  const doStopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.onend = null;
      recognitionRef.current.onerror = null;
      try {
        recognitionRef.current.stop();
      } catch {}
      recognitionRef.current = null;
    }
    setListening(false);
    setTranscript("");
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.onend = null;
        try {
          recognitionRef.current.stop();
        } catch {}
      }
    };
  }, []);

  // ─── Stream handling ────────────────────────────────────────

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setStreaming(false);
    setSessionId(null);
  }, []);

  const startStream = async () => {
    if (!file || !selectedModel) return;

    try {
      // Step 1: Upload video → get session ID (JSON response, no header)
      const { session_id: sid } = await createStreamSession(
        file,
        selectedModel,
        conf
      );
      setSessionId(sid);
      sessionIdRef.current = sid;

      // Step 2: Open MJPEG feed using session ID
      const feedUrl = getStreamFeedUrl(sid);
      const res = await fetch(feedUrl);
      if (!res.ok || !res.body) throw new Error("Stream feed failed");

      setStreaming(true);
      streamingRef.current = true;
      setDisplayConfig({
        show_masks: true,
        show_labels: true,
        hidden_classes: [],
      });
      renderMjpegStream(res.body);
    } catch {
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
        const newBuf = new Uint8Array(buffer.length + value.length);
        newBuf.set(buffer);
        newBuf.set(value, buffer.length);
        buffer = newBuf;
        while (true) {
          const soiIdx = findBytes(buffer, [0xff, 0xd8]);
          if (soiIdx === -1) break;
          const eoiIdx = findBytes(buffer, [0xff, 0xd9], soiIdx + 2);
          if (eoiIdx === -1) break;
          const jpegEnd = eoiIdx + 2;
          const jpeg = buffer.slice(soiIdx, jpegEnd);
          buffer = buffer.slice(jpegEnd);
          const blob = new Blob([jpeg], { type: "image/jpeg" });
          const blobUrl = URL.createObjectURL(blob);
          if (imgRef.current) {
            const prev = imgRef.current.src;
            imgRef.current.src = blobUrl;
            if (prev.startsWith("blob:")) URL.revokeObjectURL(prev);
          }
        }
      }
    } catch {
    } finally {
      reader.releaseLock();
      setStreaming(false);
    }
  };

  const findBytes = (h: Uint8Array, n: number[], s = 0) => {
    for (let i = s; i <= h.length - n.length; i++) {
      if (n.every((b, j) => h[i + j] === b)) return i;
    }
    return -1;
  };

  const handleStop = async () => {
    doStopListening();
    if (sessionIdRef.current) {
      try {
        await stopStream(sessionIdRef.current);
      } catch {}
    }
    setStreaming(false);
    setSessionId(null);
  };

  // ─── Manual command buttons ─────────────────────────────────

  const sendManualCommand = async (
    command: string,
    className?: string,
    label?: string
  ) => {
    const sid = sessionIdRef.current;
    if (!sid) {
      addLogEntry(label || command, "No active session", false);
      return;
    }
    try {
      const res = await sendStreamCommand(sid, command, className);
      setDisplayConfig(res.config);
      addLogEntry(label || command, res.confirmation, true);
      playTts(res.confirmation);
      if (command === "stop") {
        setStreaming(false);
        setSessionId(null);
        doStopListening();
      }
    } catch (err) {
      addLogEntry(label || command, `Failed: ${err}`, false);
    }
  };

  // ─── Render ─────────────────────────────────────────────────

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">
          Voice-Controlled Live Inference
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Upload a video and control inference with voice commands
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-4">
        {/* ── Left panel ── */}
        <div className="space-y-4">
          {/* Video Upload */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Video Input</CardTitle>
            </CardHeader>
            <CardContent>
              <div
                onClick={() => fileInputRef.current?.click()}
                className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border py-5 transition-colors hover:border-primary/50 hover:bg-accent/30"
              >
                {file ? (
                  <>
                    <FileVideo className="mb-2 h-6 w-6 text-primary" />
                    <p className="text-sm font-medium truncate max-w-full px-2">
                      {file.name}
                    </p>
                    <p className="mt-0.5 text-[11px] text-muted-foreground">
                      {(file.size / (1024 * 1024)).toFixed(1)} MB
                    </p>
                  </>
                ) : (
                  <>
                    <Upload className="mb-2 h-6 w-6 text-muted-foreground" />
                    <p className="text-sm font-medium">Upload video</p>
                    <p className="mt-0.5 text-[11px] text-muted-foreground">
                      MP4, AVI, MOV
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

          {/* Settings */}
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
                  Confidence
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
                Start
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

          {/* Display State */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">
                Display State
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex items-center gap-2 text-xs">
                <Layers className="h-3.5 w-3.5" />
                <span>Masks</span>
                {displayConfig.show_masks ? (
                  <Badge
                    variant="outline"
                    className="ml-auto text-[10px] border-green-300 text-green-600"
                  >
                    ON
                  </Badge>
                ) : (
                  <Badge
                    variant="outline"
                    className="ml-auto text-[10px] border-red-300 text-red-500"
                  >
                    OFF
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs">
                <Tag className="h-3.5 w-3.5" />
                <span>Labels</span>
                {displayConfig.show_labels ? (
                  <Badge
                    variant="outline"
                    className="ml-auto text-[10px] border-green-300 text-green-600"
                  >
                    ON
                  </Badge>
                ) : (
                  <Badge
                    variant="outline"
                    className="ml-auto text-[10px] border-red-300 text-red-500"
                  >
                    OFF
                  </Badge>
                )}
              </div>
              {displayConfig.hidden_classes.length > 0 && (
                <div className="mt-1 space-y-1">
                  <span className="text-[10px] text-muted-foreground">
                    Hidden:
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {displayConfig.hidden_classes.map((c) => (
                      <Badge
                        key={c}
                        variant="secondary"
                        className="text-[10px]"
                      >
                        <EyeOff className="mr-1 h-2.5 w-2.5" />
                        {c}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* ── Main panel ── */}
        <div className="xl:col-span-3 space-y-4">
          {/* Stream */}
          <Card className="overflow-hidden">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Video className="h-4 w-4 text-primary" />
                  <CardTitle className="text-sm">Inference Output</CardTitle>
                </div>
                <div className="flex items-center gap-2">
                  {streaming && (
                    <Badge className="bg-red-500 text-white animate-pulse">
                      ● PROCESSING
                    </Badge>
                  )}
                  {listening && (
                    <Badge className="bg-primary text-white animate-pulse">
                      <Mic className="mr-1 h-3 w-3" /> LISTENING
                    </Badge>
                  )}
                </div>
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
                    Upload a video and click &ldquo;Start&rdquo;
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Voice Control Panel */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Mic className="h-4 w-4 text-primary" />
                  <CardTitle className="text-sm">Voice Commands</CardTitle>
                </div>
                <div className="flex items-center gap-2">
                  {!listening ? (
                    <Button
                      size="sm"
                      onClick={startListening}
                      disabled={!streaming}
                    >
                      <Mic className="mr-1.5 h-3.5 w-3.5" />
                      Start Listening
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={doStopListening}
                    >
                      <MicOff className="mr-1.5 h-3.5 w-3.5" />
                      Stop Listening
                    </Button>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {/* Interim transcript */}
              {transcript && (
                <div className="mb-3 flex items-center gap-2 rounded-lg bg-primary/5 px-3 py-2 text-sm">
                  <Mic className="h-3.5 w-3.5 animate-pulse text-primary" />
                  <span className="italic text-muted-foreground">
                    {transcript}
                  </span>
                </div>
              )}

              {/* ── Manual test buttons ── */}
              <div className="mb-4 rounded-lg border bg-muted/20 p-3">
                <p className="mb-2 text-xs font-medium text-muted-foreground">
                  Quick Commands (click to test)
                </p>
                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-7 text-xs"
                    disabled={!streaming}
                    onClick={() =>
                      sendManualCommand("show_masks", undefined, "show masks")
                    }
                  >
                    <Eye className="mr-1 h-3 w-3" /> Show Masks
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-7 text-xs"
                    disabled={!streaming}
                    onClick={() =>
                      sendManualCommand("hide_masks", undefined, "hide masks")
                    }
                  >
                    <EyeOff className="mr-1 h-3 w-3" /> Hide Masks
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-7 text-xs"
                    disabled={!streaming}
                    onClick={() =>
                      sendManualCommand(
                        "show_labels",
                        undefined,
                        "show labels"
                      )
                    }
                  >
                    <Tag className="mr-1 h-3 w-3" /> Show Labels
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-7 text-xs"
                    disabled={!streaming}
                    onClick={() =>
                      sendManualCommand(
                        "hide_labels",
                        undefined,
                        "hide labels"
                      )
                    }
                  >
                    <Tag className="mr-1 h-3 w-3 opacity-50" /> Hide Labels
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-7 text-xs"
                    disabled={!streaming}
                    onClick={() =>
                      sendManualCommand("show_all", undefined, "show all")
                    }
                  >
                    <Layers className="mr-1 h-3 w-3" /> Show All
                  </Button>
                </div>
                {/* Class-specific buttons */}
                <p className="mt-3 mb-1.5 text-[10px] font-medium text-muted-foreground">
                  Toggle Classes
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {[
                    "artery",
                    "vein",
                    "instrument",
                    "nerve",
                    "ovary",
                    "ureter",
                    "uterine",
                    "uterus",
                  ].map((cls) => (
                    <div key={cls} className="flex gap-0.5">
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-6 rounded-r-none text-[10px] px-2"
                        disabled={!streaming}
                        onClick={() =>
                          sendManualCommand(
                            "show_class",
                            cls,
                            `show ${cls}`
                          )
                        }
                      >
                        <Eye className="mr-0.5 h-2.5 w-2.5" />
                        {cls}
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-6 rounded-l-none text-[10px] px-1.5 text-red-500 hover:text-red-600"
                        disabled={!streaming}
                        onClick={() =>
                          sendManualCommand(
                            "hide_class",
                            cls,
                            `hide ${cls}`
                          )
                        }
                      >
                        <EyeOff className="h-2.5 w-2.5" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Command Log */}
              <div>
                <p className="mb-2 text-xs font-medium text-muted-foreground">
                  Command Log
                </p>
                <div className="max-h-48 space-y-1.5 overflow-y-auto">
                  {commandLog.length === 0 ? (
                    <p className="py-4 text-center text-xs text-muted-foreground">
                      {streaming
                        ? "Use voice or click buttons above to send commands"
                        : "Start inference first, then use voice commands"}
                    </p>
                  ) : (
                    commandLog.map((entry) => (
                      <div
                        key={entry.id}
                        className={`flex items-start gap-2 rounded-md px-3 py-2 text-xs ${
                          entry.success
                            ? "bg-green-50 border border-green-100"
                            : "bg-red-50 border border-red-100"
                        }`}
                      >
                        <MessageSquare className="mt-0.5 h-3 w-3 shrink-0 text-muted-foreground" />
                        <div className="min-w-0 flex-1">
                          <p className="font-medium truncate">
                            &ldquo;{entry.transcript}&rdquo;
                          </p>
                          <p className="text-muted-foreground">
                            → {entry.confirmation}
                          </p>
                        </div>
                        <span className="shrink-0 text-[10px] text-muted-foreground">
                          {entry.time}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
