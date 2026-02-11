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
import { Separator } from "@/components/ui/separator";
import { Cpu, Loader2, Eye, HardDrive, Tag } from "lucide-react";
import {
  listModels,
  getModelInfo,
  type ModelInfo,
} from "@/lib/api-client";

interface DetailedModel extends ModelInfo {
  class_names?: Record<number, string>;
  task?: string;
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailedModel, setDetailedModel] = useState<DetailedModel | null>(
    null
  );
  const [loadingDetail, setLoadingDetail] = useState(false);

  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const viewModelDetail = async (name: string) => {
    setLoadingDetail(true);
    try {
      const info = await getModelInfo(name);
      setDetailedModel(info);
    } catch {
      setDetailedModel(null);
    } finally {
      setLoadingDetail(false);
    }
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">Models</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          View and inspect available YOLO models
        </p>
      </div>

      {loading ? (
        <div className="flex h-48 items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          {/* Model List */}
          <div className="xl:col-span-2">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {models.map((m) => (
                <Card
                  key={m.name}
                  className={`cursor-pointer transition-shadow hover:shadow-md ${
                    detailedModel?.name === m.name
                      ? "ring-2 ring-primary"
                      : ""
                  }`}
                  onClick={() => viewModelDetail(m.name)}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary/10">
                          <Cpu className="h-4 w-4 text-primary" />
                        </div>
                        <CardTitle className="text-base">{m.name}</CardTitle>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {m.size_mb} MB
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                      <HardDrive className="h-3 w-3" />
                      <span className="truncate font-mono">
                        {m.directory.split("/").slice(-2).join("/")}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          {/* Model Detail Panel */}
          <div>
            <Card className="sticky top-8">
              <CardHeader>
                <CardTitle className="text-sm">Model Details</CardTitle>
                <CardDescription className="text-xs">
                  Click a model card to inspect
                </CardDescription>
              </CardHeader>
              <CardContent>
                {loadingDetail ? (
                  <div className="flex h-32 items-center justify-center">
                    <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                  </div>
                ) : detailedModel ? (
                  <div className="space-y-4">
                    <div>
                      <p className="text-lg font-semibold">
                        {detailedModel.name}
                      </p>
                      {detailedModel.task && (
                        <Badge variant="secondary" className="mt-1">
                          {detailedModel.task}
                        </Badge>
                      )}
                    </div>

                    <Separator />

                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Size</span>
                        <span className="font-medium">
                          {detailedModel.size_mb} MB
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Path</span>
                        <span className="max-w-[160px] truncate font-mono text-xs">
                          {detailedModel.path.split("/").pop()}
                        </span>
                      </div>
                    </div>

                    {detailedModel.class_names && (
                      <>
                        <Separator />
                        <div>
                          <p className="mb-2 flex items-center gap-1 text-xs font-medium text-muted-foreground">
                            <Tag className="h-3 w-3" />
                            Classes (
                            {Object.keys(detailedModel.class_names).length})
                          </p>
                          <div className="flex flex-wrap gap-1.5">
                            {Object.entries(detailedModel.class_names).map(
                              ([id, name]) => (
                                <Badge
                                  key={id}
                                  variant="outline"
                                  className="text-[11px]"
                                >
                                  {name}
                                </Badge>
                              )
                            )}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="flex h-32 flex-col items-center justify-center text-muted-foreground">
                    <Eye className="mb-2 h-6 w-6 opacity-40" />
                    <p className="text-xs">Select a model to view details</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
