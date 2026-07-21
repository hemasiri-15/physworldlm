"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Box,
  Link2,
  Waypoints,
  Clock,
  ShieldCheck,
  Route,
  Sparkles,
  Film,
  CheckCircle2,
  CircleDashed,
  LoaderCircle,
} from "lucide-react";

const PIPELINE = [
  { key: "entity", label: "EntityEncoder", icon: Box },
  { key: "relation", label: "RelationEncoder", icon: Link2 },
  { key: "graph", label: "GraphBuilder", icon: Waypoints },
  { key: "temporal", label: "TemporalWorldModel", icon: Clock },
  { key: "state", label: "StateEngine", icon: ShieldCheck },
  { key: "trajectory", label: "TrajectoryEngine", icon: Route },
  { key: "physics", label: "PhysX Backend", icon: Sparkles },
  { key: "render", label: "Renderer", icon: Film },
] as const;

type StageStatus = "pending" | "active" | "done";

interface Metrics {
  objects: number;
  constraints: number;
  materials: number;
  environment: number;
  readiness: number;
}

const IDLE_METRICS: Metrics = {
  objects: 0,
  constraints: 0,
  materials: 0,
  environment: 0,
  readiness: 0,
};

const TARGET_METRICS: Metrics = {
  objects: 14,
  constraints: 26,
  materials: 8,
  environment: 1,
  readiness: 100,
};

function useCountUp(target: number, run: boolean, duration = 900) {
  const [value, setValue] = useState(0);
  const frame = useRef<number>();

  useEffect(() => {
    if (!run) {
      setValue(0);
      return;
    }
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(Math.round(eased * target));
      if (t < 1) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
    return () => {
      if (frame.current) cancelAnimationFrame(frame.current);
    };
  }, [run, target, duration]);

  return value;
}

interface WorldPreviewProps {
  isGenerating: boolean;
  hasResult: boolean;
}

export default function WorldPreview({
  isGenerating,
  hasResult,
}: WorldPreviewProps) {
  const [activeIndex, setActiveIndex] = useState(-1);

  useEffect(() => {
    if (!isGenerating) {
      setActiveIndex(-1);
      return;
    }
    let i = 0;
    setActiveIndex(0);
    const interval = setInterval(() => {
      i += 1;
      if (i >= PIPELINE.length) {
        clearInterval(interval);
        return;
      }
      setActiveIndex(i);
    }, 260);
    return () => clearInterval(interval);
  }, [isGenerating]);

  const stageStatus = (index: number): StageStatus => {
    if (hasResult) return "done";
    if (!isGenerating) return "pending";
    if (index < activeIndex) return "done";
    if (index === activeIndex) return "active";
    return "pending";
  };

  const metricsRun = isGenerating || hasResult;
  const objects = useCountUp(TARGET_METRICS.objects, metricsRun);
  const constraints = useCountUp(TARGET_METRICS.constraints, metricsRun);
  const materials = useCountUp(TARGET_METRICS.materials, metricsRun);
  const environment = useCountUp(TARGET_METRICS.environment, metricsRun);
  const readiness = useCountUp(TARGET_METRICS.readiness, metricsRun);

  const metricEntries = [
    { label: "Objects", value: objects },
    { label: "Constraints", value: constraints },
    { label: "Materials", value: materials },
    { label: "Environment", value: environment },
  ];

  return (
    <div className="glass-panel flex h-full flex-col rounded-2xl">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/[0.06] px-5 py-4">
        <div className="flex items-center gap-2">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              hasResult
                ? "bg-emerald-400"
                : isGenerating
                  ? "animate-pulse bg-blue-400"
                  : "bg-zinc-600"
            }`}
          />
          <span className="font-mono-data text-xs uppercase tracking-wider text-zinc-500">
            world_preview
          </span>
        </div>
        <span className="font-mono-data text-[11px] text-zinc-600">
          {hasResult ? "compiled" : isGenerating ? "compiling…" : "idle"}
        </span>
      </div>

      {/* Metrics strip */}
      <div className="grid grid-cols-2 gap-px border-b border-white/[0.06] bg-white/[0.03] sm:grid-cols-4">
        {metricEntries.map((m) => (
          <div key={m.label} className="bg-[#0b0c0f] px-4 py-3">
            <div className="font-mono-data text-xl font-medium tabular-nums text-zinc-100">
              {m.value.toString().padStart(2, "0")}
            </div>
            <div className="mt-0.5 text-[11px] text-zinc-500">{m.label}</div>
          </div>
        ))}
      </div>

      {/* Pipeline */}
      <div className="flex-1 space-y-1 overflow-y-auto p-4">
        {PIPELINE.map((stage, i) => {
          const status = stageStatus(i);
          return (
            <div
              key={stage.key}
              className={`flex items-center gap-3 rounded-lg border px-3 py-2.5 transition-all duration-300 ${
                status === "active"
                  ? "border-blue-500/30 bg-blue-500/[0.06]"
                  : status === "done"
                    ? "border-white/[0.06] bg-white/[0.02]"
                    : "border-transparent"
              }`}
            >
              <stage.icon
                className={`h-3.5 w-3.5 shrink-0 ${
                  status === "done"
                    ? "text-emerald-400"
                    : status === "active"
                      ? "text-blue-400"
                      : "text-zinc-600"
                }`}
                strokeWidth={2}
              />
              <span
                className={`font-mono-data text-[13px] ${
                  status === "pending" ? "text-zinc-600" : "text-zinc-200"
                }`}
              >
                {stage.label}
              </span>
              <span className="ml-auto">
                {status === "done" && (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                )}
                {status === "active" && (
                  <LoaderCircle className="h-3.5 w-3.5 animate-spin text-blue-400" />
                )}
                {status === "pending" && (
                  <CircleDashed className="h-3.5 w-3.5 text-zinc-700" />
                )}
              </span>
            </div>
          );
        })}
      </div>

      {/* Simulation readiness footer */}
      <div className="border-t border-white/[0.06] px-5 py-4">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-[11px] text-zinc-500">
            Simulation readiness
          </span>
          <span className="font-mono-data text-[11px] text-zinc-400">
            {readiness}%
          </span>
        </div>
        <div className="h-1 overflow-hidden rounded-full bg-white/[0.06]">
          <motion.div
            className="h-full rounded-full bg-gradient-to-r from-blue-500 via-violet-500 to-cyan-400"
            initial={{ width: 0 }}
            animate={{ width: `${readiness}%` }}
            transition={{ ease: "easeOut" }}
          />
        </div>
      </div>
    </div>
  );
}
