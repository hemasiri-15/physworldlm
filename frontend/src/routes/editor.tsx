import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { generateWorld } from "../lib/api";
import {
  Sparkles,
  Github,
  BookOpen,
  Info,
  Play,
  Activity,
  Box,
  GitBranch,
  Layers,
  Cpu,
  Globe,
  Waves,
  Circle,
  ArrowRight,
  Copy,
  Download,
  Maximize2,
  Minimize2,
  ImagePlus,
  History,
  X,
  Check,
  Command,
  Network,
  MonitorPlay,
  Boxes,
  Mountain,
  CloudSun,
  Package,
  Target as TargetIcon,
} from "lucide-react";

export const Route = createFileRoute("/editor")({
  head: () => ({
    meta: [
      { title: "PhysWorldLM — Physics-Aware Conversational World Editor" },
      {
        name: "description",
        content:
          "Transform natural language into structured WorldSpecs, OpenUSD scenes, planners, and NVIDIA Omniverse environments.",
      },
    ],
  }),
  component: Editor,
});

/* ────────────────────────────── constants ────────────────────────────── */

const PIPELINE = [
  { id: "nl", name: "Natural Language" },
  { id: "llm", name: "LLM Parsing" },
  { id: "ontology", name: "Ontology Construction" },
  { id: "worldspec", name: "WorldSpec Generation" },
  { id: "graph", name: "Scene Graph Construction" },
  { id: "compiler", name: "Scene Compiler" },
  { id: "planner", name: "Planner" },
  { id: "usd", name: "OpenUSD" },
  { id: "omniverse", name: "Omniverse" },
] as const;

const STATUS_ROWS = [
  { name: "LLM", state: "Ready", tone: "ok" },
  { name: "Ontology", state: "Waiting", tone: "wait" },
  { name: "WorldSpec", state: "Waiting", tone: "wait" },
  { name: "Planner", state: "Idle", tone: "idle" },
  { name: "USD", state: "Idle", tone: "idle" },
  { name: "Omniverse", state: "Disconnected", tone: "off" },
] as const;

const EXAMPLE_CHIPS = [
  {
    label: "Air Combat",
    prompt:
      "Generate two F-16 aircraft intercepting a hostile bomber over mountainous terrain while avoiding enemy radar.",
  },
  {
    label: "Urban Traffic",
    prompt:
      "Simulate rush-hour traffic across a 4km² downtown grid with 2,400 vehicles, 180 traffic lights, and emergency response corridors.",
  },
  {
    label: "Warehouse Robot",
    prompt:
      "Deploy an autonomous pick-and-place robot across a 3-aisle fulfilment warehouse with 12,000 SKUs and human workers.",
  },
  {
    label: "Disaster Response",
    prompt:
      "Coordinate search-and-rescue drones through a flooded coastal city after a category-4 hurricane makes landfall.",
  },
  {
    label: "Drone Swarm",
    prompt:
      "Command a 64-drone swarm performing distributed area coverage over 20km² of forest with contested GPS.",
  },
  {
    label: "Satellite Mission",
    prompt:
      "Model a 6-satellite LEO constellation tasked with continuous imaging of the Arctic circle at 30cm resolution.",
  },
];

const RECENT_PROMPTS = [
  "F-16 pair intercepting hostile bomber over alpine terrain",
  "Autonomous convoy through contested desert corridor",
  "Mars rover traversing Jezero crater rim",
];

const TABS = [
  { id: "mission", label: "Mission", icon: TargetIcon },
  { id: "physics", label: "Physics", icon: Cpu },
  { id: "terrain", label: "Terrain", icon: Mountain },
  { id: "weather", label: "Weather", icon: CloudSun },
  { id: "assets", label: "Assets", icon: Package },
] as const;

type TabId = (typeof TABS)[number]["id"];

const MODULES = [
  { name: "Ontology", icon: Layers, desc: "Typed concept graph & domain axioms" },
  { name: "Scene Graph", icon: GitBranch, desc: "Hierarchical entity & transform tree" },
  { name: "Compiler", icon: Cpu, desc: "Lowers WorldSpec → executable stage" },
  { name: "Planner", icon: Network, desc: "Goal-conditioned task decomposition" },
  { name: "OpenUSD", icon: Box, desc: "Simulation-ready stage export" },
  { name: "Simulation", icon: Waves, desc: "Omniverse physics runtime" },
];

const FUTURE_INTEGRATIONS = [
  { name: "NVIDIA Omniverse", tag: "runtime" },
  { name: "Isaac Sim", tag: "robotics" },
  { name: "PhysX", tag: "physics" },
  { name: "Cesium", tag: "geospatial" },
  { name: "Planner", tag: "reasoning" },
  { name: "OpenUSD", tag: "scene" },
];

const SAMPLE_SPEC = `{
  "worldspec_version": "0.4.2",
  "scenario": {
    "id": "wsp_a1f2e9",
    "name": "F-16 Intercept — Alpine Corridor",
    "domain": "aerospace.combat",
    "duration_s": 480
  },
  "environment": {
    "terrain": "mountainous.alpine",
    "elevation_range_m": [1200, 3900],
    "atmosphere": { "wind_ms": 8.4, "visibility_km": 22 },
    "time_of_day": "07:14 UTC"
  },
  "agents": [
    {
      "id": "blue_01",
      "class": "aircraft.f16c",
      "role": "interceptor",
      "spawn": { "lat": 46.71, "lon": 8.92, "alt_m": 8500 }
    },
    {
      "id": "blue_02",
      "class": "aircraft.f16c",
      "role": "interceptor.wing",
      "formation": "combat_spread"
    },
    {
      "id": "red_01",
      "class": "aircraft.bomber.hostile",
      "role": "target",
      "heading_deg": 274
    }
  ],
  "threats": [
    { "id": "sam_ne_01", "class": "radar.sam", "detect_km": 45 }
  ],
  "objectives": [
    "intercept(red_01) before boundary(alpha)",
    "minimize(exposure(sam_ne_01))"
  ]
}`;

/* ────────────────────────────── page ────────────────────────────── */

type StageStatus = "done" | "active" | "pending";

function Editor() {
  const [prompt, setPrompt] = useState(EXAMPLE_CHIPS[0].prompt);
  const [worldSpec, setWorldSpec] = useState<any>(null);
  const [tab, setTab] = useState<TabId>("mission");
  const [activeStage, setActiveStage] = useState(5); // Scene Compiler by default
  const [generating, setGenerating] = useState(false);
  const [completed, setCompleted] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Ctrl+Enter shortcut.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        setGenerating(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const handleGenerate = async () => {
      console.log("Generate button clicked!");

      setCompleted(false);
      setGenerating(true);

      // Reset pipeline
      setActiveStage(0);

      try {
          // Natural Language
          setActiveStage(1);

          const result = await generateWorld(prompt);

          console.log(result);

          // Backend finished successfully
          setWorldSpec(result);

          // Complete all stages
          setActiveStage(PIPELINE.length - 1);

          setCompleted(true);

          setTimeout(() => {
              setCompleted(false);
          }, 3000);

      } catch (err) {
          console.error("Generation failed:", err);

          alert(
              "Failed to generate WorldSpec.\n\nMake sure the FastAPI server is running."
          );

          setActiveStage(0);

      } finally {
          setGenerating(false);
      }
  };

  const stageStatuses: StageStatus[] = useMemo(
    () =>
    PIPELINE.map((_, i) =>
        i < activeStage ? "done" : i === activeStage ? "active" : "pending",
      ),
    [activeStage],
  );

  return (
    <div className="min-h-screen text-foreground">
      <Nav />
      <main className="relative mx-auto max-w-[1400px] px-4 pb-24 md:px-6">
        <Hero />
        <Workspace
          prompt={prompt}
          setPrompt={setPrompt}
          tab={tab}
          setTab={setTab}
          generating={generating}
          completed={completed}
          onGenerate={handleGenerate}	
          stageStatuses={stageStatuses}
          textareaRef={textareaRef}
        />
        <SpecViewer worldSpec={worldSpec} />
      </main>
      <Footer />
      <SuccessToast show={completed} />
    </div>
  );
}

/* ────────────────────────────── chrome ────────────────────────────── */

function BackgroundFX() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute inset-0 bg-grid opacity-40 [mask-image:radial-gradient(ellipse_at_center,black_20%,transparent_70%)]" />
      <div className="absolute -top-40 left-1/2 h-[600px] w-[900px] -translate-x-1/2 rounded-full bg-[radial-gradient(closest-side,oklch(0.62_0.25_295/0.35),transparent)] blur-3xl" />
      <div className="absolute top-1/3 -right-40 h-[500px] w-[500px] rounded-full bg-[radial-gradient(closest-side,oklch(0.68_0.2_250/0.25),transparent)] blur-3xl" />
    </div>
  );
}

function Nav() {
  return (
    <header
      className="sticky top-0 z-40 bg-black text-white"
    >
      <div className="mx-auto flex max-w-[1400px] items-center justify-between px-4 py-4 md:px-6">
        <a href="/" className="flex items-center gap-2.5 text-white">
          <div
            className="relative flex h-8 w-8 items-center justify-center rounded-lg"
            style={{ background: "var(--gradient-primary)" }}
          >
            <Globe className="h-4 w-4 text-white" />
            <div
              className="absolute inset-0 rounded-lg blur-md opacity-60"
              style={{ background: "var(--gradient-primary)" }}
            />
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="font-display text-[17px] font-semibold tracking-tight">
              PhysWorldLM
            </span>
            <span className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-[10px] text-gray-300">
              v0.4
            </span>
          </div>
        </a>
        <nav className="hidden items-center gap-1 md:flex">
          <NavLink icon={<BookOpen className="h-3.5 w-3.5" />}>Documentation</NavLink>
          <NavLink icon={<Github className="h-3.5 w-3.5" />}>GitHub</NavLink>
          <NavLink icon={<Info className="h-3.5 w-3.5" />}>About</NavLink>
        </nav>
        <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-gray-300 md:flex">
          <span className="relative flex h-1.5 w-1.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400" />
          </span>
          <span className="font-mono">runtime online</span>
        </div>
      </div>
    </header>
  );
}

function NavLink({
  children,
  icon,
}: {
  children: React.ReactNode;
  icon: React.ReactNode;
}) {
  return (
    <a
      href="#"
      className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm text-gray-300 transition-colors hover:bg-white/10 hover:text-white"
    >
      {icon}
      {children}
    </a>
  );
}

function Hero() {
  return (
    <section className="relative pt-16 pb-10 text-center md:pt-20 md:pb-14">
      <h1 className="mx-auto max-w-4xl text-balance text-4xl font-semibold leading-[1.05] tracking-tight md:text-[68px]">
        <span className="text-black">Generate Physics-Aware</span>
        <br />
        <span className="text-black">Virtual Worlds</span>
      </h1>
      <p className="mx-auto mt-6 max-w-2xl text-balance text-base leading-relaxed text-muted-foreground md:text-lg">
        Transform natural language into structured{" "}
        <span className="font-mono text-foreground">WorldSpecs</span>, simulation-ready
        OpenUSD scenes, planners, and NVIDIA Omniverse environments.
      </p>
      <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
        <a
          href="#workspace"
          className="group inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium text-white transition-all hover:scale-[1.02]"
          style={{ background: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
        >
          Launch Workspace
          <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
        </a>
        <a
          href="#pipeline"
          className="inline-flex items-center gap-2 rounded-full border border-black/10 bg-black/[0.03] px-5 py-2.5 text-sm font-medium text-foreground backdrop-blur-md transition-colors hover:bg-black/[0.06]"
        >
          <Network className="h-4 w-4" />
          View Pipeline
        </a>
      </div>
    </section>
  );
}

/* ────────────────────────────── workspace ────────────────────────────── */

function Workspace({
  prompt,
  setPrompt,
  tab,
  setTab,
  generating,
  completed,
  onGenerate,
  stageStatuses,
  textareaRef,
}: {
  prompt: string;
  setPrompt: (s: string) => void;
  tab: TabId;
  setTab: (t: TabId) => void;
  generating: boolean;
  completed: boolean;
  onGenerate: () => void;
  stageStatuses: StageStatus[];
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
}) {
  return (
    <div className="mt-8 flex flex-col gap-5">
      <section id="workspace" className="grid gap-5 lg:grid-cols-[1.55fr_1fr]">
        <div className="flex flex-col gap-4">
          <PromptEditor
            prompt={prompt}
            setPrompt={setPrompt}
            tab={tab}
            setTab={setTab}
            generating={generating}
            onGenerate={onGenerate}
            textareaRef={textareaRef}
          />
          <RecentPrompts onPick={(p) => setPrompt(p)} />
          <ExampleChips onPick={(p) => setPrompt(p)} />
        </div>
        <div className="flex flex-col gap-5">
          <PipelineStatusCard />
          <PipelineProgressCard
            stageStatuses={stageStatuses}
            generating={generating}
            completed={completed}
          />
        </div>
      </section>
      <WorldPreviewCard />
    </div>
  );
}

function ExampleChips({ onPick }: { onPick: (p: string) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="mr-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        examples
      </span>
      {EXAMPLE_CHIPS.map((c) => (
        <button
          key={c.label}
          onClick={() => onPick(c.prompt)}
          className="group rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs text-muted-foreground backdrop-blur-md transition-all hover:border-[color:var(--violet-glow)]/40 hover:bg-white/[0.06] hover:text-foreground"
        >
          {c.label}
        </button>
      ))}
    </div>
  );
}

function PromptEditor({
  prompt,
  setPrompt,
  tab,
  setTab,
  generating,
  onGenerate,
  textareaRef,
}: {
  prompt: string;
  setPrompt: (s: string) => void;
  tab: TabId;
  setTab: (t: TabId) => void;
  generating: boolean;
  onGenerate: () => void;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
}) {
  const [dragOver, setDragOver] = useState(false);

  return (
    <div
      className={`glass-strong gradient-border relative overflow-hidden rounded-2xl transition-shadow ${
        generating ? "glow-primary" : ""
      }`}
    >
      {/* Tabs */}
      <div className="flex items-center justify-between border-b border-white/5 px-3 pt-3">
        <div className="flex items-center gap-1 overflow-x-auto">
          {TABS.map(({ id, label, icon: Icon }) => {
            const active = tab === id;
            return (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={`relative flex items-center gap-1.5 rounded-t-lg px-3 py-2 text-xs transition-colors ${
                  active
                    ? "text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
                {active && (
                  <span
                    className="absolute inset-x-2 -bottom-px h-px"
                    style={{ background: "var(--gradient-primary)" }}
                  />
                )}
              </button>
            );
          })}
        </div>
        <div className="hidden items-center gap-2 pr-2 font-mono text-[11px] text-muted-foreground md:flex">
          <Cpu className="h-3 w-3" />
          <span>physworld-large · 8B</span>
        </div>
      </div>

      <div className="p-4 md:p-5">
        {tab === "mission" ? (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
            }}
            className={`relative rounded-xl border ${
              dragOver
                ? "border-[color:var(--violet-glow)]/50 bg-white/[0.04]"
                : "border-white/5 bg-black/30"
            } transition-colors`}
          >
            <textarea
              ref={textareaRef}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the world you want to build…"
              className="min-h-[200px] w-full resize-none bg-transparent p-5 pr-5 font-sans text-[15px] leading-relaxed text-foreground placeholder:text-muted-foreground/70 focus:outline-none"
            />
          </div>
        ) : (
          <AdvancedPanel tab={tab} />
        )}

        <div className="mt-4 flex flex-col items-stretch gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap items-center gap-3 font-mono text-[11px] text-muted-foreground">
            <span>{prompt.length} chars</span>
            <span className="opacity-40">·</span>
            <span>~{Math.round(prompt.length / 4)} tokens</span>
            <span className="opacity-40">·</span>
            <span className="text-[color:var(--cyan-glow)]">seed 42</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={onGenerate}
              disabled={generating}
              className="inline-flex items-center gap-1.5 rounded-xl border border-black/10 bg-black/5 px-4 py-3 text-sm font-medium text-foreground transition-colors hover:bg-black/10 disabled:opacity-50"
            >
              <Play className="h-4 w-4" />
              {generating ? "Composing…" : "Generate World"}
            </button>
            <button
              className="group relative inline-flex items-center gap-2 overflow-hidden rounded-xl px-5 py-3 text-sm font-semibold text-white transition-all hover:scale-[1.02]"
              style={{
                background: "var(--gradient-primary)",
                boxShadow: "var(--shadow-glow)",
              }}
            >
              <Network className="h-4 w-4 fill-white" />
              Connect Omniverse
              <span className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/25 to-transparent transition-transform duration-700 group-hover:translate-x-full" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function AdvancedPanel({ tab }: { tab: TabId }) {
  const CONFIG: Record<Exclude<TabId, "mission">, { k: string; v: string }[]> = {
    physics: [
      { k: "engine", v: "PhysX 5.4" },
      { k: "timestep", v: "1/240 s" },
      { k: "gravity", v: "9.81 m/s²" },
      { k: "solver", v: "TGS · 8 iters" },
    ],
    terrain: [
      { k: "biome", v: "mountainous.alpine" },
      { k: "elevation", v: "1,200 – 3,900 m" },
      { k: "resolution", v: "5 m / px" },
      { k: "source", v: "SRTM + Cesium" },
    ],
    weather: [
      { k: "time", v: "07:14 UTC" },
      { k: "wind", v: "8.4 m/s · 274°" },
      { k: "visibility", v: "22 km" },
      { k: "cloud cover", v: "12%" },
    ],
    assets: [
      { k: "aircraft", v: "F-16C · x2" },
      { k: "hostile", v: "bomber · x1" },
      { k: "threats", v: "SAM radar · x1" },
      { k: "props", v: "0" },
    ],
  };
  const rows = CONFIG[tab as Exclude<TabId, "mission">];
  return (
    <div className="rounded-xl border border-black/10 bg-black/5 p-2">
      <div className="grid gap-px overflow-hidden rounded-lg bg-black/10 sm:grid-cols-2">
        {rows.map((r) => (
          <div
            key={r.k}
            className="flex items-center justify-between bg-white px-4 py-3 font-mono text-xs"
          >
            <span className="text-muted-foreground">{r.k}</span>
            <span className="text-foreground">{r.v}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 px-1 font-mono text-[10px] text-muted-foreground">
        advanced parameters override matching fields in the WorldSpec.
      </div>
    </div>
  );
}

function RecentPrompts({ onPick }: { onPick: (p: string) => void }) {
  return (
    <div className="glass rounded-2xl p-4">
      <div className="mb-2 flex items-center gap-2">
        <History className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          recent prompts
        </span>
      </div>
      <ul className="divide-y divide-white/5">
        {RECENT_PROMPTS.map((p) => (
          <li key={p}>
            <button
              onClick={() => onPick(p)}
              className="group flex w-full items-center justify-between gap-3 py-2 text-left text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              <span className="truncate">{p}</span>
              <ArrowRight className="h-3.5 w-3.5 shrink-0 opacity-0 transition-opacity group-hover:opacity-100" />
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ────────────────────────────── right column ────────────────────────────── */

function PipelineStatusCard() {
  return (
    <div className="glass rounded-2xl p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-[color:var(--cyan-glow)]" />
          <h3 className="text-sm font-semibold">Pipeline Status</h3>
        </div>
        <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-emerald-300">
          nominal
        </span>
      </div>
      <ul className="divide-y divide-white/5 font-mono text-xs">
        {STATUS_ROWS.map((r) => (
          <li key={r.name} className="flex items-center justify-between py-2">
            <span className="text-muted-foreground">{r.name}</span>
            <StatusBadge tone={r.tone} label={r.state} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function StatusBadge({ tone, label }: { tone: string; label: string }) {
  const map: Record<string, string> = {
    ok: "border-emerald-400/30 bg-emerald-400/10 text-emerald-300",
    wait: "border-amber-300/30 bg-amber-300/10 text-amber-200",
    idle: "border-white/10 bg-white/5 text-muted-foreground",
    off: "border-red-400/30 bg-red-400/10 text-red-300",
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider ${map[tone]}`}
    >
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ background: "currentColor" }}
      />
      {label}
    </span>
  );
}

function WorldPreviewCard() {
  return (
    <div className="glass relative overflow-hidden rounded-2xl">
      <div className="flex items-center justify-between p-5 pb-3">
        <div className="flex items-center gap-2">
          <MonitorPlay className="h-4 w-4 text-[color:var(--violet-glow)]" />
          <h3 className="text-sm font-semibold">World Preview</h3>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          openusd · viewport
        </span>
      </div>
      <div
        className="relative mx-5 mb-5 flex h-44 items-center justify-center overflow-hidden rounded-xl border border-white/5"
        style={{
          background:
            "radial-gradient(ellipse at 30% 20%, oklch(0.62 0.25 295 / 0.25), transparent 60%), radial-gradient(ellipse at 80% 90%, oklch(0.68 0.2 250 / 0.22), transparent 60%), #0a0a12",
        }}
      >
        <div className="absolute inset-0 bg-grid opacity-25" />
        <div className="relative z-10 flex flex-col items-center gap-2 text-center">
          <div className="relative flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-white/[0.03] backdrop-blur-md">
            <Boxes className="h-5 w-5 text-[color:var(--violet-glow)]" />
            <div
              className="absolute inset-0 rounded-xl blur-lg opacity-40"
              style={{ background: "var(--gradient-primary)" }}
            />
          </div>
          <div className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
            OpenUSD Viewport
          </div>
          <div className="flex items-center gap-2 font-mono text-[11px] text-foreground">
            <span className="relative flex h-1.5 w-1.5">
              <span
                className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75"
                style={{ background: "var(--cyan-glow)" }}
              />
              <span
                className="relative inline-flex h-1.5 w-1.5 rounded-full"
                style={{ background: "var(--cyan-glow)" }}
              />
            </span>
            Loading Scene…
          </div>
        </div>
        <div className="absolute left-3 top-3 font-mono text-[10px] text-muted-foreground">
          <div>SCENE · alpine_intercept</div>
          <div className="text-[color:var(--cyan-glow)]">▲ awaiting stage</div>
        </div>
        <div className="absolute bottom-3 right-3 font-mono text-[10px] text-muted-foreground">
          omniverse · disconnected
        </div>
      </div>
    </div>
  );
}

function PipelineProgressCard({
  stageStatuses,
  generating,
  completed,
}: {
  stageStatuses: StageStatus[];
  generating: boolean;
  completed: boolean;
}) {
  const doneCount = stageStatuses.filter((s) => s === "done").length;
  const pct = completed
  ? 100
  : Math.round(
      ((doneCount + (generating ? 0.5 : 0)) / PIPELINE.length) * 100
    );
  return (
    <div id="pipeline" className="glass rounded-2xl p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Waves className="h-4 w-4 text-[color:var(--blue-glow)]" />
          <h3 className="text-sm font-semibold">Pipeline Progress</h3>
        </div>
        <span className="font-mono text-[11px] text-muted-foreground">{pct}%</span>
      </div>
      <div className="relative mb-4 h-1.5 overflow-hidden rounded-full bg-white/5">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-500"
          style={{
            width: `${pct}%`,
            background: "var(--gradient-primary)",
            boxShadow: "0 0 12px oklch(0.62 0.25 295 / 0.6)",
          }}
        />
      </div>
      <ol className="space-y-2">
        {PIPELINE.map((s, i) => (
          <li
            key={s.id}
            className="flex items-center justify-between font-mono text-[11px]"
          >
            <div className="flex items-center gap-2">
              <StepDot status={stageStatuses[i]} />
              <span
                className={
                  stageStatuses[i] === "pending"
                    ? "text-muted-foreground"
                    : "text-foreground"
                }
              >
                {s.name}
              </span>
            </div>
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
              {stageStatuses[i]}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function StepDot({ status }: { status: StageStatus }) {
  if (status === "done")
    return (
      <Circle className="h-2.5 w-2.5 fill-[color:var(--cyan-glow)] text-[color:var(--cyan-glow)]" />
    );
  if (status === "active")
    return (
      <span className="relative flex h-2.5 w-2.5">
        <span
          className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75"
          style={{ background: "var(--violet-glow)" }}
        />
        <span
          className="relative inline-flex h-2.5 w-2.5 rounded-full"
          style={{ background: "var(--violet-glow)" }}
        />
      </span>
    );
  return <Circle className="h-2.5 w-2.5 text-white/20" />;
}

/* ────────────────────────────── spec viewer ────────────────────────────── */

function SpecViewer({
  worldSpec,
}: {
  worldSpec: any;
}) {
  const [expanded, setExpanded] = useState(true);
  const [copied, setCopied] = useState(false);
  const jsonText = worldSpec
    ? JSON.stringify(worldSpec, null, 2)
    : SAMPLE_SPEC;

  const lines = jsonText.split("\n");

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(jsonText);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* noop */
    }
  };

  const onDownload = () => {
    const blob = new Blob([jsonText], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "worldspec.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="mt-5">
      <div className="glass-strong gradient-border overflow-hidden rounded-2xl">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/5 px-5 py-3">
          <div className="flex items-center gap-2">
            <GitBranch className="h-4 w-4 text-[color:var(--blue-glow)]" />
            <h3 className="text-sm font-semibold">WorldSpec</h3>
            <span className="ml-1 rounded-full border border-white/10 bg-white/5 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              worldspec.json
            </span>
            <span className="hidden font-mono text-[11px] text-muted-foreground md:inline">
              schema v0.4.2 · <span className="text-emerald-300">validated</span>
            </span>
          </div>
          <div className="flex items-center gap-1">
            <IconAction onClick={onCopy} icon={copied ? Check : Copy}>
              {copied ? "Copied" : "Copy"}
            </IconAction>
            <IconAction onClick={onDownload} icon={Download}>
              Download JSON
            </IconAction>
            <IconAction
              onClick={() => setExpanded((v) => !v)}
              icon={expanded ? Minimize2 : Maximize2}
            >
              {expanded ? "Collapse" : "Expand"}
            </IconAction>
          </div>
        </div>
        {expanded && (
          <div className="relative max-h-[440px] overflow-auto">
            <pre className="grid grid-cols-[auto_1fr] gap-x-4 p-5 font-mono text-[12.5px] leading-6">
              {lines.map((line, i) => (
                <div key={i} className="contents">
                  <span className="select-none text-right text-muted-foreground/50">
                    {i + 1}
                  </span>
                  <code
                    className="whitespace-pre"
                    dangerouslySetInnerHTML={{ __html: highlight(line) }}
                  />
                </div>
              ))}
            </pre>
          </div>
        )}
      </div>
    </section>
  );
}

function IconAction({
  icon: Icon,
  onClick,
  children,
}: {
  icon: React.ComponentType<{ className?: string }>;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center gap-1.5 rounded-md border border-black/10 bg-black/5 px-2.5 py-1.5 font-mono text-[11px] text-muted-foreground transition-colors hover:bg-black/10 hover:text-foreground"
    >
      <Icon className="h-3 w-3" />
      {children}
    </button>
  );
}

function highlight(line: string) {
  return line
    .replace(
      /(&|<|>)/g,
      (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[m] || m,
    )
    .replace(/(\"[^\"]+\")(\s*:)/g, '<span style="color:oklch(0.4 0.15 260)">$1</span>$2')
    .replace(/:\s*(\"[^\"]*\")/g, ': <span style="color:oklch(0.45 0.15 150)">$1</span>')
    .replace(/:\s*(-?\d+\.?\d*)/g, ': <span style="color:oklch(0.5 0.2 25)">$1</span>')
    .replace(/\b(true|false|null)\b/g, '<span style="color:oklch(0.4 0.2 295)">$1</span>');
}


/* ────────────────────────────── overlays ────────────────────────────── */

function SuccessToast({ show }: { show: boolean }) {
  return (
    <div
      className={`pointer-events-none fixed inset-x-0 bottom-6 z-50 flex justify-center transition-all duration-300 ${
        show ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0"
      }`}
    >
      <div
        className="glass-strong pointer-events-auto flex items-center gap-3 rounded-full px-4 py-2.5"
        style={{ boxShadow: "var(--shadow-glow)" }}
      >
        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-400/20 text-emerald-300">
          <Check className="h-3.5 w-3.5" />
        </span>
        <span className="text-sm font-medium">World Generated Successfully</span>
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          worldspec ready
        </span>
      </div>
    </div>
  );
}


/* ────────────────────────────── footer ────────────────────────────── */

function Footer() {
  return (
    <footer className="border-t border-white/5 py-8">
      <div className="mx-auto flex max-w-[1400px] flex-col items-center justify-between gap-3 px-6 text-xs text-muted-foreground md:flex-row">
        <div className="font-mono">
          © 2026 PhysWorldLM Research · alpine build 4.2.1
        </div>
        <div className="flex items-center gap-4 font-mono">
          <span>compiled with OpenUSD 24.11</span>
          <span className="opacity-40">·</span>
          <span>omniverse kit 106.5</span>
        </div>
      </div>
    </footer>
  );
}
