"use client";

import { motion } from "framer-motion";
import {
  Swords,
  Waves,
  Building2,
  LifeBuoy,
  Car,
  Mountain,
  Satellite,
  Anchor,
} from "lucide-react";

interface Suggestion {
  label: string;
  icon: React.ElementType;
  prompt: string;
}

const SUGGESTIONS: Suggestion[] = [
  {
    label: "Dogfight",
    icon: Swords,
    prompt:
      "Generate two F-16 aircraft intercepting a hostile bomber over mountainous terrain while avoiding enemy radar.",
  },
  {
    label: "Flood simulation",
    icon: Waves,
    prompt:
      "Simulate a river overflowing its banks after sustained rainfall, flooding a low-lying town with rigid-body debris carried by the current.",
  },
  {
    label: "Smart city",
    icon: Building2,
    prompt:
      "Model a city intersection with autonomous traffic signals, pedestrian flow, and delivery drones coordinating airspace above street level.",
  },
  {
    label: "Rescue mission",
    icon: LifeBuoy,
    prompt:
      "Generate a coastal search-and-rescue scenario with a helicopter lowering a rescue basket to a capsized sailboat in rough seas.",
  },
  {
    label: "Autonomous driving",
    icon: Car,
    prompt:
      "Simulate a self-driving vehicle navigating a rain-slicked highway merge with reduced tire friction and adjacent vehicles changing lanes.",
  },
  {
    label: "Volcanic eruption",
    icon: Mountain,
    prompt:
      "Generate a stratovolcano eruption with pyroclastic flow spreading down a forested slope, igniting vegetation on contact.",
  },
  {
    label: "Space docking",
    icon: Satellite,
    prompt:
      "Simulate a spacecraft performing an orbital docking maneuver with a rotating space station under zero-gravity constraints.",
  },
  {
    label: "Naval battle",
    icon: Anchor,
    prompt:
      "Generate two destroyer-class ships exchanging fire in open water, with buoyancy, wave displacement, and hull damage states.",
  },
];

interface PromptSuggestionsProps {
  onSelect: (prompt: string) => void;
}

export default function PromptSuggestions({
  onSelect,
}: PromptSuggestionsProps) {
  return (
    <div className="mt-4 flex flex-wrap gap-2">
      {SUGGESTIONS.map((s, i) => (
        <motion.button
          key={s.label}
          type="button"
          onClick={() => onSelect(s.prompt)}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, delay: 0.03 * i }}
          whileHover={{ y: -1 }}
          className="group flex items-center gap-1.5 rounded-full border border-white/[0.08] bg-white/[0.02] px-3.5 py-1.5 text-[13px] text-zinc-400 transition-all hover:border-blue-500/30 hover:bg-blue-500/[0.06] hover:text-zinc-100"
        >
          <s.icon
            className="h-3.5 w-3.5 text-zinc-500 transition-colors group-hover:text-blue-400"
            strokeWidth={2}
          />
          {s.label}
        </motion.button>
      ))}
    </div>
  );
}
