"use client";

import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Terminal, ArrowUpRight, Loader2, CornerDownLeft } from "lucide-react";

const MAX_LENGTH = 800;

interface PromptEditorProps {
  value: string;
  onChange: (value: string) => void;
  onGenerate: () => void;
  isGenerating: boolean;
}

export default function PromptEditor({
  value,
  onChange,
  onGenerate,
  isGenerating,
}: PromptEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 360)}px`;
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      if (value.trim() && !isGenerating) onGenerate();
    }
  };

  const nearLimit = value.length > MAX_LENGTH * 0.9;

  return (
    <div
      id="editor"
      className="glass-panel relative rounded-2xl p-1.5 shadow-[0_0_0_1px_rgba(255,255,255,0.02),0_20px_60px_-20px_rgba(0,0,0,0.6)] transition-colors focus-within:border-blue-500/40"
    >
      <div className="flex items-center justify-between px-4 pb-2 pt-3">
        <div className="flex items-center gap-2 text-zinc-500">
          <Terminal className="h-3.5 w-3.5" strokeWidth={2} />
          <span className="font-mono-data text-[11px] uppercase tracking-wider">
            scene_prompt.txt
          </span>
        </div>
        <span
          className={`font-mono-data text-[11px] tabular-nums ${
            nearLimit ? "text-amber-400" : "text-zinc-600"
          }`}
        >
          {value.length.toString().padStart(3, "0")} / {MAX_LENGTH}
        </span>
      </div>

      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value.slice(0, MAX_LENGTH))}
        onKeyDown={handleKeyDown}
        rows={4}
        placeholder={
          "Describe a physically realistic world…\n\nExample: generate two F-16 aircraft intercepting a hostile bomber over mountainous terrain while avoiding enemy radar."
        }
        className="max-h-[360px] min-h-[120px] w-full resize-none bg-transparent px-4 pb-3 text-[15px] leading-relaxed text-zinc-100 placeholder:text-zinc-600 focus:outline-none"
      />

      <div className="flex items-center justify-between gap-3 border-t border-white/[0.06] p-3">
        <div className="hidden items-center gap-1.5 pl-1 font-mono-data text-[11px] text-zinc-600 sm:flex">
          <kbd className="rounded border border-white/10 bg-white/[0.04] px-1.5 py-0.5">
            ⌘
          </kbd>
          <kbd className="rounded border border-white/10 bg-white/[0.04] px-1.5 py-0.5">
            <CornerDownLeft className="h-2.5 w-2.5" />
          </kbd>
          <span>to compile</span>
        </div>

        <motion.button
          type="button"
          onClick={onGenerate}
          disabled={!value.trim() || isGenerating}
          whileTap={{ scale: 0.98 }}
          className="group relative flex flex-1 items-center justify-center gap-2 overflow-hidden rounded-xl bg-gradient-to-r from-blue-600 to-blue-500 py-3 text-sm font-medium text-white shadow-[0_0_24px_-6px_rgba(59,130,246,0.6)] transition-all disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none sm:flex-initial sm:px-8"
        >
          <span className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/25 to-transparent transition-transform duration-700 group-hover:translate-x-full" />
          <AnimatePresence mode="wait" initial={false}>
            {isGenerating ? (
              <motion.span
                key="generating"
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="relative flex items-center gap-2"
              >
                <Loader2 className="h-4 w-4 animate-spin" />
                Generating world…
              </motion.span>
            ) : (
              <motion.span
                key="idle"
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="relative flex items-center gap-2"
              >
                Compile WorldSpec
                <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </motion.span>
            )}
          </AnimatePresence>
        </motion.button>
      </div>
    </div>
  );
}
