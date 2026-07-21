"use client";

import { motion } from "framer-motion";
import { Boxes, Github, ChevronRight } from "lucide-react";

const NAV_LINKS = [
  { label: "WorldSpec", href: "#worldspec" },
  { label: "Pipeline", href: "#pipeline" },
  { label: "Docs", href: "#docs" },
];

export default function Navbar() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="sticky top-0 z-50 border-b border-white/[0.06] bg-[#09090b]/70 backdrop-blur-xl"
    >
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6 lg:px-8">
        <a href="/" className="flex items-center gap-2.5">
          <div className="relative flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-gradient-to-br from-blue-500/20 to-violet-500/20">
            <Boxes className="h-4 w-4 text-blue-400" strokeWidth={2} />
          </div>
          <span className="text-[15px] font-semibold tracking-tight text-zinc-100">
            PhysWorldLM
          </span>
          <span className="ml-1 hidden rounded-full border border-white/10 bg-white/[0.03] px-2 py-0.5 font-mono-data text-[10px] uppercase tracking-wider text-zinc-500 sm:inline-block">
            v0.4 alpha
          </span>
        </a>

        <nav className="hidden items-center gap-8 md:flex">
          {NAV_LINKS.map((link) => (
            <a
              key={link.label}
              href={link.href}
              className="text-sm text-zinc-400 transition-colors hover:text-zinc-100"
            >
              {link.label}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <a
            href="https://github.com"
            className="hidden items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.02] px-3.5 py-1.5 text-sm text-zinc-300 transition-all hover:border-white/20 hover:bg-white/[0.05] sm:flex"
          >
            <Github className="h-3.5 w-3.5" strokeWidth={2} />
            Source
          </a>
          <a
            href="#editor"
            className="group flex items-center gap-1 rounded-lg bg-zinc-100 px-3.5 py-1.5 text-sm font-medium text-zinc-900 transition-all hover:bg-white"
          >
            Open Editor
            <ChevronRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
          </a>
        </div>
      </div>
    </motion.header>
  );
}
