"use client";

import { motion } from "framer-motion";

export default function Hero() {
  return (
    <section className="relative overflow-hidden pb-16 pt-20 sm:pb-20 sm:pt-28">
      {/* Ambient grid + radial fade */}
      <div className="pointer-events-none absolute inset-0 bg-grid [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,black_10%,transparent_70%)]" />

      {/* Drifting gradient orbs */}
      <motion.div
        aria-hidden
        className="pointer-events-none absolute left-1/2 top-[-120px] h-[420px] w-[620px] -translate-x-1/2 rounded-full bg-blue-600/[0.14] blur-[110px]"
        animate={{ x: [0, 40, -30, 0], y: [0, 20, -10, 0] }}
        transition={{ duration: 22, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        aria-hidden
        className="pointer-events-none absolute left-1/2 top-[-60px] h-[360px] w-[480px] -translate-x-[80%] rounded-full bg-violet-600/[0.1] blur-[100px]"
        animate={{ x: [0, -30, 20, 0], y: [0, -15, 10, 0] }}
        transition={{ duration: 26, repeat: Infinity, ease: "easeInOut" }}
      />

      <div className="relative mx-auto max-w-4xl px-6 text-center lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mx-auto mb-6 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1.5"
        >
          <span className="relative flex h-1.5 w-1.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue-400 opacity-75" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-blue-400" />
          </span>
          <span className="font-mono-data text-xs tracking-wide text-zinc-400">
            language → WorldSpec → OpenUSD + PhysX
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.05, ease: [0.16, 1, 0.3, 1] }}
          className="text-balance text-5xl font-semibold tracking-tight text-gradient sm:text-6xl lg:text-7xl"
        >
          PhysWorldLM
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
          className="mx-auto mt-5 max-w-xl text-lg text-zinc-300 sm:text-xl"
        >
          Physics-Aware Conversational World Editor
        </motion.p>

        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.25, ease: [0.16, 1, 0.3, 1] }}
          className="mx-auto mt-3 max-w-md text-sm text-zinc-500"
        >
          Generate physically consistent virtual worlds using natural
          language.
        </motion.p>
      </div>
    </section>
  );
}
