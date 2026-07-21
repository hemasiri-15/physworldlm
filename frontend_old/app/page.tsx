export default function Home() {
  return (
    <main className="min-h-screen bg-[#09090b] text-white">

      {/* Background */}
      <div className="absolute inset-0 bg-grid opacity-20" />

      {/* Header */}
      <header className="relative z-10 border-b border-white/10 backdrop-blur-xl">
        <div className="mx-auto max-w-7xl px-8 py-5 flex justify-between items-center">

          <div>
            <h1 className="text-3xl font-bold text-gradient">
              🚀 PhysWorldLM
            </h1>
            <p className="text-sm text-zinc-400">
              Physics-Aware Conversational World Editor
            </p>
          </div>

          <button className="rounded-xl border border-blue-500/40 px-5 py-2 hover:bg-blue-600 transition">
            Documentation
          </button>

        </div>
      </header>

      {/* Hero */}

      <section className="relative z-10 mx-auto max-w-7xl px-8 pt-20">

        <h2 className="text-6xl font-bold leading-tight max-w-4xl">
          Generate
          <span className="text-gradient"> Physics-Aware </span>
          Virtual Worlds
        </h2>

        <p className="mt-6 max-w-3xl text-xl text-zinc-400 leading-relaxed">
          Transform natural language into structured WorldSpecs,
          simulation-ready OpenUSD scenes, planners and Omniverse
          environments.
        </p>

      </section>

      {/* Workspace */}

      <section className="relative z-10 mx-auto max-w-7xl px-8 mt-16 pb-20">

        <div className="grid grid-cols-12 gap-8">

          {/* LEFT */}

          <div className="col-span-7">

            <div className="glass-panel rounded-3xl p-8">

              <h3 className="text-2xl font-semibold mb-6">
                Describe your world
              </h3>

              <textarea
                placeholder={`Example:

Generate two F-16 aircraft intercepting a hostile bomber over mountainous terrain while avoiding enemy radar.

Mission:
• Altitude 7000m
• Speed 250 m/s
• Mountain terrain
• Enemy SAM radar
• Sunset lighting`}
                className="w-full h-80 rounded-2xl bg-black/30 border border-white/10 p-6 text-lg resize-none outline-none"
              />

              <button className="mt-8 w-full rounded-2xl bg-gradient-to-r from-blue-600 to-violet-600 py-5 text-xl font-semibold hover:scale-[1.02] transition">
                ✨ Generate World
              </button>

            </div>

          </div>

          {/* RIGHT */}

          <div className="col-span-5 space-y-6">

            <div className="glass-panel rounded-3xl p-6">

              <h3 className="text-xl font-semibold mb-4">
                Status
              </h3>

              <div className="space-y-3">

                <p>🟢 Ready</p>
                <p className="text-zinc-400">
                  Waiting for prompt...
                </p>

              </div>

            </div>

            <div className="glass-panel rounded-3xl p-6 h-[250px]">

              <h3 className="text-xl font-semibold mb-4">
                Scene Preview
              </h3>

              <div className="h-full rounded-xl border border-dashed border-white/20 flex items-center justify-center text-zinc-500">

                OpenUSD Preview

              </div>

            </div>

          </div>

        </div>

        {/* Bottom */}

        <div className="glass-panel rounded-3xl mt-8 p-8">

          <div className="flex justify-between items-center">

            <h3 className="text-2xl font-semibold">
              Generated WorldSpec
            </h3>

            <span className="text-green-400">
              JSON
            </span>

          </div>

          <pre className="mt-6 rounded-2xl bg-black/40 p-6 overflow-auto text-sm text-green-300 font-mono-data">

{`{
  "scene": "...",
  "entities": [],
  "constraints": [],
  "physics": {},
  "planner": {}
}`}

          </pre>

        </div>

      </section>

    </main>
  );
}
