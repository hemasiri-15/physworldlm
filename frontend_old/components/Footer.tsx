const POWERED_BY = ["OpenUSD", "PhysX", "LLMs", "WorldSpec"];

export default function Footer() {
  return (
    <footer className="border-t border-white/[0.06] py-10">
      <div className="mx-auto flex max-w-7xl flex-col items-center gap-4 px-6 text-center lg:px-8">
        <span className="text-[11px] uppercase tracking-wider text-zinc-600">
          Powered by
        </span>
        <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2">
          {POWERED_BY.map((item) => (
            <span
              key={item}
              className="font-mono-data text-sm text-zinc-500 transition-colors hover:text-zinc-300"
            >
              {item}
            </span>
          ))}
        </div>
        <p className="mt-2 text-xs text-zinc-700">
          © {new Date().getFullYear()} PhysWorldLM. Research preview.
        </p>
      </div>
    </footer>
  );
}
