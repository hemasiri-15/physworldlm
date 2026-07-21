export default function Home() {
  return (
    <main className="min-h-screen bg-gray-950 text-white flex items-center justify-center p-8">
      <div className="w-full max-w-5xl">

        <h1 className="text-6xl font-bold text-center">
          🚀 PhysWorldLM
        </h1>

        <p className="text-center text-gray-400 mt-3 mb-10 text-lg">
          Physics-Aware Conversational World Editor
        </p>

        <div className="bg-gray-900 rounded-2xl p-8 shadow-xl">

          <h2 className="text-xl font-semibold mb-4">
            Describe your world
          </h2>

          <textarea
            className="w-full h-56 rounded-xl bg-gray-800 border border-gray-700 p-5 text-lg outline-none resize-none"
            placeholder="Example:

Generate two F-16 aircraft intercepting a hostile bomber over mountainous terrain while avoiding enemy radar."
          />

          <button
            className="mt-6 w-full rounded-xl bg-blue-600 hover:bg-blue-700 py-4 text-xl font-semibold transition"
          >
            Generate World
          </button>

        </div>

      </div>
    </main>
  );
}
