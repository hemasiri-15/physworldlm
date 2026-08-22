import { createFileRoute, Link } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Physics-Aware Executable Digital Twins" },
      {
        name: "description",
        content: "An Integrated Semantic Architecture for Physics-Aware Executable Digital Twins",
      },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <div className="min-h-screen bg-white text-gray-900 font-sans selection:bg-blue-100">
      <div className="max-w-4xl mx-auto px-6 py-16 md:py-24">
        {/* Header Section */}
        <header className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-6 leading-tight">
            An Integrated Semantic Architecture for
            <br />
            Physics-Aware Executable Digital Twins
          </h1>

          <div className="flex flex-wrap justify-center gap-x-6 gap-y-2 text-lg text-blue-600 mb-4 font-medium">
            <span>Vijay Rao</span>
            <span>Hema Siri Guduru</span>
            <span>Shashanth Reddy Mannem</span>
            <span>Kushal Pandey</span>
            <span>Kesari Charan</span>
          </div>

          <div className="text-gray-600 text-lg mb-8">
            Department of AI&CSE, Mahindra University
            <br />
            Hyderabad, India
          </div>

          <div className="flex justify-center gap-4">
            <Link
              to="/editor"
              className="inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-full text-white bg-black hover:bg-gray-800 transition-colors"
            >
              Open World Editor
            </Link>
            <a
              href="#"
              className="inline-flex items-center justify-center px-6 py-3 border border-gray-300 text-base font-medium rounded-full text-gray-700 bg-white hover:bg-gray-50 transition-colors"
            >
              Paper
            </a>
          </div>
        </header>

        {/* Main Content */}
        <article className="max-w-none text-gray-700 text-lg leading-relaxed space-y-6">
          <h2 className="text-2xl font-bold text-gray-900 mt-12 mb-6 pb-2 border-b border-gray-200">
            Abstract
          </h2>
          <p className="text-justify">
            Modern large language models excel at descriptive text and image generation but remain
            fundamentally limited in producing physically consistent, semantically grounded, and
            executable representations of the real world. This gap between natural language
            understanding and interactive, physics-driven simulation constrains their utility for
            digital twin generation and high-fidelity training environments.
          </p>
          <p className="text-justify">
            We present an ontology-driven modeling and simulation framework that translates natural
            language descriptions into structured, machine-interpretable WorldSpec representations
            spanning entities, environments, physics, sensors, events, timelines, assets, and
            inter-object relationships. At its core, a modular Scene Compiler interprets validated
            WorldSpec instances and constructs a canonical Scene Graph, decoupling semantic
            reasoning, scene compilation, physics simulation, and visualization into independent,
            interoperable modules rather than coupling generation to a single rendering backend.
          </p>
          <p className="text-justify">
            Physical simulation is performed using NVIDIA PhysX, supporting rigid-body dynamics,
            collision detection, constraints, and vehicle and projectile behavior. Compiled scenes
            are exported as standardized OpenUSD assets for interoperability across modern digital
            content creation pipelines, with photorealistic visualization provided by NVIDIA
            Omniverse RTX and real-world geospatial fidelity achieved through Cesium-based terrain,
            elevation, and satellite imagery integration.
          </p>
          <p className="text-justify">
            Together, these components establish a reproducible, semantically consistent, and
            backend-agnostic foundation for synthetic dataset generation, reinforcement learning
            environments, executable digital twins, and mission rehearsal applications.
          </p>

          <h2 className="text-2xl font-bold text-gray-900 mt-12 mb-6 pb-2 border-b border-gray-200">
            System Architecture Overview
          </h2>
          <p className="text-justify">
            The World Compiler pipeline comprises: (1) Semantic Parsing, (2) Ontology Grounding, (3)
            WorldSpec Construction, (4) Reasoning Layer, (5) Scene Compilation, (6) Scene Graph
            Generation, and (7) Backend Execution. Language interpretation maps natural language
            onto the hierarchical domain ontology, resolving entities, relationships, and
            constraints into a validated WorldSpec instance. The Reasoning Layer checks this
            instance for physical and semantic consistency before the Scene Compiler traverses it to
            construct the Scene Graph, applying modular builders for transforms, materials, physics
            metadata, sensors, and relationships. The Scene Graph is the canonical,
            backend-independent representation consumed both by the export pipeline and by extension
            modules. Each stage communicates exclusively through standardized schemas rather than
            backend-specific APIs, allowing individual modules to be replaced or extended without
            affecting upstream compiler stages.
          </p>

          <h2 className="text-2xl font-bold text-gray-900 mt-12 mb-6 pb-2 border-b border-gray-200">
            Contributions
          </h2>
          <ul className="list-disc pl-6 space-y-3">
            <li>
              A backend-agnostic World Compiler that translates natural language descriptions into
              executable digital twins through a compiler-inspired pipeline comprising semantic
              parsing, intermediate representation, reasoning, scene compilation, and
              backend-specific code generation.
            </li>
            <li>
              A hierarchical, ontology-driven semantic grounding framework spanning military and
              civilian domains, supporting inheritance, extensibility, semantic consistency, and
              cross-domain reuse.
            </li>
            <li>
              A canonical WorldSpec representation enabling structured scene validation, reasoning,
              and interoperability across heterogeneous simulation backends.
            </li>
            <li>
              A modular Scene Compiler that transforms WorldSpec instances into backend-independent
              Scene Graphs and OpenUSD assets via a builder-pattern architecture.
            </li>
            <li>
              A dedicated Reasoning Layer that enforces semantic, physical, and environmental
              consistency checking prior to scene compilation, treating reasoning as a first-class
              architectural component rather than an implicit validation stage.
            </li>
            <li>
              Unified integration of autonomous mission planning, physics simulation, and geospatial
              reasoning within a unified ontology-driven execution framework.
            </li>
            <li>
              An extensible and backend-independent architecture that supports future physics
              engines, rendering pipelines, robotics middleware, and simulation platforms without
              requiring modifications to upstream semantic representations or compiler components.
            </li>
          </ul>
        </article>

        <footer className="mt-24 pt-8 border-t border-gray-200 text-center text-gray-500 text-sm">
          <p>© {new Date().getFullYear()} Mahindra University. All rights reserved.</p>
        </footer>
      </div>
    </div>
  );
}
