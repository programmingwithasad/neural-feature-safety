import {
  Beaker,
  CheckCircle2,
  Layers3,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import Sidebar from "../../components/Sidebar";
import Topbar from "../../components/Topbar";

const experiments = [
  {
    id: "EXP-01",
    title: "Sparse Autoencoder Training",
    description:
      "Learning a sparse 2,048-dimensional representation from 768-dimensional transformer activations.",
    status: "Completed",
    icon: Layers3,
  },
  {
    id: "EXP-02",
    title: "Safety Classifier Training",
    description:
      "Training the safety classifier using learned sparse autoencoder features.",
    status: "Completed",
    icon: ShieldCheck,
  },
  {
    id: "EXP-03",
    title: "Feature 850 Intervention Study",
    description:
      "Research evaluation of Feature 850 suppression and its effect on held-out safety classification.",
    status: "Evaluated",
    icon: Sparkles,
  },
];

export default function ExperimentsPage() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="flex min-h-screen">
        <Sidebar />

        <main className="min-w-0 flex-1">
          <Topbar page="Experiments" />

          <div className="mx-auto max-w-[1500px] px-5 py-7 sm:px-7 lg:px-9">
            <section className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="absolute right-0 top-0 h-full w-[45%] bg-gradient-to-l from-violet-50/70 to-transparent" />

              <div className="relative px-7 py-8 sm:px-9 sm:py-10">
                <div className="inline-flex items-center gap-2 rounded-full border border-violet-200 bg-violet-50 px-3 py-1.5">
                  <Beaker className="h-3.5 w-3.5 text-violet-600" />

                  <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-violet-700">
                    Research Workspace
                  </span>
                </div>

                <h2 className="mt-5 text-3xl font-bold tracking-tight text-slate-950 sm:text-4xl">
                  Model{" "}
                  <span className="text-violet-600">
                    Experiments
                  </span>
                </h2>

                <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-500">
                  Track the major experimental stages used to build and
                  evaluate the neural feature safety system.
                </p>
              </div>
            </section>

            <section className="mt-6 grid gap-4">
              {experiments.map((experiment) => {
                const Icon = experiment.icon;

                return (
                  <div
                    key={experiment.id}
                    className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md"
                  >
                    <div className="flex flex-col gap-5 sm:flex-row sm:items-center">
                      <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-violet-50 ring-1 ring-violet-100">
                        <Icon className="h-5 w-5 text-violet-600" />
                      </div>

                      <div className="flex-1">
                        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                          <span className="font-mono text-[10px] font-bold text-slate-400">
                            {experiment.id}
                          </span>

                          <h3 className="text-base font-bold text-slate-950">
                            {experiment.title}
                          </h3>
                        </div>

                        <p className="mt-2 max-w-3xl text-xs leading-6 text-slate-500">
                          {experiment.description}
                        </p>
                      </div>

                      <div className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-2">
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />

                        <span className="text-[10px] font-bold text-emerald-700">
                          {experiment.status}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </section>

            <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400">
                Research Configuration
              </p>

              <h3 className="mt-1 text-base font-bold text-slate-950">
                Current system configuration
              </h3>

              <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-[10px] text-slate-400">
                    Base Encoder
                  </p>

                  <p className="mt-2 text-sm font-bold text-slate-800">
                    DistilBERT
                  </p>
                </div>

                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-[10px] text-slate-400">
                    SAE
                  </p>

                  <p className="mt-2 text-sm font-bold text-slate-800">
                    768 → 2048 → 768
                  </p>
                </div>

                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-[10px] text-slate-400">
                    Activation
                  </p>

                  <p className="mt-2 text-sm font-bold text-slate-800">
                    ReLU
                  </p>
                </div>

                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-[10px] text-slate-400">
                    Threshold
                  </p>

                  <p className="mt-2 text-sm font-bold text-slate-800">
                    0.50
                  </p>
                </div>
              </div>
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}