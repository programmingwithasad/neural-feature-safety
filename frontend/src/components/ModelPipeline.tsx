import {
  ArrowRight,
  BrainCircuit,
  Layers3,
  ScanText,
  ShieldCheck,
} from "lucide-react";

const stages = [
  {
    title: "Prompt",
    subtitle: "Input text",
    icon: ScanText,
  },
  {
    title: "DistilBERT",
    subtitle: "768D representation",
    icon: BrainCircuit,
  },
  {
    title: "Sparse Autoencoder",
    subtitle: "2,048 sparse features",
    icon: Layers3,
  },
  {
    title: "Safety Classifier",
    subtitle: "Risk probability",
    icon: ShieldCheck,
  },
];

export default function ModelPipeline() {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-slate-400">
            Neural Architecture
          </p>

          <h2 className="mt-1 text-base font-bold text-slate-950">
            Safety inference pipeline
          </h2>
        </div>

        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-[10px] font-semibold text-slate-500">
          Production Path
        </span>
      </div>

      <div className="mt-6 flex flex-col gap-3 xl:flex-row xl:items-center">
        {stages.map((stage, index) => {
          const Icon = stage.icon;

          return (
            <div
              key={stage.title}
              className="flex min-w-0 flex-1 items-center gap-3"
            >
              <div className="flex min-w-0 flex-1 items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white ring-1 ring-slate-200">
                  <Icon className="h-4 w-4 text-cyan-600" />
                </div>

                <div className="min-w-0">
                  <p className="truncate text-xs font-bold text-slate-800">
                    {stage.title}
                  </p>

                  <p className="mt-1 truncate text-[10px] text-slate-500">
                    {stage.subtitle}
                  </p>
                </div>
              </div>

              {index < stages.length - 1 && (
                <ArrowRight className="hidden h-4 w-4 shrink-0 text-slate-300 xl:block" />
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}