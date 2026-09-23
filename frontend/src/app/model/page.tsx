"use client";

import { useEffect, useState } from "react";
import {
  ArrowDown,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Layers3,
  ScanText,
  ShieldCheck,
} from "lucide-react";
import Sidebar from "../../components/Sidebar";
import Topbar from "../../components/Topbar";

type HealthData = {
  status: string;
  device: string;
  threshold: number;
};

const stages = [
  {
    number: "01",
    title: "Prompt Input",
    dimension: "Natural Language",
    description: "User prompt enters the neural safety pipeline.",
    icon: ScanText,
  },
  {
    number: "02",
    title: "DistilBERT",
    dimension: "768 Dimensions",
    description: "The prompt is transformed into a mean-pooled semantic representation.",
    icon: BrainCircuit,
  },
  {
    number: "03",
    title: "Sparse Autoencoder",
    dimension: "2,048 Features",
    description: "The 768-dimensional representation is mapped into a sparse neural feature space.",
    icon: Layers3,
  },
  {
    number: "04",
    title: "Safety Classifier",
    dimension: "Probability",
    description: "The learned sparse representation is passed to the trained safety classifier.",
    icon: ShieldCheck,
  },
  {
    number: "05",
    title: "Safety Decision",
    dimension: "Threshold 0.50",
    description: "Harmful probability is compared against the configured decision threshold.",
    icon: CheckCircle2,
  },
];

export default function ModelPage() {
  const [health, setHealth] = useState<HealthData | null>(null);

  useEffect(() => {
    fetch("/api/analyze")
      .catch(() => null);

    fetch("/api/health")
      .then((response) => response.json())
      .then((data) => setHealth(data))
      .catch(() => setHealth(null));
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="flex min-h-screen">
        <Sidebar />

        <main className="min-w-0 flex-1">
          <Topbar page="Model Architecture" />

          <div className="mx-auto max-w-[1500px] px-5 py-7 sm:px-7 lg:px-9">
            <section className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="absolute right-0 top-0 h-full w-[45%] bg-gradient-to-l from-violet-50/70 to-transparent" />

              <div className="relative px-7 py-8 sm:px-9 sm:py-10">
                <div className="inline-flex items-center gap-2 rounded-full border border-violet-200 bg-violet-50 px-3 py-1.5">
                  <BrainCircuit className="h-3.5 w-3.5 text-violet-600" />

                  <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-violet-700">
                    Model Architecture
                  </span>
                </div>

                <h2 className="mt-5 text-3xl font-bold tracking-tight text-slate-950 sm:text-4xl">
                  Neural Safety{" "}
                  <span className="text-cyan-600">
                    Pipeline
                  </span>
                </h2>

                <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-500">
                  A layered inference architecture combining transformer
                  representations, sparse feature extraction, and a trained
                  safety classifier.
                </p>
              </div>
            </section>

            <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  Encoder
                </p>

                <p className="mt-2 text-xl font-bold text-slate-950">
                  DistilBERT
                </p>

                <p className="mt-1 text-[11px] text-slate-500">
                  768-dimensional representation
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  SAE Input
                </p>

                <p className="mt-2 text-xl font-bold text-slate-950">
                  768
                </p>

                <p className="mt-1 text-[11px] text-slate-500">
                  Transformer representation
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  SAE Features
                </p>

                <p className="mt-2 text-xl font-bold text-slate-950">
                  2,048
                </p>

                <p className="mt-1 text-[11px] text-slate-500">
                  Sparse ReLU features
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  Decision
                </p>

                <p className="mt-2 text-xl font-bold text-slate-950">
                  0.50
                </p>

                <p className="mt-1 text-[11px] text-slate-500">
                  Harmful probability threshold
                </p>
              </div>
            </div>

            <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="mb-6">
                <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400">
                  Inference Flow
                </p>

                <h3 className="mt-1 text-base font-bold text-slate-950">
                  End-to-end neural processing
                </h3>
              </div>

              <div className="mx-auto max-w-3xl">
                {stages.map((stage, index) => {
                  const Icon = stage.icon;

                  return (
                    <div key={stage.number}>
                      <div className="flex items-center gap-4 rounded-2xl border border-slate-200 bg-slate-50 p-5 transition hover:border-cyan-200 hover:bg-cyan-50/30">
                        <div className="hidden text-[10px] font-bold text-slate-400 sm:block">
                          {stage.number}
                        </div>

                        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-white ring-1 ring-slate-200">
                          <Icon className="h-5 w-5 text-cyan-600" />
                        </div>

                        <div className="min-w-0 flex-1">
                          <div className="flex flex-col justify-between gap-1 sm:flex-row sm:items-center">
                            <h4 className="text-sm font-bold text-slate-950">
                              {stage.title}
                            </h4>

                            <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-600">
                              {stage.dimension}
                            </span>
                          </div>

                          <p className="mt-1 text-[11px] leading-5 text-slate-500">
                            {stage.description}
                          </p>
                        </div>
                      </div>

                      {index < stages.length - 1 && (
                        <div className="flex justify-center py-2">
                          <ArrowDown className="h-4 w-4 text-slate-300" />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>

            <section className="mt-6 grid gap-6 lg:grid-cols-2">
              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-50">
                    <Cpu className="h-5 w-5 text-cyan-600" />
                  </div>

                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                      Runtime
                    </p>

                    <h3 className="mt-1 text-sm font-bold text-slate-950">
                      Inference Environment
                    </h3>
                  </div>
                </div>

                <div className="mt-5 space-y-3">
                  <div className="flex justify-between rounded-xl bg-slate-50 px-4 py-3">
                    <span className="text-xs text-slate-500">
                      Device
                    </span>

                    <span className="text-xs font-bold text-slate-800">
                      {health?.device || "CUDA"}
                    </span>
                  </div>

                  <div className="flex justify-between rounded-xl bg-slate-50 px-4 py-3">
                    <span className="text-xs text-slate-500">
                      Threshold
                    </span>

                    <span className="text-xs font-bold text-slate-800">
                      {health?.threshold?.toFixed(2) || "0.50"}
                    </span>
                  </div>

                  <div className="flex justify-between rounded-xl bg-slate-50 px-4 py-3">
                    <span className="text-xs text-slate-500">
                      API Status
                    </span>

                    <span className="text-xs font-bold text-emerald-600">
                      {health?.status === "healthy"
                        ? "Healthy"
                        : "Configured"}
                    </span>
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  Feature Interpretability
                </p>

                <h3 className="mt-1 text-sm font-bold text-slate-950">
                  SAE Feature 850
                </h3>

                <p className="mt-4 text-[11px] leading-6 text-slate-500">
                  Feature 850 is exposed in the application as an
                  interpretability signal. Its activation is observed for
                  individual prompts without modifying the production
                  representation.
                </p>

                <div className="mt-5 rounded-xl border border-emerald-200 bg-emerald-50 p-4">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />

                    <span className="text-xs font-bold text-emerald-700">
                      Observational Mode
                    </span>
                  </div>

                  <p className="mt-2 text-[10px] leading-5 text-slate-500">
                    Production intervention is disabled.
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