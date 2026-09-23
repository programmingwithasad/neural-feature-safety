"use client";

import Link from "next/link";
import { useState } from "react";
import {
  Activity,
  ArrowLeft,
  BrainCircuit,
  CheckCircle2,
  History,
  Info,
  Layers3,
  LayoutDashboard,
  ScanSearch,
  ScanText,
  Shield,
  Sparkles,
  Zap,
} from "lucide-react";
import Sidebar from "../../components/Sidebar";
import Topbar from "../../components/Topbar";

type AnalysisResult = {
  harmful_probability: number;
  unharmful_probability: number;
  risk_level: string;
  decision: string;
  threshold: number;
  feature_850_activation: number;
  device: string;
  error?: string;
};

export default function FeatureExplorer() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);

  const analyzeFeature = async () => {
    const value = text.trim();

    if (!value || loading) {
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: value,
        }),
        cache: "no-store",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Feature analysis failed.");
      }

      setResult(data);
    } catch (error) {
      setResult({
        harmful_probability: 0,
        unharmful_probability: 0,
        risk_level: "ERROR",
        decision: "UNAVAILABLE",
        threshold: 0.5,
        feature_850_activation: 0,
        device: "Unknown",
        error:
          error instanceof Error
            ? error.message
            : "Unable to connect to the safety API.",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLTextAreaElement>
  ) => {
    if (event.ctrlKey && event.key === "Enter") {
      event.preventDefault();
      analyzeFeature();
    }
  };

  const activation = result?.feature_850_activation ?? 0;

  const activationWidth = Math.min(
    Math.max(activation * 20, 0),
    100
  );

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="flex min-h-screen">
        <Sidebar />

        <main className="min-w-0 flex-1">
          <Topbar page = "Feature Explorer" />

          <div className="mx-auto max-w-[1500px] px-5 py-7 sm:px-7 lg:px-9">
            <Link
              href="/"
              className="mb-6 inline-flex items-center gap-2 text-xs font-medium text-slate-500 transition hover:text-cyan-600"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              Back to Dashboard
            </Link>

            <section className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="absolute right-0 top-0 h-full w-[48%] bg-gradient-to-l from-cyan-50/80 via-cyan-50/20 to-transparent" />

              <div className="relative px-7 py-8 sm:px-9 sm:py-10">
                <div className="flex flex-col justify-between gap-7 lg:flex-row lg:items-center">
                  <div>
                    <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1.5">
                      <Layers3 className="h-3.5 w-3.5 text-cyan-600" />

                      <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-cyan-700">
                        Sparse Autoencoder
                      </span>
                    </div>

                    <h2 className="text-3xl font-bold tracking-tight text-slate-950 sm:text-4xl">
                      Neural Feature{" "}
                      <span className="text-cyan-600">
                        Explorer
                      </span>
                    </h2>

                    <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-500">
                      Explore the observed activation of SAE Feature 850
                      for individual prompts processed through the neural
                      safety pipeline.
                    </p>
                  </div>

                  <div className="relative flex shrink-0 items-center gap-3 rounded-xl border border-emerald-200 bg-emerald-50 px-5 py-4">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-white">
                      <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                    </div>

                    <div>
                      <p className="text-xs font-bold text-emerald-700">
                        Observational Mode
                      </p>

                      <p className="mt-1 text-[10px] text-slate-500">
                        No production intervention
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <div className="mt-6 grid gap-6 xl:grid-cols-[1.35fr_0.65fr]">
              <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
                <div className="border-b border-slate-200 px-6 py-5">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-50 ring-1 ring-cyan-100">
                      <ScanSearch className="h-4 w-4 text-cyan-600" />
                    </div>

                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-cyan-600">
                        Interpretability
                      </p>

                      <h3 className="mt-1 text-base font-bold text-slate-950">
                        Feature Activation Analyzer
                      </h3>

                      <p className="mt-0.5 text-[11px] text-slate-500">
                        Submit a prompt and observe Feature 850 activation.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="p-6">
                  <textarea
                    value={text}
                    onChange={(event) => setText(event.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Enter a prompt to inspect Feature 850..."
                    className="min-h-[190px] w-full resize-none rounded-xl border border-slate-200 bg-slate-50 p-5 text-sm leading-6 text-slate-800 outline-none transition placeholder:text-slate-400 focus:border-cyan-400 focus:bg-white focus:ring-4 focus:ring-cyan-50"
                  />

                  <div className="mt-4 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
                    <div className="flex items-center gap-2 text-[11px] text-slate-400">
                      <span>Shortcut</span>

                      <span className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1 font-medium text-slate-500">
                        Ctrl
                      </span>

                      <span>+</span>

                      <span className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1 font-medium text-slate-500">
                        Enter
                      </span>
                    </div>

                    <button
                      type="button"
                      onClick={analyzeFeature}
                      disabled={!text.trim() || loading}
                      className="flex items-center justify-center gap-2 rounded-xl bg-slate-950 px-6 py-3 text-xs font-bold text-white shadow-sm transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {loading ? (
                        <>
                          <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                          Inspecting Feature
                        </>
                      ) : (
                        <>
                          Inspect Feature
                          <ScanSearch className="h-4 w-4" />
                        </>
                      )}
                    </button>
                  </div>

                  {result && !result.error && (
                    <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-6">
                      <div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
                        <div>
                          <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400">
                            Feature 850 Activation
                          </p>

                          <p className="mt-2 font-mono text-3xl font-bold tracking-tight text-cyan-600">
                            {activation.toFixed(4)}
                          </p>
                        </div>

                        <div className="sm:text-right">
                          <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                            Inference Device
                          </p>

                          <p className="mt-1 text-sm font-bold text-slate-700">
                            {result.device}
                          </p>
                        </div>
                      </div>

                      <div className="mt-6">
                        <div className="h-3 overflow-hidden rounded-full bg-slate-200">
                          <div
                            className="h-full rounded-full bg-cyan-500 transition-all duration-700"
                            style={{
                              width: `${activationWidth}%`,
                            }}
                          />
                        </div>

                        <div className="mt-2 flex justify-between text-[10px] text-slate-400">
                          <span>0.00</span>
                          <span>Activation intensity</span>
                          <span>5.00+</span>
                        </div>
                      </div>

                      <div className="mt-6 grid gap-3 sm:grid-cols-2">
                        <div className="rounded-xl border border-slate-200 bg-white p-4">
                          <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                            Safety Classification
                          </p>

                          <p className="mt-2 text-sm font-bold text-slate-800">
                            {result.risk_level}
                          </p>
                        </div>

                        <div className="rounded-xl border border-slate-200 bg-white p-4">
                          <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                            Decision
                          </p>

                          <p className="mt-2 text-sm font-bold text-slate-800">
                            {result.decision}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {result?.error && (
                    <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-5">
                      <p className="text-xs font-bold text-red-700">
                        Feature analysis failed
                      </p>

                      <p className="mt-2 text-[11px] leading-5 text-red-600">
                        {result.error}
                      </p>
                    </div>
                  )}
                </div>
              </section>

              <div className="space-y-6">
                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400">
                        Feature ID
                      </p>

                      <p className="mt-2 font-mono text-3xl font-bold text-slate-950">
                        850
                      </p>
                    </div>

                    <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-cyan-50 ring-1 ring-cyan-100">
                      <Layers3 className="h-5 w-5 text-cyan-600" />
                    </div>
                  </div>

                  <div className="mt-6 border-t border-slate-200 pt-5">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-500">
                        Feature space
                      </span>

                      <span className="text-xs font-bold text-slate-800">
                        2,048
                      </span>
                    </div>

                    <div className="mt-4 flex items-center justify-between">
                      <span className="text-xs text-slate-500">
                        Input representation
                      </span>

                      <span className="text-xs font-bold text-slate-800">
                        768
                      </span>
                    </div>

                    <div className="mt-4 flex items-center justify-between">
                      <span className="text-xs text-slate-500">
                        Activation function
                      </span>

                      <span className="text-xs font-bold text-emerald-600">
                        ReLU
                      </span>
                    </div>
                  </div>
                </section>

                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <div className="flex items-start gap-3">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-cyan-50">
                      <Info className="h-4 w-4 text-cyan-600" />
                    </div>

                    <div>
                      <h3 className="text-sm font-bold text-slate-950">
                        Interpretability Note
                      </h3>

                      <p className="mt-3 text-[11px] leading-6 text-slate-500">
                        Feature 850 is exposed as an observed sparse
                        representation feature. Activation varies between
                        prompts and is presented for interpretability
                        research.
                      </p>
                    </div>
                  </div>
                </section>

                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <div className="flex items-center gap-3">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-violet-50">
                      <BrainCircuit className="h-4 w-4 text-violet-600" />
                    </div>

                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400">
                        Architecture
                      </p>

                      <h3 className="mt-1 text-sm font-bold text-slate-950">
                        Feature Processing
                      </h3>
                    </div>
                  </div>

                  <div className="mt-5 space-y-2">
                    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                      <p className="text-xs font-bold text-slate-700">
                        DistilBERT
                      </p>

                      <p className="mt-1 text-[10px] text-slate-500">
                        768-dimensional representation
                      </p>
                    </div>

                    <div className="text-center text-slate-300">
                      ↓
                    </div>

                    <div className="rounded-xl border border-cyan-200 bg-cyan-50 px-4 py-3">
                      <p className="text-xs font-bold text-cyan-700">
                        Sparse Autoencoder
                      </p>

                      <p className="mt-1 text-[10px] text-slate-500">
                        2,048 learned features
                      </p>
                    </div>

                    <div className="text-center text-slate-300">
                      ↓
                    </div>

                    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                      <p className="text-xs font-bold text-slate-700">
                        Feature #850
                      </p>

                      <p className="mt-1 text-[10px] text-slate-500">
                        Observed activation
                      </p>
                    </div>
                  </div>
                </section>
              </div>
            </div>

            <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400">
                    Research Context
                  </p>

                  <h3 className="mt-1 text-base font-bold text-slate-950">
                    Sparse feature observability
                  </h3>
                </div>

                <div className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-2">
                  <Activity className="h-3.5 w-3.5 text-emerald-600" />

                  <span className="text-[10px] font-semibold text-emerald-700">
                    Production observation only
                  </span>
                </div>
              </div>

              <div className="mt-5 grid gap-4 md:grid-cols-3">
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    Encoder Representation
                  </p>

                  <p className="mt-2 text-lg font-bold text-slate-950">
                    768D
                  </p>

                  <p className="mt-1 text-[11px] leading-5 text-slate-500">
                    DistilBERT representation supplied to the sparse
                    autoencoder.
                  </p>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    Sparse Feature Space
                  </p>

                  <p className="mt-2 text-lg font-bold text-slate-950">
                    2,048
                  </p>

                  <p className="mt-1 text-[11px] leading-5 text-slate-500">
                    Expanded sparse representation generated by the SAE.
                  </p>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    Selected Feature
                  </p>

                  <p className="mt-2 text-lg font-bold text-slate-950">
                    #850
                  </p>

                  <p className="mt-1 text-[11px] leading-5 text-slate-500">
                    Activation exposed for interpretability analysis.
                  </p>
                </div>
              </div>
            </section>

            <footer className="mt-8 flex flex-col justify-between gap-3 border-t border-slate-200 py-6 text-[10px] text-slate-400 sm:flex-row">
              <span>
                Neural Feature Safety Platform
              </span>

              <span>
                DistilBERT · Sparse Autoencoder · Feature 850
              </span>
            </footer>
          </div>
        </main>
      </div>
    </div>
  );
}