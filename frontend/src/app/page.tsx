"use client";

import { useState } from "react";
import {
  AlertTriangle,
  ArrowDown,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Gauge,
  Layers3,
  Loader2,
  MessageSquare,
  Shield,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import Sidebar from "../components/Sidebar";
import Topbar from "../components/Topbar";

type SafetyResult = {
  allowed: boolean;
  decision: string;
  risk_level: string;
  harmful_probability: number;
  unharmful_probability: number;
  threshold: number;
  feature_850_activation: number;
  device: string;
};

type GatewayResult = {
  response: string;
  input_safety: SafetyResult;
  output_safety: SafetyResult;
  path: string;
  recovery_attempts: number;
};

export default function Home() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<GatewayResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const sendMessage = async () => {
    const value = message.trim();

    if (!value || loading) {
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: value,
        }),
        cache: "no-store",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            data?.error ||
            "Safety Gateway request failed."
        );
      }

      setResult(data);
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : "Unable to connect to the Safety Gateway."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLTextAreaElement>
  ) => {
    if (event.ctrlKey && event.key === "Enter") {
      event.preventDefault();
      sendMessage();
    }
  };

  const input = result?.input_safety;
  const output = result?.output_safety;

  const inputPercentage =
    (input?.harmful_probability ?? 0) * 100;

  const outputPercentage =
    (output?.harmful_probability ?? 0) * 100;

  const inputBlocked = input?.decision === "BLOCK";
  const outputBlocked = output?.decision === "BLOCK";

  const formatPath = (path: string) => {
    return path
      .replaceAll("_", " ")
      .replace(/\b\w/g, (char) => char.toUpperCase());
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="flex min-h-screen">
        <Sidebar />

        <main className="min-w-0 flex-1">
          <Topbar page="Live Gateway" />

          <div className="mx-auto max-w-[1500px] px-5 py-7 sm:px-7 lg:px-9">
            <section className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="absolute right-0 top-0 h-full w-[45%] bg-gradient-to-l from-cyan-50/80 to-transparent" />

              <div className="relative grid gap-8 px-7 py-8 lg:grid-cols-[1fr_280px] lg:px-9 lg:py-10">
                <div>
                  <div className="inline-flex items-center gap-2 rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-cyan-500" />

                    <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-cyan-700">
                      Live Neural Safety Gateway
                    </span>
                  </div>

                  <h2 className="mt-5 max-w-3xl text-3xl font-bold leading-[1.15] tracking-tight text-slate-950 sm:text-4xl lg:text-[42px]">
                    Protected AI interaction with{" "}
                    <span className="text-cyan-600">
                      independent safety control.
                    </span>
                  </h2>

                  <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-500">
                    Every interaction passes through the Neural Feature
                    Safety detector before and after local Llama generation.
                    Unsafe inputs are controlled before generation, while
                    generated responses are independently checked before
                    delivery.
                  </p>
                </div>

                <div className="hidden items-center justify-center lg:flex">
                  <div className="relative flex h-44 w-44 items-center justify-center rounded-full border border-cyan-100 bg-cyan-50/60">
                    <div className="absolute h-32 w-32 rounded-full border border-cyan-200 bg-white" />

                    <div className="relative flex h-20 w-20 items-center justify-center rounded-2xl bg-slate-950 shadow-xl">
                      <ShieldCheck className="h-9 w-9 text-cyan-300" />
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400">
                      Safety Encoder
                    </p>

                    <p className="mt-2 text-lg font-bold text-slate-950">
                      DistilBERT
                    </p>

                    <p className="mt-1 text-[11px] leading-5 text-slate-500">
                      768-dimensional representation
                    </p>
                  </div>

                  <BrainCircuit className="h-5 w-5 text-cyan-600" />
                </div>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400">
                      Sparse Capacity
                    </p>

                    <p className="mt-2 text-lg font-bold text-slate-950">
                      2,048
                    </p>

                    <p className="mt-1 text-[11px] leading-5 text-slate-500">
                      Learned sparse neural features
                    </p>
                  </div>

                  <Layers3 className="h-5 w-5 text-violet-600" />
                </div>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400">
                      Safety Threshold
                    </p>

                    <p className="mt-2 text-lg font-bold text-slate-950">
                      0.50
                    </p>

                    <p className="mt-1 text-[11px] leading-5 text-slate-500">
                      Production decision boundary
                    </p>
                  </div>

                  <Gauge className="h-5 w-5 text-cyan-600" />
                </div>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400">
                      Local Generation
                    </p>

                    <p className="mt-2 text-lg font-bold text-slate-950">
                      Llama 3.2 3B
                    </p>

                    <p className="mt-1 text-[11px] leading-5 text-slate-500">
                      Local model with safety adapter
                    </p>
                  </div>

                  <Cpu className="h-5 w-5 text-emerald-600" />
                </div>
              </div>
            </section>

            <div className="mt-6 grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
              <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
                <div className="border-b border-slate-200 px-6 py-5">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-950">
                      <MessageSquare className="h-5 w-5 text-cyan-300" />
                    </div>

                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400">
                        Gateway Chat
                      </p>

                      <h2 className="mt-1 text-base font-bold text-slate-950">
                        Talk to the protected local model
                      </h2>
                    </div>
                  </div>
                </div>

                <div className="p-6">
                  <textarea
                    value={message}
                    onChange={(event) => setMessage(event.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={loading}
                    placeholder="Enter a message ..."
                    className="min-h-[220px] w-full resize-y rounded-xl border border-slate-200 bg-slate-50 px-5 py-4 text-sm leading-6 text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-cyan-400 focus:bg-white focus:ring-4 focus:ring-cyan-50 disabled:cursor-not-allowed disabled:opacity-60"
                  />

                  <div className="mt-4 flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
                    <div className="text-[11px] text-slate-400">
                      Press{" "}
                      <span className="font-semibold text-slate-600">
                        Ctrl + Enter
                      </span>{" "}
                      to send
                    </div>

                    <button
                      type="button"
                      onClick={sendMessage}
                      disabled={!message.trim() || loading}
                      className="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-950 px-6 py-3 text-xs font-bold text-white transition hover:bg-cyan-600 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin" />
                          Gateway Processing...
                        </>
                      ) : (
                        <>
                          <Sparkles className="h-4 w-4" />
                          Send to Gateway
                        </>
                      )}
                    </button>
                  </div>

                  {error && (
                    <div className="mt-5 rounded-xl border border-red-200 bg-red-50 p-4">
                      <div className="flex items-center gap-3">
                        <AlertTriangle className="h-5 w-5 text-red-600" />

                        <div>
                          <p className="text-sm font-bold text-red-700">
                            Gateway unavailable
                          </p>

                          <p className="mt-1 text-xs text-red-600">
                            {error}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {result && (
                    <div className="mt-6 border-t border-slate-200 pt-6">
                      <div className="mb-3 flex items-center justify-between">
                        <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400">
                          Final Response
                        </p>

                        <div className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5">
                          <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />

                          <span className="text-[10px] font-bold text-emerald-700">
                            Output Verified
                          </span>
                        </div>
                      </div>

                      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
                        <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">
                          {result.response}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </section>

              <div className="space-y-6">
                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400">
                        Gateway Path
                      </p>

                      <h2 className="mt-1 text-base font-bold text-slate-950">
                        Safety execution
                      </h2>
                    </div>

                    <Shield className="h-5 w-5 text-cyan-600" />
                  </div>

                  <div className="mt-6 space-y-2">
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                      <p className="text-xs font-bold text-slate-800">
                        User Input
                      </p>

                      <p className="mt-1 text-[10px] text-slate-500">
                        Message received by gateway
                      </p>
                    </div>

                    <ArrowDown className="mx-auto h-4 w-4 text-slate-300" />

                    <div
                      className={`rounded-xl border p-3 ${
                        inputBlocked
                          ? "border-red-200 bg-red-50"
                          : input
                            ? "border-emerald-200 bg-emerald-50"
                            : "border-slate-200 bg-slate-50"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <p className="text-xs font-bold text-slate-800">
                          Input Safety
                        </p>

                        {input && (
                          <span
                            className={`text-[10px] font-bold ${
                              inputBlocked
                                ? "text-red-700"
                                : "text-emerald-700"
                            }`}
                          >
                            {input.decision}
                          </span>
                        )}
                      </div>

                      <p className="mt-1 text-[10px] text-slate-500">
                        DistilBERT → SAE → Production Classifier
                      </p>
                    </div>

                    <ArrowDown className="mx-auto h-4 w-4 text-slate-300" />

                    <div className="rounded-xl border border-violet-200 bg-violet-50 p-3">
                      <p className="text-xs font-bold text-slate-800">
                        Controller + Llama 3.2 3B
                      </p>

                      <p className="mt-1 text-[10px] text-slate-500">
                        Local generation and recovery logic
                      </p>
                    </div>

                    <ArrowDown className="mx-auto h-4 w-4 text-slate-300" />

                    <div
                      className={`rounded-xl border p-3 ${
                        outputBlocked
                          ? "border-red-200 bg-red-50"
                          : output
                            ? "border-emerald-200 bg-emerald-50"
                            : "border-slate-200 bg-slate-50"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <p className="text-xs font-bold text-slate-800">
                          Output Safety
                        </p>

                        {output && (
                          <span
                            className={`text-[10px] font-bold ${
                              outputBlocked
                                ? "text-red-700"
                                : "text-emerald-700"
                            }`}
                          >
                            {output.decision}
                          </span>
                        )}
                      </div>

                      <p className="mt-1 text-[10px] text-slate-500">
                        Generated response checked before delivery
                      </p>
                    </div>
                  </div>
                </section>

                {result && (
                  <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400">
                          Safety Telemetry
                        </p>

                        <h2 className="mt-1 text-base font-bold text-slate-950">
                          Gateway analysis
                        </h2>
                      </div>

                      <ShieldCheck className="h-5 w-5 text-cyan-600" />
                    </div>

                    <div className="mt-5 grid gap-3 sm:grid-cols-2">
                      <div
                        className={`rounded-xl border p-4 ${
                          inputBlocked
                            ? "border-red-200 bg-red-50"
                            : "border-emerald-200 bg-emerald-50"
                        }`}
                      >
                        <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
                          Input Decision
                        </p>

                        <p
                          className={`mt-2 text-lg font-bold ${
                            inputBlocked
                              ? "text-red-700"
                              : "text-emerald-700"
                          }`}
                        >
                          {input?.decision}
                        </p>
                      </div>

                      <div
                        className={`rounded-xl border p-4 ${
                          outputBlocked
                            ? "border-red-200 bg-red-50"
                            : "border-emerald-200 bg-emerald-50"
                        }`}
                      >
                        <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
                          Output Decision
                        </p>

                        <p
                          className={`mt-2 text-lg font-bold ${
                            outputBlocked
                              ? "text-red-700"
                              : "text-emerald-700"
                          }`}
                        >
                          {output?.decision}
                        </p>
                      </div>
                    </div>

                    <div className="mt-3 grid gap-3 sm:grid-cols-2">
                      <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-500">
                            Input harmful probability
                          </span>

                          <span className="font-bold text-slate-800">
                            {inputPercentage.toFixed(2)}%
                          </span>
                        </div>

                        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-200">
                          <div
                            className="h-full rounded-full bg-red-400"
                            style={{
                              width: `${Math.min(inputPercentage, 100)}%`,
                            }}
                          />
                        </div>
                      </div>

                      <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-500">
                            Output harmful probability
                          </span>

                          <span className="font-bold text-slate-800">
                            {outputPercentage.toFixed(2)}%
                          </span>
                        </div>

                        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-200">
                          <div
                            className="h-full rounded-full bg-orange-400"
                            style={{
                              width: `${Math.min(outputPercentage, 100)}%`,
                            }}
                          />
                        </div>
                      </div>
                    </div>

                    <div className="mt-3 grid gap-3 sm:grid-cols-3">
                      <div className="rounded-xl border border-slate-200 bg-white p-4">
                        <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
                          Gateway Path
                        </p>

                        <p className="mt-2 text-sm font-bold text-slate-900">
                          {formatPath(result.path)}
                        </p>
                      </div>

                      <div className="rounded-xl border border-slate-200 bg-white p-4">
                        <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
                          Recovery Attempts
                        </p>

                        <p className="mt-2 text-lg font-bold text-slate-900">
                          {result.recovery_attempts}
                        </p>
                      </div>

                      <div className="rounded-xl border border-slate-200 bg-white p-4">
                        <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
                          Runtime
                        </p>

                        <p className="mt-2 text-sm font-bold text-slate-900">
                          {input?.device ?? "Unknown"}
                        </p>
                      </div>
                    </div>

                    <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
                            Feature 850
                          </p>

                          <p className="mt-1 text-[10px] text-slate-400">
                            Observational activation
                          </p>
                        </div>

                        <div className="text-right">
                          <p className="font-mono text-sm font-bold text-cyan-600">
                            {input?.feature_850_activation.toFixed(4)}
                          </p>

                          <p className="text-[9px] text-slate-400">
                            input
                          </p>
                        </div>
                      </div>
                    </div>
                  </section>
                )}
              </div>
            </div>

            <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-slate-400">
                    Neural Architecture
                  </p>

                  <h2 className="mt-1 text-base font-bold text-slate-950">
                    Complete safety gateway
                  </h2>
                </div>

                <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-[10px] font-bold text-emerald-700">
                  Production Path
                </span>
              </div>

              <div className="mt-6 grid gap-3 md:grid-cols-5">
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <BrainCircuit className="h-5 w-5 text-cyan-600" />

                  <p className="mt-3 text-xs font-bold text-slate-800">
                    DistilBERT
                  </p>

                  <p className="mt-1 text-[10px] text-slate-500">
                    768D representation
                  </p>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <Layers3 className="h-5 w-5 text-violet-600" />

                  <p className="mt-3 text-xs font-bold text-slate-800">
                    Sparse Autoencoder
                  </p>

                  <p className="mt-1 text-[10px] text-slate-500">
                    2,048 sparse features
                  </p>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <Shield className="h-5 w-5 text-cyan-600" />

                  <p className="mt-3 text-xs font-bold text-slate-800">
                    Safety Classifier
                  </p>

                  <p className="mt-1 text-[10px] text-slate-500">
                    Harmful probability
                  </p>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <Sparkles className="h-5 w-5 text-violet-600" />

                  <p className="mt-3 text-xs font-bold text-slate-800">
                    Llama 3.2 3B
                  </p>

                  <p className="mt-1 text-[10px] text-slate-500">
                    Local response generation
                  </p>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <ShieldCheck className="h-5 w-5 text-emerald-600" />

                  <p className="mt-3 text-xs font-bold text-slate-800">
                    Output Safety
                  </p>

                  <p className="mt-1 text-[10px] text-slate-500">
                    Final response verification
                  </p>
                </div>
              </div>
            </section>

            <section className="mt-6 grid gap-4 md:grid-cols-3">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400">
                  Input Protection
                </p>

                <p className="mt-2 text-lg font-bold text-slate-950">
                  Pre-Generation
                </p>

                <p className="mt-1 text-[11px] leading-5 text-slate-500">
                  The independent safety detector evaluates the user input
                  before normal Llama generation occurs.
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400">
                  Local Intelligence
                </p>

                <p className="mt-2 text-lg font-bold text-slate-950">
                  Llama 3.2 3B
                </p>

                <p className="mt-1 text-[11px] leading-5 text-slate-500">
                  Local generation using the project-trained safety adapter.
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-400">
                  Output Protection
                </p>

                <p className="mt-2 text-lg font-bold text-slate-950">
                  Post-Generation
                </p>

                <p className="mt-1 text-[11px] leading-5 text-slate-500">
                  Generated responses are independently evaluated before
                  being returned to the user.
                </p>
              </div>
            </section>

            <footer className="mt-8 flex flex-col justify-between gap-3 border-t border-slate-200 py-6 text-[10px] text-slate-400 sm:flex-row">
              <span>Neural Feature Safety Platform</span>

              <span>
                DistilBERT · Sparse Autoencoder · Safety Classifier · Llama
                3.2 3B
              </span>
            </footer>
          </div>
        </main>
      </div>
    </div>
  );
}