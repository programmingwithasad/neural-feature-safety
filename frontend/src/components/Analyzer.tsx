"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Cpu,
  Gauge,
  Info,
  Layers3,
  LockKeyhole,
  ShieldCheck,
  ShieldAlert,
  Sparkles,
  Target,
  Zap
} from "lucide-react";

type Feature = {
  feature: number;
  activation: number;
  classifier_weight: number;
  contribution: number;
};

type Evidence = {
  harmful_probability: number;
  unharmful_probability: number;
  risk_level: string;
  decision: string;
  threshold: number;
  feature_850_activation: number;
  device: string;
  feature_space: {
    input_dimensions: number;
    latent_dimensions: number;
  };
  classifier: {
    type: string;
    class_weight: string;
    intercept: number;
  };
  top_activated_features: Feature[];
  top_positive_contributions: Feature[];
  top_negative_contributions: Feature[];
};

type ExplanationResult = {
  evidence: Evidence;
  llama_explanation: string;
  explanation_source: string;
  generation_time: number;
  explanation_output_safety: {
    allowed: boolean;
    decision: string;
    risk_level: string;
    harmful_probability: number;
    unharmful_probability: number;
    threshold: number;
  };
};

function formatProbability(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value: number) {
  return value.toFixed(3);
}

function formatContribution(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function getDecisionStyles(decision: string) {
  if (decision === "BLOCK") {
    return {
      container:
        "border-red-200 bg-red-50 text-red-700   ",
      icon: ShieldAlert,
      label: "BLOCKED"
    };
  }

  return {
    container:
      "border-emerald-200 bg-emerald-50 text-emerald-700   ",
    icon: ShieldCheck,
    label: "ALLOWED"
  };
}

function ContributionRow({
  item,
  max,
  positive
}: {
  item: Feature;
  max: number;
  positive: boolean;
}) {
  const width = Math.max(
    5,
    Math.min(100, (Math.abs(item.contribution) / max) * 100)
  );

  return (
    <div className="group rounded-xl border border-slate-200/80 bg-white/70 p-3 transition hover:border-slate-300 hover:bg-white ">
      <div className="mb-2 flex items-center justify-between gap-4">
        <div className="flex min-w-0 items-center gap-2">
          <span className="rounded-md bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-700 ">
            F{item.feature}
          </span>
          <span className="text-xs text-slate-500">
            activation {formatNumber(item.activation)}
          </span>
        </div>

        <span
          className={`shrink-0 text-sm font-bold ${
            positive
              ? "text-red-600 "
              : "text-slate-600 "
          }`}
        >
          {formatContribution(item.contribution)}
        </span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-slate-100 ">
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            positive
              ? "bg-gradient-to-r from-orange-400 to-red-500"
              : "bg-gradient-to-r from-slate-300 to-slate-500  "
          }`}
          style={{ width: `${width}%` }}
        />
      </div>

      <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
        <span>Classifier weight {formatNumber(item.classifier_weight)}</span>
        <span>{positive ? "Toward harmful" : "Away from harmful"}</span>
      </div>
    </div>
  );
}

function ActivationRow({
  item,
  max
}: {
  item: Feature;
  max: number;
}) {
  const width = Math.max(
    4,
    Math.min(100, (item.activation / max) * 100)
  );

  return (
    <div className="rounded-xl border border-slate-200/80 bg-white/70 p-3 ">
      <div className="mb-2 flex items-center justify-between">
        <span className="rounded-md bg-cyan-50 px-2 py-1 text-xs font-semibold text-cyan-700 ">
          Feature {item.feature}
        </span>
        <span className="text-sm font-semibold text-slate-700 ">
          {formatNumber(item.activation)}
        </span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-slate-100 ">
        <div
          className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-blue-500 transition-all duration-700"
          style={{ width: `${width}%` }}
        />
      </div>

      <div className="mt-2 flex justify-between text-[11px] text-slate-400">
        <span>Classifier weight {formatNumber(item.classifier_weight)}</span>
        <span>{formatContribution(item.contribution)}</span>
      </div>
    </div>
  );
}

export default function Analyzer() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<ExplanationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPositive, setShowPositive] = useState(true);
  const [showNegative, setShowNegative] = useState(true);
  const [showActivations, setShowActivations] = useState(true);

  const evidence = result?.evidence;

  const positiveMax = useMemo(() => {
    if (!evidence?.top_positive_contributions.length) {
      return 1;
    }

    return Math.max(
      1,
      ...evidence.top_positive_contributions.map((item) =>
        Math.abs(item.contribution)
      )
    );
  }, [evidence]);

  const negativeMax = useMemo(() => {
    if (!evidence?.top_negative_contributions.length) {
      return 1;
    }

    return Math.max(
      1,
      ...evidence.top_negative_contributions.map((item) =>
        Math.abs(item.contribution)
      )
    );
  }, [evidence]);

  const activationMax = useMemo(() => {
    if (!evidence?.top_activated_features.length) {
      return 1;
    }

    return Math.max(
      1,
      ...evidence.top_activated_features.map((item) => item.activation)
    );
  }, [evidence]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();

        if (!loading && text.trim()) {
          handleAnalyze();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [text, loading]);

  async function handleAnalyze() {
    const prompt = text.trim();

    if (!prompt) {
      setError("Enter a prompt to analyze.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/explain/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          text: prompt
        }),
        cache: "no-store"
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            data?.error ||
            "Unable to analyze the prompt."
        );
      }

      setResult(data);

      const historyItem = {
      id: Date.now(),
      text: prompt,
      decision: data.evidence?.decision,
      harmful_probability: data.evidence?.harmful_probability,
      unharmful_probability: data.evidence?.unharmful_probability,
      risk_level: data.evidence?.risk_level,
      threshold: data.evidence?.threshold,
      feature_850_activation: data.evidence?.feature_850_activation,
      device: data.evidence?.device,
      timestamp: new Date().toISOString()
      };

      try {
        const existing = JSON.parse(
          localStorage.getItem("neural-feature-history") || "[]"
        );

        localStorage.setItem(
          "neural-feature-history",
          JSON.stringify([
            historyItem,
            ...existing.filter(
              (item: { text?: string }) => item.text !== prompt
            )
          ].slice(0, 20))
        );
      } catch {
        localStorage.removeItem("neural-feature-history");
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to analyze the prompt."
      );
    } finally {
      setLoading(false);
    }
  }

  function loadExample(example: string) {
    setText(example);
    setResult(null);
    setError("");
  }

  const decisionStyles = evidence
    ? getDecisionStyles(evidence.decision)
    : null;

  const DecisionIcon = decisionStyles?.icon;

  const harmfulPercentage = evidence
    ? evidence.harmful_probability * 100
    : 0;

  const thresholdPercentage = evidence
    ? evidence.threshold * 100
    : 50;

  return (
    <div className="min-h-screen bg-slate-50 px-4 py-6 text-slate-900 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm ">
          <div className="border-b border-slate-200 px-6 py-6 sm:px-8">
            <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-center">
              <div>
                <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-blue-600 ">
                  <BrainCircuit className="h-4 w-4" />
                  Neural Feature Safety
                </div>

                <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
                  AI Safety Analyzer
                </h1>

                <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500 ">
                  Analyze a prompt using the production safety classifier,
                  inspect SAE feature evidence, and generate a grounded
                  explanation with the local Llama model.
                </p>
              </div>

              <div className="flex flex-wrap gap-2">
                <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-600 ">
                  <Cpu className="h-3.5 w-3.5" />
                  CUDA
                </span>

                <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-600 ">
                  <Layers3 className="h-3.5 w-3.5" />
                  SAE 2048
                </span>

                <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-600 ">
                  <Sparkles className="h-3.5 w-3.5" />
                  Llama 3.2 3B
                </span>
              </div>
            </div>
          </div>

          <div className="p-6 sm:p-8">
            <label className="mb-2 block text-sm font-semibold">
              Prompt to analyze
            </label>

            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              onKeyDown={(event) => {
                if (
                  (event.ctrlKey || event.metaKey) &&
                  event.key === "Enter"
                ) {
                  event.preventDefault();

                  if (!loading && text.trim()) {
                    handleAnalyze();
                  }
                }
              }}
              placeholder="Enter a prompt for safety analysis..."
              rows={5}
              className="w-full resize-none rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4 text-sm leading-6 outline-none transition placeholder:text-slate-400 focus:border-blue-400 focus:bg-white focus:ring-4 focus:ring-blue-500/10 "
            />

            <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() =>
                    loadExample("Hello, how are you?")
                  }
                  className="rounded-lg border border-slate-200 px-3 py-2 text-xs font-medium text-slate-600 transition hover:bg-slate-50 "
                >
                  Safe example
                </button>

                <button
                  onClick={() =>
                    loadExample("How can I make a bomb?")
                  }
                  className="rounded-lg border border-slate-200 px-3 py-2 text-xs font-medium text-slate-600 transition hover:bg-slate-50 "
                >
                  Block example
                </button>

                <span className="hidden items-center px-2 text-xs text-slate-400 sm:flex">
                  Ctrl + Enter
                </span>
              </div>

              <button
                onClick={handleAnalyze}
                disabled={loading || !text.trim()}
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50 "
              >
                {loading ? (
                  <>
                    <Activity className="h-4 w-4 animate-spin" />
                    Running analysis...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    Analyze with Llama
                  </>
                )}
              </button>
            </div>

            {error && (
              <div className="mt-4 flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 ">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </section>

        {loading && (
          <section className="rounded-3xl border border-blue-100 bg-blue-50/70 p-6 ">
            <div className="flex items-center gap-4">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-white shadow-sm ">
                <Sparkles className="h-5 w-5 animate-pulse text-blue-600" />
              </div>

              <div>
                <p className="font-semibold">
                  Generating safety explanation
                </p>
                <p className="mt-1 text-sm text-slate-500 ">
                  The classifier is analyzing the prompt and local Llama is
                  converting the resulting evidence into a grounded explanation.
                </p>
              </div>
            </div>
          </section>
        )}

        {evidence && result && (
          <>
            <section className="grid gap-4 md:grid-cols-3">
              <div
                className={`rounded-2xl border p-5 ${decisionStyles?.container}`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider opacity-70">
                    Safety decision
                  </span>
                  {DecisionIcon && <DecisionIcon className="h-5 w-5" />}
                </div>

                <div className="mt-4 flex items-end justify-between">
                  <div>
                    <div className="text-3xl font-bold">
                      {decisionStyles?.label}
                    </div>
                    <div className="mt-1 text-xs opacity-70">
                      {evidence.risk_level}
                    </div>
                  </div>

                  <Target className="h-9 w-9 opacity-20" />
                </div>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Harmful probability
                  </span>
                  <Gauge className="h-5 w-5 text-slate-400" />
                </div>

                <div className="mt-3 text-3xl font-bold">
                  {formatProbability(evidence.harmful_probability)}
                </div>

                <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100 ">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-amber-400 to-red-500"
                    style={{
                      width: `${harmfulPercentage}%`
                    }}
                  />
                </div>

                <div className="mt-2 flex justify-between text-xs text-slate-400">
                  <span>0%</span>
                  <span>Threshold {formatProbability(evidence.threshold)}</span>
                  <span>100%</span>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Model evidence
                  </span>
                  <Zap className="h-5 w-5 text-slate-400" />
                </div>

                <div className="mt-4 grid grid-cols-2 gap-3">
                  <div className="rounded-xl bg-slate-50 p-3 ">
                    <p className="text-xs text-slate-400">Input</p>
                    <p className="mt-1 text-lg font-bold">
                      {evidence.feature_space.input_dimensions}
                    </p>
                    <p className="text-[11px] text-slate-400">
                      dimensions
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-50 p-3 ">
                    <p className="text-xs text-slate-400">Latent</p>
                    <p className="mt-1 text-lg font-bold">
                      {evidence.feature_space.latent_dimensions}
                    </p>
                    <p className="text-[11px] text-slate-400">
                      SAE features
                    </p>
                  </div>
                </div>
              </div>
            </section>

            <section className="overflow-hidden rounded-3xl border border-blue-100 bg-gradient-to-br from-blue-50 via-white to-indigo-50 shadow-sm ">
              <div className="border-b border-blue-100 px-6 py-5 sm:px-8">
                <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
                  <div>
                    <div className="flex items-center gap-2">
                      <Sparkles className="h-5 w-5 text-blue-600 " />
                      <h2 className="text-lg font-bold">
                        AI-Generated Safety Explanation
                      </h2>
                    </div>

                    <p className="mt-1 text-xs text-slate-500 ">
                      Generated by local Llama from Neural Feature Safety evidence
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <span className="inline-flex items-center gap-1.5 rounded-full border border-blue-200 bg-white/80 px-3 py-1.5 text-xs font-medium text-blue-700 ">
                      <BrainCircuit className="h-3.5 w-3.5" />
                      Llama 3.2 3B
                    </span>

                    <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700 ">
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      Output checked
                    </span>
                  </div>
                </div>
              </div>

              <div className="px-6 py-6 sm:px-8">
                <div className="rounded-2xl border border-white/80 bg-white/80 p-5 text-sm leading-7 text-slate-700 shadow-sm ">
                  {result.llama_explanation}
                </div>

                <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-400">
                  <span>
                    Source: {result.explanation_source}
                  </span>
                  <span>
                    Generation: {result.generation_time.toFixed(2)}s
                  </span>
                  <span>
                    Safety decision remains classifier-controlled
                  </span>
                </div>
              </div>
            </section>

            <section className="rounded-3xl border border-slate-200 bg-white shadow-sm ">
              <div className="border-b border-slate-200 px-6 py-5 sm:px-8">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <Activity className="h-5 w-5 text-slate-500" />
                      <h2 className="text-lg font-bold">
                        Classifier Contribution Map
                      </h2>
                    </div>

                    <p className="mt-1 text-xs text-slate-500 ">
                      Direction and magnitude of displayed SAE evidence in the
                      classifier decision
                    </p>
                  </div>

                  <div className="hidden items-center gap-4 text-xs sm:flex">
                    <span className="flex items-center gap-2 text-red-600 ">
                      <span className="h-2 w-2 rounded-full bg-red-500" />
                      Toward harmful
                    </span>
                    <span className="flex items-center gap-2 text-slate-500">
                      <span className="h-2 w-2 rounded-full bg-slate-400" />
                      Away from harmful
                    </span>
                  </div>
                </div>
              </div>

              <div className="grid gap-6 p-6 lg:grid-cols-2 sm:p-8">
                <div>
                  <div className="mb-4 flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold">
                        Toward harmful class
                      </h3>
                      <p className="mt-1 text-xs text-slate-400">
                        Positive classifier contributions
                      </p>
                    </div>

                    <span className="rounded-full bg-red-50 px-3 py-1 text-xs font-semibold text-red-600 ">
                      {evidence.top_positive_contributions.length} features
                    </span>
                  </div>

                  <div className="space-y-3">
                    {showPositive &&
                      evidence.top_positive_contributions.map((item) => (
                        <ContributionRow
                          key={`positive-${item.feature}`}
                          item={item}
                          max={positiveMax}
                          positive
                        />
                      ))}
                  </div>

                  <button
                    onClick={() => setShowPositive((value) => !value)}
                    className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 py-2.5 text-xs font-semibold text-slate-500 transition hover:bg-slate-50 "
                  >
                    {showPositive ? (
                      <>
                        Collapse
                        <ChevronUp className="h-4 w-4" />
                      </>
                    ) : (
                      <>
                        Expand
                        <ChevronDown className="h-4 w-4" />
                      </>
                    )}
                  </button>
                </div>

                <div>
                  <div className="mb-4 flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold">
                        Away from harmful class
                      </h3>
                      <p className="mt-1 text-xs text-slate-400">
                        Negative classifier contributions
                      </p>
                    </div>

                    <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600 ">
                      {evidence.top_negative_contributions.length} features
                    </span>
                  </div>

                  <div className="space-y-3">
                    {showNegative &&
                      evidence.top_negative_contributions.map((item) => (
                        <ContributionRow
                          key={`negative-${item.feature}`}
                          item={item}
                          max={negativeMax}
                          positive={false}
                        />
                      ))}
                  </div>

                  <button
                    onClick={() => setShowNegative((value) => !value)}
                    className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 py-2.5 text-xs font-semibold text-slate-500 transition hover:bg-slate-50 "
                  >
                    {showNegative ? (
                      <>
                        Collapse
                        <ChevronUp className="h-4 w-4" />
                      </>
                    ) : (
                      <>
                        Expand
                        <ChevronDown className="h-4 w-4" />
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="border-t border-slate-200 px-6 py-4 sm:px-8">
                <div className="flex items-start gap-3">
                  <Info className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
                  <p className="text-xs leading-5 text-slate-500 ">
                    A feature&apos;s activation and classifier contribution
                    are different quantities. Contribution indicates the
                    direction and magnitude of the displayed feature&apos;s
                    effect on the logistic classifier score. It does not
                    establish a causal or human-interpretable meaning for the
                    SAE feature.
                  </p>
                </div>
              </div>
            </section>

            <section className="rounded-3xl border border-slate-200 bg-white shadow-sm ">
              <button
                onClick={() => setShowActivations((value) => !value)}
                className="flex w-full items-center justify-between px-6 py-5 text-left sm:px-8"
              >
                <div>
                  <div className="flex items-center gap-2">
                    <Layers3 className="h-5 w-5 text-cyan-500" />
                    <h2 className="text-lg font-bold">
                      Top SAE Activations
                    </h2>
                  </div>

                  <p className="mt-1 text-xs text-slate-500 ">
                    Highest activated features within the 2048-dimensional
                    latent space
                  </p>
                </div>

                {showActivations ? (
                  <ChevronUp className="h-5 w-5 text-slate-400" />
                ) : (
                  <ChevronDown className="h-5 w-5 text-slate-400" />
                )}
              </button>

              {showActivations && (
                <div className="grid gap-3 border-t border-slate-200 p-6 sm:grid-cols-2 lg:grid-cols-4 sm:p-8">
                  {evidence.top_activated_features.map((item) => (
                    <ActivationRow
                      key={item.feature}
                      item={item}
                      max={activationMax}
                    />
                  ))}
                </div>
              )}
            </section>

            <section className="grid gap-4 lg:grid-cols-3">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ">
                <div className="flex items-center gap-2">
                  <LockKeyhole className="h-4 w-4 text-slate-500" />
                  <span className="text-sm font-semibold">
                    Feature 850
                  </span>
                </div>

                <div className="mt-4 flex items-end justify-between">
                  <span className="text-3xl font-bold">
                    {formatNumber(evidence.feature_850_activation)}
                  </span>

                  <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[11px] font-semibold text-amber-700 ">
                    Observational
                  </span>
                </div>

                <p className="mt-3 text-xs leading-5 text-slate-500 ">
                  Feature 850 is displayed for research observation only and
                  is not treated as an independent safety control.
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ">
                <div className="flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-slate-500" />
                  <span className="text-sm font-semibold">
                    Inference device
                  </span>
                </div>

                <div className="mt-4 text-3xl font-bold">
                  {evidence.device.toUpperCase()}
                </div>

                <p className="mt-3 text-xs leading-5 text-slate-500 ">
                  DistilBERT and the safety analysis pipeline are running on
                  the detected inference device.
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ">
                <div className="flex items-center gap-2">
                  <Target className="h-4 w-4 text-slate-500" />
                  <span className="text-sm font-semibold">
                    Decision threshold
                  </span>
                </div>

                <div className="mt-4 text-3xl font-bold">
                  {formatProbability(evidence.threshold)}
                </div>

                <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100 ">
                  <div
                    className="h-full rounded-full bg-slate-700 "
                    style={{
                      width: `${thresholdPercentage}%`
                    }}
                  />
                </div>

                <p className="mt-3 text-xs leading-5 text-slate-500 ">
                  The production classifier uses this threshold for the
                  documented safety decision.
                </p>
              </div>
            </section>

            <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
              <div className="mb-7">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="h-5 w-5 text-blue-500" />
                  <h2 className="text-lg font-bold">
                    Safety Analysis Pipeline
                  </h2>
                </div>

                <p className="mt-1 text-xs text-slate-500 ">
                  The safety decision is produced before Llama generates the
                  natural-language explanation.
                </p>
              </div>

              <div className="grid gap-3 md:grid-cols-5">
                {[
                  {
                    icon: BrainCircuit,
                    title: "Prompt",
                    value: "User input"
                  },
                  {
                    icon: Cpu,
                    title: "DistilBERT",
                    value: "768 dimensions"
                  },
                  {
                    icon: Layers3,
                    title: "SAE",
                    value: "2048 features"
                  },
                  {
                    icon: Target,
                    title: "Classifier",
                    value: "Logistic Regression"
                  },
                  {
                    icon: Sparkles,
                    title: "Llama",
                    value: "Explanation writer"
                  }
                ].map((step, index) => {
                  const Icon = step.icon;

                  return (
                    <div
                      key={step.title}
                      className="relative rounded-2xl border border-slate-200 bg-slate-50 p-4 "
                    >
                      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-white shadow-sm ">
                        <Icon className="h-4 w-4 text-blue-500" />
                      </div>

                      <p className="mt-4 text-sm font-semibold">
                        {step.title}
                      </p>

                      <p className="mt-1 text-xs text-slate-400">
                        {step.value}
                      </p>

                      {index < 4 && (
                        <div className="absolute right-[-10px] top-1/2 hidden h-px w-5 bg-slate-300 md:block " />
                      )}
                    </div>
                  );
                })}
              </div>
            </section>
          </>
        )}
      </div>
    </div>
  );
}