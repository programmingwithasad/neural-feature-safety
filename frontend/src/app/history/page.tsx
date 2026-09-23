"use client";

import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  History as HistoryIcon,
  Trash2,
} from "lucide-react";
import Sidebar from "../../components/Sidebar";
import Topbar from "../../components/Topbar";

type HistoryItem = {
  id: string;
  text: string;
  harmful_probability: number;
  risk_level: string;
  decision: string;
  feature_850_activation?: number;
  device: string;
  timestamp: string;
};

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    const stored = localStorage.getItem("neural-feature-history");

    if (stored) {
      try {
        setHistory(JSON.parse(stored));
      } catch {
        setHistory([]);
      }
    }
  }, []);

  const clearHistory = () => {
    localStorage.removeItem("neural-feature-history");
    setHistory([]);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="flex min-h-screen">
        <Sidebar />

        <main className="min-w-0 flex-1">
          <Topbar page="Analysis History" />

          <div className="mx-auto max-w-[1500px] px-5 py-7 sm:px-7 lg:px-9">
            <section className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="absolute right-0 top-0 h-full w-[45%] bg-gradient-to-l from-cyan-50/80 to-transparent" />

              <div className="relative flex flex-col justify-between gap-6 px-7 py-8 sm:px-9 sm:py-10 lg:flex-row lg:items-center">
                <div>
                  <div className="inline-flex items-center gap-2 rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1.5">
                    <HistoryIcon className="h-3.5 w-3.5 text-cyan-600" />

                    <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-cyan-700">
                      System History
                    </span>
                  </div>

                  <h2 className="mt-5 text-3xl font-bold tracking-tight text-slate-950 sm:text-4xl">
                    Analysis{" "}
                    <span className="text-cyan-600">
                      History
                    </span>
                  </h2>

                  <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-500">
                    Review prompts analyzed during this browser session.
                  </p>
                </div>

                {history.length > 0 && (
                  <button
                    type="button"
                    onClick={clearHistory}
                    className="flex items-center justify-center gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-2.5 text-xs font-bold text-red-600 transition hover:bg-red-100"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    Clear History
                  </button>
                )}
              </div>
            </section>

            <section className="mt-6">
              {history.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-16 text-center shadow-sm">
                  <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-slate-50">
                    <Clock3 className="h-5 w-5 text-slate-400" />
                  </div>

                  <h3 className="mt-4 text-sm font-bold text-slate-950">
                    No analyses yet
                  </h3>

                  <p className="mx-auto mt-2 max-w-md text-xs leading-6 text-slate-500">
                    Run a prompt through the Analyzer and your results will
                    appear here automatically.
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {history.map((item) => {
                    const blocked = item.decision === "BLOCK";

                    return (
                      <div
                        key={item.id}
                        className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
                      >
                        <div className="flex flex-col gap-4 lg:flex-row lg:items-center">
                          <div className="flex min-w-0 flex-1 items-start gap-3">
                            <div
                              className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${
                                blocked
                                  ? "bg-red-50"
                                  : "bg-emerald-50"
                              }`}
                            >
                              {blocked ? (
                                <AlertTriangle className="h-4 w-4 text-red-600" />
                              ) : (
                                <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                              )}
                            </div>

                            <div className="min-w-0">
                              <p className="line-clamp-2 text-sm font-semibold text-slate-800">
                                {item.text}
                              </p>

                              <p className="mt-2 text-[10px] text-slate-400">
                                {new Date(
                                  item.timestamp
                                ).toLocaleString()}
                              </p>
                            </div>
                          </div>

                          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:w-[560px]">
                            <div className="rounded-xl bg-slate-50 p-3">
                              <p className="text-[9px] uppercase tracking-wider text-slate-400">
                                Harmful
                              </p>

                              <p className="mt-1 text-xs font-bold text-slate-800">
                                {(
                                  item.harmful_probability * 100
                                ).toFixed(2)}
                                %
                              </p>
                            </div>

                            <div className="rounded-xl bg-slate-50 p-3">
                              <p className="text-[9px] uppercase tracking-wider text-slate-400">
                                Risk
                              </p>

                              <p
                                className={`mt-1 text-xs font-bold ${
                                  blocked
                                    ? "text-red-600"
                                    : "text-emerald-600"
                                }`}
                              >
                                {item.risk_level}
                              </p>
                            </div>

                            <div className="rounded-xl bg-slate-50 p-3">
                              <p className="text-[9px] uppercase tracking-wider text-slate-400">
                                Feature 850
                              </p>

                              <p className="mt-1 font-mono text-xs font-bold text-cyan-600">
                                {typeof item.feature_850_activation === "number"
                                    ? item.feature_850_activation.toFixed(4)
                                    : "N/A"}
                              </p>
                            </div>

                            <div className="rounded-xl bg-slate-50 p-3">
                              <p className="text-[9px] uppercase tracking-wider text-slate-400">
                                Device
                              </p>

                              <p className="mt-1 text-xs font-bold text-slate-800">
                                {item.device}
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}