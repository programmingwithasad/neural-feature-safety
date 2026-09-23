import {
  Activity,
  BarChart3,
  CheckCircle2,
  Target,
  TrendingUp,
} from "lucide-react";
import Sidebar from "../../components/Sidebar";
import Topbar from "../../components/Topbar";

const baseline = {
  accuracy: 80.4591,
  precision: 80.5797,
  recall: 73.7401,
  f1: 77.0083,
};

const intervention = {
  accuracy: 80.93,
  precision: 81.8047,
  recall: 73.3422,
  f1: 77.3427,
};

const confusion = {
  tp: 556,
  tn: 811,
  fp: 134,
  fn: 198,
};

const interventionConfusion = {
  tp: 553,
  tn: 822,
  fp: 123,
  fn: 201,
};

export default function EvaluationPage() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="flex min-h-screen">
        <Sidebar />

        <main className="min-w-0 flex-1">
          <Topbar page="Evaluation" />

          <div className="mx-auto max-w-[1500px] px-5 py-7 sm:px-7 lg:px-9">
            <section className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="absolute right-0 top-0 h-full w-[45%] bg-gradient-to-l from-emerald-50/80 to-transparent" />

              <div className="relative px-7 py-8 sm:px-9 sm:py-10">
                <div className="inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5">
                  <BarChart3 className="h-3.5 w-3.5 text-emerald-600" />

                  <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-emerald-700">
                    Model Evaluation
                  </span>
                </div>

                <h2 className="mt-5 text-3xl font-bold tracking-tight text-slate-950 sm:text-4xl">
                  Safety Model{" "}
                  <span className="text-emerald-600">
                    Evaluation
                  </span>
                </h2>

                <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-500">
                  Held-out evaluation metrics for the trained safety
                  classifier and the Feature 850 research intervention.
                </p>
              </div>
            </section>

            <section className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {[
                ["Accuracy", baseline.accuracy],
                ["Precision", baseline.precision],
                ["Recall", baseline.recall],
                ["F1 Score", baseline.f1],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
                >
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    {label}
                  </p>

                  <p className="mt-3 text-2xl font-bold text-slate-950">
                    {Number(value).toFixed(2)}%
                  </p>

                  <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-100">
                    <div
                      className="h-full rounded-full bg-cyan-500"
                      style={{ width: `${Number(value)}%` }}
                    />
                  </div>
                </div>
              ))}
            </section>

            <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-50">
                  <TrendingUp className="h-5 w-5 text-cyan-600" />
                </div>

                <div>
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    Comparison
                  </p>

                  <h3 className="mt-1 text-base font-bold text-slate-950">
                    Baseline vs Feature 850 intervention
                  </h3>
                </div>
              </div>

              <div className="mt-6 overflow-x-auto">
                <table className="w-full min-w-[600px] text-left">
                  <thead>
                    <tr className="border-b border-slate-200 text-[10px] uppercase tracking-wider text-slate-400">
                      <th className="px-4 py-3 font-bold">
                        Metric
                      </th>
                      <th className="px-4 py-3 font-bold">
                        Baseline
                      </th>
                      <th className="px-4 py-3 font-bold">
                        Feature 850 Study
                      </th>
                    </tr>
                  </thead>

                  <tbody className="text-xs">
                    {[
                      ["Accuracy", baseline.accuracy, intervention.accuracy],
                      ["Precision", baseline.precision, intervention.precision],
                      ["Recall", baseline.recall, intervention.recall],
                      ["F1 Score", baseline.f1, intervention.f1],
                    ].map(([metric, first, second]) => (
                      <tr
                        key={metric}
                        className="border-b border-slate-100 last:border-0"
                      >
                        <td className="px-4 py-4 font-semibold text-slate-700">
                          {metric}
                        </td>

                        <td className="px-4 py-4 text-slate-600">
                          {Number(first).toFixed(4)}%
                        </td>

                        <td className="px-4 py-4 font-semibold text-slate-800">
                          {Number(second).toFixed(4)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="mt-6 grid gap-6 lg:grid-cols-2">
              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-50">
                    <Target className="h-5 w-5 text-cyan-600" />
                  </div>

                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                      Baseline
                    </p>

                    <h3 className="mt-1 text-sm font-bold text-slate-950">
                      Confusion Matrix
                    </h3>
                  </div>
                </div>

                <div className="mt-5 grid grid-cols-2 gap-3">
                  <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
                    <p className="text-[10px] text-slate-400">
                      True Positive
                    </p>

                    <p className="mt-2 text-xl font-bold text-slate-950">
                      {confusion.tp}
                    </p>
                  </div>

                  <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
                    <p className="text-[10px] text-slate-400">
                      True Negative
                    </p>

                    <p className="mt-2 text-xl font-bold text-slate-950">
                      {confusion.tn}
                    </p>
                  </div>

                  <div className="rounded-xl border border-red-200 bg-red-50 p-4">
                    <p className="text-[10px] text-slate-400">
                      False Positive
                    </p>

                    <p className="mt-2 text-xl font-bold text-slate-950">
                      {confusion.fp}
                    </p>
                  </div>

                  <div className="rounded-xl border border-red-200 bg-red-50 p-4">
                    <p className="text-[10px] text-slate-400">
                      False Negative
                    </p>

                    <p className="mt-2 text-xl font-bold text-slate-950">
                      {confusion.fn}
                    </p>
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-50">
                    <Activity className="h-5 w-5 text-violet-600" />
                  </div>

                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                      Research Study
                    </p>

                    <h3 className="mt-1 text-sm font-bold text-slate-950">
                      Feature 850 Intervention
                    </h3>
                  </div>
                </div>

                <div className="mt-5 grid grid-cols-2 gap-3">
                  <div className="rounded-xl bg-slate-50 p-4">
                    <p className="text-[10px] text-slate-400">
                      True Positive
                    </p>

                    <p className="mt-2 text-xl font-bold text-slate-950">
                      {interventionConfusion.tp}
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-50 p-4">
                    <p className="text-[10px] text-slate-400">
                      True Negative
                    </p>

                    <p className="mt-2 text-xl font-bold text-slate-950">
                      {interventionConfusion.tn}
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-50 p-4">
                    <p className="text-[10px] text-slate-400">
                      False Positive
                    </p>

                    <p className="mt-2 text-xl font-bold text-slate-950">
                      {interventionConfusion.fp}
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-50 p-4">
                    <p className="text-[10px] text-slate-400">
                      False Negative
                    </p>

                    <p className="mt-2 text-xl font-bold text-slate-950">
                      {interventionConfusion.fn}
                    </p>
                  </div>
                </div>
              </div>
            </section>

            <div className="mt-6 rounded-2xl border border-amber-200 bg-amber-50 p-5">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />

                <p className="text-[11px] leading-6 text-slate-600">
                  Feature 850 intervention is presented as a research
                  evaluation only. The production safety pipeline uses the
                  unmodified sparse representation.
                </p>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}