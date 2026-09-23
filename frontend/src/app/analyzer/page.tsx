import Link from "next/link";
import { ArrowLeft, ScanText, ShieldCheck } from "lucide-react";
import Analyzer from "../../components/Analyzer";
import Sidebar from "../../components/Sidebar";
import Topbar from "../../components/Topbar";

export default function AnalyzerPage() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="flex min-h-screen">
        <Sidebar />

        <main className="min-w-0 flex-1">
          <Topbar page="Analyzer" />

          <div className="mx-auto max-w-[1500px] px-5 py-7 sm:px-7 lg:px-9">
            <Link
              href="/"
              className="mb-6 inline-flex items-center gap-2 text-xs font-medium text-slate-500 transition hover:text-cyan-600"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              Back to Dashboard
            </Link>

            <section className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="absolute right-0 top-0 h-full w-[45%] bg-gradient-to-l from-cyan-50/80 to-transparent" />

              <div className="relative px-7 py-8 sm:px-9 sm:py-10">
                <div className="inline-flex items-center gap-2 rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1.5">
                  <ScanText className="h-3.5 w-3.5 text-cyan-600" />

                  <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-cyan-700">
                    Neural Analysis
                  </span>
                </div>

                <h2 className="mt-5 text-3xl font-bold tracking-tight text-slate-950 sm:text-4xl">
                  Prompt{" "}
                  <span className="text-cyan-600">
                    Safety Analyzer
                  </span>
                </h2>

                <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-500">
                  Send a prompt through the production neural safety
                  pipeline and inspect its classification, probability,
                  and sparse feature activity.
                </p>

                <div className="mt-5 flex items-center gap-2 text-[11px] font-medium text-emerald-600">
                  <ShieldCheck className="h-4 w-4" />
                  Production inference path
                </div>
              </div>
            </section>

            <div className="mt-6">
              <Analyzer />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}