import { Cpu, Menu, Wifi } from "lucide-react";

type TopbarProps = {
  page?: string;
};

export default function Topbar({
  page = "Dashboard",
}: TopbarProps) {
  return (
    <header className="flex min-h-[78px] items-center justify-between border-b border-slate-200 bg-white px-5 sm:px-7 lg:px-9">
      <div className="flex items-center gap-3">
        <button
          type="button"
          className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 text-slate-500 lg:hidden"
        >
          <Menu className="h-4 w-4" />
        </button>

        <div>
          <div className="flex items-center gap-2 text-[11px] font-medium text-slate-400">
            <span>Workspace</span>
            <span>/</span>
            <span className="text-slate-600">
              {page}
            </span>
          </div>

          <h1 className="mt-1 text-[18px] font-bold tracking-tight text-slate-950">
            {page === "Dashboard"
              ? "Safety Intelligence"
              : `Neural ${page}`}
          </h1>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <div className="hidden items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-2 sm:flex">
          <Wifi className="h-3.5 w-3.5 text-emerald-600" />

          <span className="text-[11px] font-semibold text-slate-600">
            API Online
          </span>
        </div>

        <div className="flex items-center gap-2 rounded-full border border-cyan-200 bg-cyan-50 px-3 py-2">
          <Cpu className="h-3.5 w-3.5 text-cyan-600" />

          <span className="text-[11px] font-bold text-cyan-700">
            CUDA
          </span>
        </div>
      </div>
    </header>
  );
}