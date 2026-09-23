import { LucideIcon } from "lucide-react";

type StatCardProps = {
  label: string;
  value: string;
  description: string;
  icon: LucideIcon;
  accent?: "cyan" | "emerald" | "violet";
};

export default function StatCard({
  label,
  value,
  description,
  icon: Icon,
  accent = "cyan",
}: StatCardProps) {
  const accentClasses = {
    cyan: "bg-cyan-50 text-cyan-600 ring-cyan-100",
    emerald: "bg-emerald-50 text-emerald-600 ring-emerald-100",
    violet: "bg-violet-50 text-violet-600 ring-violet-100",
  };

  return (
    <div className="group rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
            {label}
          </p>

          <p className="mt-3 text-xl font-bold tracking-tight text-slate-950">
            {value}
          </p>
        </div>

        <div
          className={`flex h-9 w-9 items-center justify-center rounded-xl ring-1 ${accentClasses[accent]}`}
        >
          <Icon className="h-4 w-4" />
        </div>
      </div>

      <p className="mt-2 text-[11px] leading-5 text-slate-500">
        {description}
      </p>
    </div>
  );
}