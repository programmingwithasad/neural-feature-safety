"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Beaker,
  BrainCircuit,
  History,
  Layers3,
  MessageSquare,
  ScanText,
  Shield,
  Target,
} from "lucide-react";

const navigation = [
  {
    label: "Live Gateway",
    href: "/",
    icon: MessageSquare,
    section: "Overview",
  },
  {
    label: "Analyzer",
    href: "/analyzer",
    icon: ScanText,
    section: "Overview",
  },
  {
    label: "Feature Explorer",
    href: "/feature-explorer",
    icon: Layers3,
    section: "Interpretability",
  },
  {
    label: "Model Architecture",
    href: "/model",
    icon: BrainCircuit,
    section: "Interpretability",
  },
  {
    label: "Experiments",
    href: "/experiments",
    icon: Beaker,
    section: "Research",
  },
  {
    label: "Evaluation",
    href: "/evaluation",
    icon: Target,
    section: "Research",
  },
  {
    label: "Analysis History",
    href: "/history",
    icon: History,
    section: "System",
  },
];

const sections = [
  "Overview",
  "Interpretability",
  "Research",
  "System",
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-[250px] shrink-0 border-r border-slate-200 bg-white lg:flex lg:flex-col">
      <div className="border-b border-slate-200 px-6 py-6">
        <Link href="/" className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-50 ring-1 ring-cyan-100">
            <Shield className="h-5 w-5 text-cyan-600" />
          </div>

          <div>
            <div className="text-[16px] font-bold tracking-tight text-slate-950">
              Neural Feature
            </div>

            <div className="mt-0.5 text-[11px] font-medium text-slate-500">
              Safety Intelligence Platform
            </div>
          </div>
        </Link>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-6">
        {sections.map((section) => (
          <div key={section} className="mb-7">
            <div className="mb-2 px-3 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-400">
              {section}
            </div>

            <div className="space-y-1">
              {navigation
                .filter((item) => item.section === section)
                .map((item) => {
                  const Icon = item.icon;

                  const active =
                    item.href === "/"
                      ? pathname === "/"
                      : pathname.startsWith(item.href);

                  return (
                    <Link
                      key={item.label}
                      href={item.href}
                      className={`group flex items-center gap-3 rounded-xl px-3 py-2.5 text-[13px] font-medium transition ${
                        active
                          ? "bg-slate-950 text-white shadow-sm"
                          : "text-slate-500 hover:bg-slate-50 hover:text-slate-900"
                      }`}
                    >
                      <Icon
                        className={`h-[17px] w-[17px] ${
                          active
                            ? "text-cyan-300"
                            : "text-slate-400 group-hover:text-cyan-600"
                        }`}
                      />

                      <span>{item.label}</span>

                      {active && (
                        <span className="ml-auto h-1.5 w-1.5 rounded-full bg-cyan-400" />
                      )}
                    </Link>
                  );
                })}
            </div>
          </div>
        ))}
      </div>

      <div className="border-t border-slate-200 p-4">
        <div className="rounded-xl border border-emerald-100 bg-emerald-50 p-4">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />

            <span className="text-xs font-bold text-emerald-700">
              System Operational
            </span>
          </div>

          <p className="mt-2 text-[11px] leading-5 text-slate-500">
            Neural inference engine is ready for protected AI interaction.
          </p>
        </div>
      </div>
    </aside>
  );
}