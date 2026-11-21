import React from 'react';
import { ShieldCheck, Info } from 'lucide-react';

interface Props {
  variant?: 'banner' | 'card' | 'inline';
}

export const MedicalDisclaimer: React.FC<Props> = ({ variant = 'banner' }) => {
  if (variant === 'inline') {
    return (
      <div className="flex items-center gap-2 text-xs text-slate-400 bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
        <Info className="w-4 h-4 text-sky-400 shrink-0" />
        <p>
          NeuroSpiral is an AI investigational motor screening system. Always consult a qualified medical professional for clinical diagnosis.
        </p>
      </div>
    );
  }

  return (
    <div className="w-full bg-slate-900/50 border border-slate-800/80 rounded-2xl px-4 py-2.5 shadow-sm print:hidden">
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-300">
        <div className="flex items-center gap-2.5">
          <div className="p-1 rounded-md bg-sky-500/20 text-sky-400">
            <Info className="w-3.5 h-3.5" />
          </div>
          <p className="text-[12px] text-slate-300">
            <span className="font-semibold text-white">Investigational Tool:</span> NeuroSpiral provides automated computer vision pattern analysis on drawing examinations. Designed to support clinical research and motor assessment.
          </p>
        </div>
        <div className="flex items-center gap-1.5 text-[11px] text-emerald-400 font-mono shrink-0">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>Validated Neural Models Active</span>
        </div>
      </div>
    </div>
  );
};
