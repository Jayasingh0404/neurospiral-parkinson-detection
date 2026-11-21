import React from 'react';
import { Cpu, ShieldCheck, Sparkles } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-800 bg-slate-950/90 text-slate-400 py-8 px-4 sm:px-6 lg:px-8 mt-16 print:hidden">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-sky-500/20 text-sky-400 flex items-center justify-center font-bold">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <span className="font-semibold text-white">NeuroSpiral AI</span>
            <span className="text-slate-500 ml-2">&bull; Deep Neuromotor Tremor Vision Platform</span>
          </div>
        </div>

        <div className="flex items-center gap-6 text-slate-400 text-xs">
          <span>TensorFlow 2.21 + Keras 3</span>
          <span>FastAPI Microservice</span>
          <span>React 18 + Vite</span>
          <span className="text-emerald-400 flex items-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5" /> Models Deployed
          </span>
        </div>
      </div>
    </footer>
  );
};
