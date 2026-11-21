import React from 'react';
import { 
  Home,
  UploadCloud, 
  FileCheck2,
  BarChart3, 
  GitCompare, 
  Cpu,
  Sparkles
} from 'lucide-react';

export type NavTab = 
  | 'landing'
  | 'predictor' 
  | 'report'
  | 'performance' 
  | 'comparison' 
  | 'about';

interface NavbarProps {
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;
  systemHealthy?: boolean;
  hasReport?: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, systemHealthy = true, hasReport = false }) => {
  const navItems = [
    { id: 'landing', label: 'Home', icon: Home },
    { id: 'predictor', label: 'Inference Lab', icon: UploadCloud, highlight: true },
    { id: 'report', label: 'Lab Report', icon: FileCheck2, badge: hasReport },
    { id: 'performance', label: 'Performance', icon: BarChart3 },
    { id: 'comparison', label: 'Benchmarks', icon: GitCompare },
    { id: 'about', label: 'Architecture', icon: Cpu },
  ];

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800 print:hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div 
            onClick={() => setActiveTab('landing')}
            className="flex items-center gap-3 cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 via-indigo-500 to-emerald-400 p-0.5 glow-blue transition-transform group-hover:scale-105">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-sky-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg text-white tracking-tight">NeuroSpiral</span>
                <span className="text-[10px] uppercase font-bold tracking-widest px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-400 border border-sky-500/30">
                  AI Engine
                </span>
              </div>
              <p className="text-[11px] text-slate-400">Neuromotor Tremor Vision System</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden lg:flex items-center gap-1.5">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id as NavTab)}
                  className={`relative flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all ${
                    isActive
                      ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  } ${item.highlight && !isActive ? 'text-sky-400' : ''}`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-sky-400' : ''}`} />
                  {item.label}
                  {item.badge && (
                    <span className="w-2 h-2 rounded-full bg-emerald-400 ml-0.5" title="Latest report ready" />
                  )}
                </button>
              );
            })}
          </nav>

          {/* Health Status Indicator */}
          <div className="flex items-center gap-2.5">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/80 border border-slate-700/60 text-xs">
              <span className="relative flex h-2 w-2">
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${systemHealthy ? 'bg-emerald-400' : 'bg-amber-400'} opacity-75`}></span>
                <span className={`relative inline-flex rounded-full h-2 w-2 ${systemHealthy ? 'bg-emerald-500' : 'bg-amber-500'}`}></span>
              </span>
              <span className="text-[11px] font-mono text-slate-300">
                {systemHealthy ? 'System Online' : 'Connecting'}
              </span>
            </div>
          </div>
        </div>

        {/* Mobile Navigation bar */}
        <div className="lg:hidden flex items-center gap-1 overflow-x-auto py-2 border-t border-slate-800 scrollbar-none">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id as NavTab)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs whitespace-nowrap font-medium ${
                  isActive
                    ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {item.label}
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
