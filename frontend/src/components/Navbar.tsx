import React from 'react';
import { Leaf, Activity, Layers, Database, BarChart3, Presentation, Radio } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  selectedDataset: string;
  setSelectedDataset: (ds: string) => void;
  apiStatus: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  selectedDataset,
  setSelectedDataset,
  apiStatus,
}) => {
  const navItems = [
    { id: 'classifier', label: 'Leaf Classifier', icon: Leaf },
    { id: 'benchmark', label: 'BCS vs GBCS Benchmark', icon: Layers },
    { id: 'simulator', label: 'Cuckoo Simulator', icon: Activity },
    { id: 'encyclopedia', label: 'Botanical Catalog', icon: Database },
    { id: 'analytics', label: 'Thesis Analytics', icon: BarChart3 },
    { id: 'defense', label: 'Defense Presentation', icon: Presentation },
  ];

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/80 px-4 lg:px-8 py-3">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Brand & System Title */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 p-0.5 shadow-lg shadow-emerald-500/20">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Leaf className="w-5 h-5 text-emerald-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-extrabold text-lg tracking-tight text-white">PhytoCuckoo</span>
              <span className="px-2 py-0.5 text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full">
                GBCS v2.0
              </span>
            </div>
            <p className="text-xs text-slate-400">Binary Cuckoo Search Plant Leaf Classification</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1 bg-slate-900/90 p-1.5 rounded-xl border border-slate-800 overflow-x-auto max-w-full">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-emerald-500 text-slate-950 shadow-md font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-slate-950' : 'text-slate-400'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Utility Bar: Dataset Switcher & Backend Status */}
        <div className="flex items-center space-x-3">
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="bg-slate-900 text-slate-200 text-xs font-medium border border-slate-700/80 rounded-lg px-3 py-1.5 focus:outline-none focus:border-emerald-500 transition-colors"
          >
            <option value="swedish">Swedish Leaf (15 Species)</option>
            <option value="flavia">Flavia Leaf (32 Species)</option>
            <option value="philippine">Philippine Leaf (Local)</option>
          </select>

          <div
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border ${
              apiStatus
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
            }`}
          >
            <Radio className={`w-3 h-3 animate-pulse ${apiStatus ? 'text-emerald-400' : 'text-amber-400'}`} />
            <span>{apiStatus ? 'API Connected' : 'Simulated API'}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
