import React from 'react';
import { ArrowRight, Sparkles, ShieldCheck, Cpu, Zap } from 'lucide-react';

interface HeroProps {
  onStartClassification: () => void;
  onStartBenchmark: () => void;
}

export const Hero: React.FC<HeroProps> = ({ onStartClassification, onStartBenchmark }) => {
  return (
    <div className="relative overflow-hidden glass-card rounded-3xl p-8 lg:p-12 mb-10 border border-slate-800/80">
      {/* Background Accent Blur */}
      <div className="absolute -top-24 -right-24 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="relative z-10 max-w-4xl">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold mb-6">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Thesis Defense Interactive Platform</span>
        </div>

        <h1 className="text-3xl lg:text-5xl font-extrabold tracking-tight text-white mb-4 leading-tight">
          Plant Leaf Prediction & <span className="text-gradient-emerald">Genetic Binary Cuckoo Search</span>
        </h1>

        <p className="text-slate-300 text-base lg:text-lg mb-8 leading-relaxed max-w-3xl">
          An advanced metaheuristic machine learning platform designed for high-dimensional feature selection. 
          Upload leaf specimens, compare <strong className="text-emerald-400 font-semibold">Baseline BCS</strong> vs. <strong className="text-teal-300 font-semibold">Proposed GBCS</strong>, 
          and evaluate real-time classification performance across Swedish, Flavia, and Philippine botanical datasets.
        </p>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-4 mb-8">
          <button
            onClick={onStartClassification}
            className="flex items-center space-x-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-400 text-slate-950 font-bold text-sm hover:opacity-95 transition-all shadow-lg shadow-emerald-500/25 transform hover:-translate-y-0.5"
          >
            <span>Classify Leaf Image</span>
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            onClick={onStartBenchmark}
            className="flex items-center space-x-2 px-6 py-3.5 rounded-xl bg-slate-900 text-slate-200 font-semibold text-sm border border-slate-700 hover:bg-slate-800 transition-all transform hover:-translate-y-0.5"
          >
            <Zap className="w-4 h-4 text-emerald-400" />
            <span>BCS vs GBCS Benchmark</span>
          </button>
        </div>

        {/* Feature Badges */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-6 border-t border-slate-800/80">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs font-bold text-white">81.4% Feature Reduction</div>
              <div className="text-[11px] text-slate-400">2048 to ~380 active attributes</div>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs font-bold text-white">98.45% Max Accuracy</div>
              <div className="text-[11px] text-slate-400">Outperforms standard BCS</div>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs font-bold text-white">Multi-Dataset Support</div>
              <div className="text-[11px] text-slate-400">Swedish, Flavia, & Philippine</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
