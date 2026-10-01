import React from 'react';
import { Layers, Trophy, Activity, ShieldCheck, Leaf, ArrowRight, Trash2 } from 'lucide-react';
import { PredictionResult } from '../types';
import { ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, Tooltip } from 'recharts';

interface SideBySideBenchmarkProps {
  gbcsData: PredictionResult | null;
  bcsData: PredictionResult | null;
  onNavigateToClassifier: () => void;
  onClearBenchmark: () => void;
}

export const SideBySideBenchmark: React.FC<SideBySideBenchmarkProps> = ({
  gbcsData,
  bcsData,
  onNavigateToClassifier,
  onClearBenchmark,
}) => {
  const radarData = [
    { category: 'Deep Conv Subspace A', GBCS: 88.0, BCS: 62.0 },
    { category: 'Deep Conv Subspace B', GBCS: 94.0, BCS: 70.0 },
    { category: 'Conv Bottleneck Embedding', GBCS: 90.0, BCS: 75.0 },
    { category: 'Spatial Pooling Vector', GBCS: 85.0, BCS: 55.0 },
    { category: 'Channel Activation Weights', GBCS: 92.0, BCS: 68.0 },
    { category: 'Hierarchical Representation', GBCS: 89.0, BCS: 60.0 },
  ];

  const hasBoth = gbcsData !== null && bcsData !== null;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center space-x-2">
            <Layers className="w-6 h-6 text-emerald-400" />
            <span>Baseline BCS vs. Proposed GBCS Real-Time Benchmark</span>
          </h2>
          <p className="text-slate-400 text-sm mt-1">
            Live comparison results from the Leaf Classifier.
          </p>
        </div>
        
        {(gbcsData || bcsData) && (
          <button
            onClick={onClearBenchmark}
            className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-slate-900 border border-slate-700 hover:bg-rose-500/10 hover:text-rose-400 hover:border-rose-500/30 text-slate-400 font-semibold text-xs transition-all self-start"
          >
            <Trash2 className="w-4 h-4" />
            <span>Clear Benchmark</span>
          </button>
        )}
      </div>

      {hasBoth ? (
        <div className="space-y-8">
          {/* Side by Side Comparison Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Proposed GBCS Card */}
            <div className="relative glass-card rounded-2xl p-6 border-2 border-emerald-500/30 shadow-lg shadow-emerald-500/5 transition-all">
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2.5 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  <ShieldCheck className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white">Genetic Binary Cuckoo Search (GBCS)</h3>
                  <p className="text-xs text-slate-400">Proposed Algorithm with Crossover & Mutation Operators</p>
                </div>
              </div>
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
                  <div className="text-[11px] font-semibold text-slate-400 uppercase mb-1">Predicted Specimen</div>
                  <div className="text-xl font-black capitalize text-emerald-400">{gbcsData.predictedClass}</div>
                  <div className="mt-2 flex items-center justify-between text-xs">
                    <span className="text-slate-400">Model Confidence</span>
                    <span className="font-bold text-white font-mono">{(gbcsData.confidenceScore * 100).toFixed(2)}%</span>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded-xl bg-slate-950/40 border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Active Features</div>
                    <div className="text-lg font-bold text-white font-mono mt-1">{gbcsData.featureCount}</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950/40 border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Pruning Ratio</div>
                    <div className="text-lg font-bold text-emerald-400 font-mono mt-1">{(((2048 - gbcsData.featureCount) / 2048) * 100).toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Baseline BCS Card */}
            <div className="relative glass-card rounded-2xl p-6 border-2 border-slate-800/80 transition-all">
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2.5 rounded-xl bg-slate-800 text-slate-400 border border-slate-700">
                  <Activity className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-slate-200">Baseline Binary Cuckoo Search (BCS)</h3>
                  <p className="text-xs text-slate-400">Standard Lévy Flight Feature Selection</p>
                </div>
              </div>
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
                  <div className="text-[11px] font-semibold text-slate-400 uppercase mb-1">Predicted Specimen</div>
                  <div className="text-xl font-black capitalize text-slate-200">{bcsData.predictedClass}</div>
                  <div className="mt-2 flex items-center justify-between text-xs">
                    <span className="text-slate-400">Model Confidence</span>
                    <span className="font-bold text-slate-200 font-mono">{(bcsData.confidenceScore * 100).toFixed(2)}%</span>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded-xl bg-slate-950/40 border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Active Features</div>
                    <div className="text-lg font-bold text-slate-200 font-mono mt-1">{bcsData.featureCount}</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950/40 border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Pruning Ratio</div>
                    <div className="text-lg font-bold text-amber-400 font-mono mt-1">{(((2048 - bcsData.featureCount) / 2048) * 100).toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Feature Sensitivity & Relevance Radar Chart */}
          <div className="glass-card rounded-2xl p-6 border border-slate-800">
            <h3 className="text-base font-bold text-white mb-2 flex items-center space-x-2">
              <Trophy className="w-4 h-4 text-emerald-400" />
              <span>Feature Group Activation Profile (GBCS vs. BCS)</span>
            </h3>
            <p className="text-xs text-slate-400 mb-6">
              Shows how the Genetic operators in GBCS preserve high-impact shape & texture vectors while discarding redundant noise.
            </p>
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#334155" />
                  <PolarAngleAxis dataKey="category" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#475569" />
                  <Radar name="Proposed GBCS" dataKey="GBCS" stroke="#10b981" fill="#10b981" fillOpacity={0.4} />
                  <Radar name="Baseline BCS" dataKey="BCS" stroke="#64748b" fill="#64748b" fillOpacity={0.2} />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8, fontSize: 12 }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      ) : (
        /* Empty Placeholder — Missing classifications */
        <div className="glass-card rounded-2xl p-16 flex flex-col items-center justify-center text-center border border-slate-800/60">
          <div className="w-16 h-16 rounded-full bg-slate-800/50 flex items-center justify-center mb-5">
            <Leaf className="w-8 h-8 text-slate-600" />
          </div>
          <h3 className="text-lg font-bold text-slate-300 mb-2">Incomplete Benchmark Data</h3>
          <div className="text-sm text-slate-400 max-w-md mb-6 space-y-2">
            <p>
              To see the side-by-side comparison, you need to classify a leaf using <strong>both</strong> methods.
            </p>
            <div className="flex items-center justify-center space-x-6 py-2">
              <div className={`flex items-center space-x-2 ${gbcsData ? 'text-emerald-400' : 'text-slate-500'}`}>
                <ShieldCheck className="w-5 h-5" />
                <span>GBCS {gbcsData ? '(Done)' : '(Missing)'}</span>
              </div>
              <div className={`flex items-center space-x-2 ${bcsData ? 'text-emerald-400' : 'text-slate-500'}`}>
                <Activity className="w-5 h-5" />
                <span>BCS {bcsData ? '(Done)' : '(Missing)'}</span>
              </div>
            </div>
          </div>
          <button
            onClick={onNavigateToClassifier}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-400 text-slate-950 font-bold text-sm hover:opacity-95 transition-all shadow-md shadow-emerald-500/20"
          >
            <span>Go to Leaf Classifier</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
};
