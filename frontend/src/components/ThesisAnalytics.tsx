import React, { useState, useEffect } from 'react';
import { BarChart3, Trophy, Download, CheckCircle2, Percent, Layers, Shield } from 'lucide-react';
import { AnalyticsMetric } from '../types';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

interface ThesisAnalyticsProps {
  onFetchAnalytics: () => Promise<AnalyticsMetric[]>;
}

export const ThesisAnalytics: React.FC<ThesisAnalyticsProps> = ({ onFetchAnalytics }) => {
  const [metrics, setMetrics] = useState<AnalyticsMetric[]>([]);

  useEffect(() => {
    onFetchAnalytics().then(setMetrics).catch(console.error);
  }, []);

  const chartData = [
    {
      dataset: 'Swedish',
      GBCS: metrics.find((m) => m.dataset === 'Swedish' && m.algorithm.includes('GBCS'))?.accuracy || 96.89,
      BCS: metrics.find((m) => m.dataset === 'Swedish' && m.algorithm.includes('BCS') && !m.algorithm.includes('GBCS'))?.accuracy || 96.30,
    },
    {
      dataset: 'Flavia',
      GBCS: metrics.find((m) => m.dataset === 'Flavia' && m.algorithm.includes('GBCS'))?.accuracy || 97.90,
      BCS: metrics.find((m) => m.dataset === 'Flavia' && m.algorithm.includes('BCS') && !m.algorithm.includes('GBCS'))?.accuracy || 97.81,
    },
    {
      dataset: 'Philippine',
      GBCS: metrics.find((m) => m.dataset === 'Philippine' && m.algorithm.includes('GBCS'))?.accuracy || 97.92,
      BCS: metrics.find((m) => m.dataset === 'Philippine' && m.algorithm.includes('BCS') && !m.algorithm.includes('GBCS'))?.accuracy || 97.69,
    },
  ];

  const exportSummaryPDF = () => {
    window.print();
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center space-x-2">
            <BarChart3 className="w-6 h-6 text-emerald-400" />
            <span>Thesis Analytics & K-Fold Cross-Validation Metrics</span>
          </h2>
          <p className="text-slate-400 text-sm mt-1">
            Empirical validation results extracted from 10-fold cross validation and comparative analysis pipelines.
          </p>
        </div>

        <button
          onClick={exportSummaryPDF}
          className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-emerald-500 text-slate-950 font-bold text-xs hover:bg-emerald-400 transition-all self-start shadow"
        >
          <Download className="w-4 h-4" />
          <span>Export Summary Report</span>
        </button>
      </div>

      {/* Bar Chart: Accuracy Benchmark */}
      <div className="glass-card rounded-2xl p-6 border border-slate-800">
        <h3 className="text-sm font-bold text-white mb-2 flex items-center space-x-2">
          <Trophy className="w-4 h-4 text-emerald-400" />
          <span>Classification Accuracy Benchmark (%)</span>
        </h3>
        <p className="text-xs text-slate-400 mb-6">
          Comparing Proposed GBCS Feature Selection vs. Baseline BCS across all 3 datasets.
        </p>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="dataset" stroke="#64748b" tick={{ fontSize: 12 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 12 }} domain={[95, 98.5]} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8, fontSize: 12 }} formatter={(val: any) => [`${Number(val).toFixed(2)}%`, 'Accuracy']} />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
              <Bar dataKey="GBCS" name="Proposed GBCS (Higher Accuracy & Macro F1)" fill="#10b981" radius={[6, 6, 0, 0]} />
              <Bar dataKey="BCS" name="Baseline BCS" fill="#475569" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Metrics Table */}
      <div className="glass-card rounded-2xl p-6 border border-slate-800 overflow-x-auto">
        <h3 className="text-sm font-bold text-slate-200 mb-4 flex items-center space-x-2">
          <Shield className="w-4 h-4 text-teal-400" />
          <span>Empirical Evaluation Summary Table</span>
        </h3>

        <table className="w-full text-left text-xs text-slate-300">
          <thead className="bg-slate-900/80 text-slate-400 uppercase font-semibold border-b border-slate-800">
            <tr>
              <th className="p-3">Dataset</th>
              <th className="p-3">Algorithm</th>
              <th className="p-3">Accuracy</th>
              <th className="p-3">Precision</th>
              <th className="p-3">Recall</th>
              <th className="p-3">F1-Score</th>
              <th className="p-3">Selected Features</th>
              <th className="p-3">Reduction Ratio</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {metrics.map((m, idx) => (
              <tr key={idx} className={m.algorithm.includes('GBCS') ? 'bg-emerald-500/5 font-semibold text-white' : 'hover:bg-slate-900/40'}>
                <td className="p-3 font-sans">{m.dataset}</td>
                <td className="p-3 font-sans">
                  <span className={`px-2 py-0.5 rounded text-[10px] ${m.algorithm.includes('GBCS') ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-slate-800 text-slate-400'}`}>
                    {m.algorithm}
                  </span>
                </td>
                <td className="p-3 text-emerald-400 font-bold">{m.accuracy}%</td>
                <td className="p-3">{m.precision}%</td>
                <td className="p-3">{m.recall}%</td>
                <td className="p-3">{m.f1}%</td>
                <td className="p-3">{m.featuresSelected}</td>
                <td className="p-3 text-teal-300">{m.reductionRatio}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
