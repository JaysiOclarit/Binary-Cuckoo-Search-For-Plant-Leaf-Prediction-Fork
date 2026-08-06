import React, { useState, useEffect } from 'react';
import { Activity, Play, Pause, RotateCcw, Zap, Sparkles } from 'lucide-react';
import { ConvergencePoint } from '../types';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

interface CuckooSimulatorProps {
  selectedDataset: string;
  onFetchConvergence: (dataset: string) => Promise<ConvergencePoint[]>;
}

export const CuckooSimulator: React.FC<CuckooSimulatorProps> = ({
  selectedDataset,
  onFetchConvergence,
}) => {
  const [convergenceData, setConvergenceData] = useState<ConvergencePoint[]>([]);
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  useEffect(() => {
    loadData();
  }, [selectedDataset]);

  const loadData = async () => {
    try {
      const pts = await onFetchConvergence(selectedDataset);
      setConvergenceData(pts);
      setCurrentStep(1);
    } catch (e) {
      console.error(e);
    }
  };

  const maxStep = convergenceData.length > 0 ? convergenceData.length : 21;

  useEffect(() => {
    let timer: any;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= maxStep) {
            setIsPlaying(false);
            return maxStep;
          }
          return prev + 1;
        });
      }, 300);
    }
    return () => clearInterval(timer);
  }, [isPlaying, maxStep]);

  const displayedData = convergenceData.slice(0, currentStep);
  const latestPoint = displayedData[displayedData.length - 1] || { gbcsFitness: 0, bcsFitness: 0 };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center space-x-2">
            <Activity className="w-6 h-6 text-emerald-400" />
            <span>Interactive Cuckoo Search Convergence Simulator</span>
          </h2>
          <p className="text-slate-400 text-sm mt-1">
            Visualizes fitness evaluation f(x) across 20 metaheuristic optimization iterations (t = 0 ... 20).
          </p>
        </div>

        {/* Play Controls */}
        <div className="flex items-center space-x-2 bg-slate-900/90 p-1.5 rounded-xl border border-slate-800 self-start">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-emerald-500 text-slate-950 font-bold text-xs hover:bg-emerald-400 transition-all shadow"
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isPlaying ? 'Pause' : 'Run Simulation'}</span>
          </button>
          <button
            onClick={() => {
              setIsPlaying(false);
              setCurrentStep(1);
            }}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-all"
            title="Reset Simulation"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Simulator Visualization Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left: Interactive Convergence Line Graph */}
        <div className="lg:col-span-8 glass-card rounded-2xl p-6 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-slate-200 flex items-center space-x-2">
              <Zap className="w-4 h-4 text-emerald-400" />
              <span>Fitness Function Convergence Curve f(x)</span>
            </h3>
            <span className="text-xs font-mono font-bold text-emerald-400 bg-slate-900 px-2.5 py-1 rounded border border-slate-800">
              Iteration {currentStep - 1} / {maxStep - 1}
            </span>
          </div>

          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={displayedData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="iteration" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[0.85, 0.98]} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8, fontSize: 12 }} formatter={(val: any) => [Number(val).toFixed(6), 'Fitness Score f(x)']} />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
                <Line type="monotone" dataKey="gbcsFitness" name="Proposed GBCS" stroke="#10b981" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                <Line type="monotone" dataKey="bcsFitness" name="Baseline BCS" stroke="#64748b" strokeWidth={2} strokeDasharray="4 4" dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right: State Breakdown */}
        <div className="lg:col-span-4 space-y-6">
          <div className="glass-card rounded-2xl p-6 border border-slate-800">
            <h3 className="text-sm font-bold text-white mb-4 flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-teal-400" />
              <span>Iteration State Breakdown</span>
            </h3>

            <div className="space-y-4">
              <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
                <div className="text-[11px] font-semibold text-slate-400 uppercase">GBCS Best Fitness f(x)</div>
                <div className="text-xl font-extrabold text-emerald-400 font-mono mt-0.5">
                  {latestPoint.gbcsFitness}
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
                <div className="text-[11px] font-semibold text-slate-400 uppercase">BCS Best Fitness f(x)</div>
                <div className="text-xl font-extrabold text-slate-300 font-mono mt-0.5">
                  {latestPoint.bcsFitness}
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-300">
                <div className="font-bold mb-1">Cuckoo Search Operators Active:</div>
                <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-300">
                  <li>Lévy Flight step length: α = 0.01</li>
                  <li>Alien Egg discovery rate: Pa = 0.25</li>
                  <li>Genetic Crossover Rate: Pc = 0.80</li>
                  <li>Genetic Mutation Rate: Pm = 0.02</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
