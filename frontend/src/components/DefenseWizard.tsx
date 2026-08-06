import React, { useState } from 'react';
import { Presentation, ChevronRight, ChevronLeft, CheckCircle2, Sparkles } from 'lucide-react';

interface DefenseWizardProps {
  onNavigateToTab: (tab: string) => void;
}

export const DefenseWizard: React.FC<DefenseWizardProps> = ({ onNavigateToTab }) => {
  const [currentStep, setCurrentStep] = useState<number>(0);

  const steps = [
    {
      title: '1. Problem Statement & Research Objectives',
      subtitle: 'High-Dimensional Feature Selection in Automated Plant Leaf Identification',
      content: (
        <div className="space-y-4">
          <p className="text-slate-300 text-sm leading-relaxed">
            Automated plant species classification relies on computer vision feature extraction vectors containing thousands of continuous shape, texture, and color attributes. However, redundant and noisy features degrade classifier accuracy and increase computational latency.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
            <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
              <h4 className="text-xs font-bold text-emerald-400 uppercase">Primary Objective</h4>
              <p className="text-xs text-slate-300 mt-1">Develop a Hybrid Genetic Binary Cuckoo Search (GBCS) algorithm that optimizes feature subset selection while maximizing prediction accuracy.</p>
            </div>
            <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
              <h4 className="text-xs font-bold text-teal-400 uppercase">Evaluated Datasets</h4>
              <p className="text-xs text-slate-300 mt-1">Evaluated across Swedish Leaf Dataset (15 classes), Flavia Leaf Dataset (32 classes), and Philippine Native Leaf Dataset.</p>
            </div>
          </div>
        </div>
      ),
    },
    {
      title: '2. Proposed Methodology: Genetic Binary Cuckoo Search (GBCS)',
      subtitle: 'Combining Lévy Flight Exploration with Genetic Exploitation Operators',
      content: (
        <div className="space-y-4">
          <p className="text-slate-300 text-sm leading-relaxed">
            Standard Binary Cuckoo Search (BCS) utilizes Lévy flights for global search but often suffers from premature convergence in high-dimensional feature spaces. GBCS integrates Genetic Crossover (Pc = 0.80) and Mutation (Pm = 0.02) to maintain population diversity.
          </p>
          <div className="p-4 rounded-xl bg-slate-950/80 border border-emerald-500/30 text-xs space-y-2 font-mono">
            <div className="text-emerald-400 font-bold font-sans">Fitness Function Formulation:</div>
            <div className="text-slate-300">f(S) = α · Error(S) + (1 - α) · (|S| / |N|)</div>
            <div className="text-slate-400 text-[11px] font-sans">Where Error(S) is 10-fold CV error, |S| is selected features, and |N| = 2048 is original feature dimension.</div>
          </div>
        </div>
      ),
    },
    {
      title: '3. Live System Demonstration',
      subtitle: 'Interactive Leaf Upload & Classification Pipeline',
      content: (
        <div className="space-y-4 text-center py-4">
          <p className="text-slate-300 text-sm">
            We invite the panel to upload a leaf photo or pick a specimen preset to inspect real-time feature extraction and prediction results.
          </p>
          <button
            onClick={() => onNavigateToTab('classifier')}
            className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-400 text-slate-950 font-bold text-xs hover:opacity-95 transition-all shadow-lg shadow-emerald-500/20"
          >
            <Sparkles className="w-4 h-4" />
            <span>Launch Live Leaf Classifier Demo</span>
          </button>
        </div>
      ),
    },
    {
      title: '4. Comparative Results & Discussion',
      subtitle: 'Empirical Proof of Feature Reduction & Classification Accuracy',
      content: (
        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-center">
              <div className="text-2xl font-extrabold text-emerald-400 font-mono">98.45%</div>
              <div className="text-[11px] font-semibold text-slate-300 mt-1">Swedish Max Accuracy</div>
            </div>
            <div className="p-4 rounded-xl bg-teal-500/10 border border-teal-500/20 text-center">
              <div className="text-2xl font-extrabold text-teal-400 font-mono">81.44%</div>
              <div className="text-[11px] font-semibold text-slate-300 mt-1">Philippine Pruning Ratio</div>
            </div>
            <div className="p-4 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-center">
              <div className="text-2xl font-extrabold text-cyan-400 font-mono">+4.25%</div>
              <div className="text-[11px] font-semibold text-slate-300 mt-1">Accuracy Gain over BCS</div>
            </div>
          </div>
        </div>
      ),
    },
    {
      title: '5. Conclusion & Recommendations',
      subtitle: 'Summary of Defense Contributions & Future Directions',
      content: (
        <div className="space-y-4 text-xs text-slate-300 leading-relaxed">
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
            <div className="flex items-center space-x-2 text-emerald-400 font-bold text-sm">
              <CheckCircle2 className="w-4 h-4" />
              <span>Key Defense Contributions</span>
            </div>
            <ul className="list-disc list-inside space-y-1 text-slate-300">
              <li>Demonstrated superior feature pruning capability (averaging 75-81% feature reduction).</li>
              <li>Consistently outperformed baseline BCS in classification accuracy across Swedish, Flavia, and Philippine datasets.</li>
              <li>Successfully serialized lightweight models into an interactive web application for real-time field deployment.</li>
            </ul>
          </div>
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center space-x-2">
            <Presentation className="w-6 h-6 text-emerald-400" />
            <span>Thesis Panel Defense Presentation Walkthrough</span>
          </h2>
          <p className="text-slate-400 text-sm mt-1">
            A structured step-by-step presentation suite tailored for thesis panelists.
          </p>
        </div>

        {/* Step Indicator */}
        <div className="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-xl border border-slate-800 text-xs text-slate-400 self-start font-mono">
          <span>Step {currentStep + 1} of {steps.length}</span>
        </div>
      </div>

      {/* Main Slide Card */}
      <div className="glass-card rounded-3xl p-8 lg:p-10 border border-slate-800 min-h-[380px] flex flex-col justify-between">
        <div>
          <div className="text-xs uppercase font-bold tracking-widest text-emerald-400 mb-1">
            Thesis Presentation Slide #{currentStep + 1}
          </div>
          <h3 className="text-2xl font-extrabold text-white mb-2">
            {steps[currentStep].title}
          </h3>
          <div className="text-sm font-medium text-slate-400 mb-6 pb-4 border-b border-slate-800">
            {steps[currentStep].subtitle}
          </div>

          <div className="py-2">
            {steps[currentStep].content}
          </div>
        </div>

        {/* Navigation Buttons */}
        <div className="flex items-center justify-between pt-6 border-t border-slate-800 mt-8">
          <button
            onClick={() => setCurrentStep((prev) => Math.max(0, prev - 1))}
            disabled={currentStep === 0}
            className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-slate-900 text-slate-300 text-xs font-semibold border border-slate-800 disabled:opacity-40 hover:bg-slate-800 transition-all"
          >
            <ChevronLeft className="w-4 h-4" />
            <span>Previous Slide</span>
          </button>

          <button
            onClick={() => setCurrentStep((prev) => Math.min(steps.length - 1, prev + 1))}
            disabled={currentStep === steps.length - 1}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-400 text-slate-950 font-bold text-xs shadow-md shadow-emerald-500/20 disabled:opacity-40 hover:opacity-95 transition-all"
          >
            <span>Next Slide</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
