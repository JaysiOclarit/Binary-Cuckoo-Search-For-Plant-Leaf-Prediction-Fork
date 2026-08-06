import React, { useState } from 'react';
import { Upload, Leaf, CheckCircle2, AlertCircle, RefreshCw, Cpu, Layers, Sparkles, Image as ImageIcon } from 'lucide-react';
import confetti from 'canvas-confetti';
import { PredictionResult } from '../types';

interface LeafClassifierProps {
  selectedDataset: string;
  setSelectedDataset: (ds: string) => void;
  onClassifyImage: (file: File, dataset: string, algorithm: string) => Promise<PredictionResult>;
  onClassifyPreset: (presetName: string, dataset: string, algorithm: string) => Promise<PredictionResult>;
}

export const LeafClassifier: React.FC<LeafClassifierProps> = ({
  selectedDataset,
  setSelectedDataset,
  onClassifyImage,
  onClassifyPreset,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [algorithm, setAlgorithm] = useState<string>('gbcs');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activePreset, setActivePreset] = useState<string | null>(null);

  // Pre-loaded Leaf Presets for Live Panelist Testing
  const presetSpecimens = [
    {
      name: 'Fagus sylvatica',
      dataset: 'swedish',
      label: 'Swedish Leaf #1',
      bgGradient: 'from-amber-700/40 to-emerald-900/60',
      description: 'European Beech leaf with Inception-V3 2048-dim representation',
    },
    {
      name: 'Quercus robur',
      dataset: 'swedish',
      label: 'Swedish Leaf #2',
      bgGradient: 'from-emerald-800/40 to-teal-950/60',
      description: 'English Oak leaf with distinct lobed margins',
    },
    {
      name: 'Acer palmatum',
      dataset: 'flavia',
      label: 'Flavia Leaf #1',
      bgGradient: 'from-rose-800/40 to-emerald-900/60',
      description: 'Japanese Maple palmately lobed leaf',
    },
    {
      name: 'Ginkgo biloba',
      dataset: 'flavia',
      label: 'Flavia Leaf #2',
      bgGradient: 'from-yellow-700/40 to-emerald-950/60',
      description: 'Fan-shaped leaf with dichotomous venation',
    },
    {
      name: 'Senna alata',
      dataset: 'philippine',
      label: 'Philippine Leaf #1 (Akapulko)',
      bgGradient: 'from-green-700/40 to-emerald-900/60',
      description: 'Philippine medicinal anti-fungal leaf specimen',
    },
    {
      name: 'Leucaena leucocephala',
      dataset: 'philippine',
      label: 'Philippine Leaf #2 (Ipil-ipil)',
      bgGradient: 'from-emerald-700/40 to-cyan-900/60',
      description: 'Bipinnately compound nitrogen-fixing forage leaf',
    },
    {
      name: 'Phyllanthus niruri',
      dataset: 'philippine',
      label: 'Philippine Leaf #3 (Sampasampalukan)',
      bgGradient: 'from-teal-700/40 to-emerald-950/60',
      description: 'Pinnately compound medicinal herbal leaf specimen',
    },
  ];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setActivePreset(null);
      setResult(null);
      setError(null);
    }
  };

  const handlePresetSelect = async (preset: typeof presetSpecimens[0]) => {
    setActivePreset(preset.name);
    setSelectedFile(null);
    setPreviewUrl(null);
    setLoading(true);
    setError(null);

    try {
      const res = await onClassifyPreset(preset.name, preset.dataset, algorithm);
      setResult(res);
      confetti({ particleCount: 80, spread: 70, origin: { y: 0.6 } });
    } catch (err: any) {
      setError(err.message || 'Classification failed');
    } finally {
      setLoading(false);
    }
  };

  const handleRunClassification = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    try {
      const res = await onClassifyImage(selectedFile, selectedDataset, algorithm);
      setResult(res);
      confetti({ particleCount: 100, spread: 70, origin: { y: 0.6 } });
    } catch (err: any) {
      setError(err.message || 'Classification failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Module Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center space-x-2">
            <Leaf className="w-6 h-6 text-emerald-400" />
            <span>Inception-V3 Leaf Classification Workbench</span>
          </h2>
          <p className="text-slate-400 text-sm mt-1">
            Extract 2048-dimensional Inception-V3 CNN features & evaluate through Tribuo machine learning models.
          </p>
        </div>

        {/* Controls: Target Dataset & Algorithm Toggle */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Target Dataset Selector */}
          <div className="flex items-center space-x-2 bg-slate-900/90 p-1.5 rounded-xl border border-slate-800">
            <span className="text-xs font-semibold text-slate-400 pl-2">Target Dataset:</span>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="bg-slate-950 text-emerald-400 text-xs font-bold border border-slate-700/80 rounded-lg px-3 py-1 focus:outline-none focus:border-emerald-500 transition-colors cursor-pointer"
            >
              <option value="swedish">Swedish Leaf (15 Species)</option>
              <option value="flavia">Flavia Leaf (32 Species)</option>
              <option value="philippine">Philippine Medicinal (40 Species)</option>
            </select>
          </div>

          {/* Algorithm Toggle */}
          <div className="flex items-center space-x-2 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setAlgorithm('gbcs')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                algorithm === 'gbcs'
                  ? 'bg-emerald-500 text-slate-950 shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Proposed GBCS (Genetic Binary)
            </button>
            <button
              onClick={() => setAlgorithm('bcs')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                algorithm === 'bcs'
                  ? 'bg-emerald-500 text-slate-950 shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Baseline BCS
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Image Upload & Presets */}
        <div className="lg:col-span-7 space-y-6">
          {/* Drag & Drop Upload Zone */}
          <div className="glass-card rounded-2xl p-6 border border-slate-800">
            <h3 className="text-sm font-bold text-slate-200 mb-4 flex items-center space-x-2">
              <Upload className="w-4 h-4 text-emerald-400" />
              <span>Option A: Upload Custom Leaf Photo</span>
            </h3>

            <label className="group relative flex flex-col items-center justify-center h-52 border-2 border-dashed border-slate-700/80 hover:border-emerald-500/80 rounded-xl cursor-pointer bg-slate-950/40 hover:bg-slate-950/70 transition-all">
              {previewUrl ? (
                <div className="relative w-full h-full p-2 flex items-center justify-center">
                  <img
                    src={previewUrl}
                    alt="Leaf specimen preview"
                    className="max-h-48 rounded-lg object-contain"
                  />
                  <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded-xl">
                    <span className="text-xs font-semibold text-white bg-slate-900/90 px-3 py-1.5 rounded-lg border border-slate-700">
                      Change Photo
                    </span>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center text-center p-4">
                  <div className="w-12 h-12 rounded-full bg-emerald-500/10 text-emerald-400 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                    <ImageIcon className="w-6 h-6" />
                  </div>
                  <p className="text-xs font-semibold text-slate-200 mb-1">
                    Drag and drop leaf image here, or <span className="text-emerald-400">browse</span>
                  </p>
                  <p className="text-[11px] text-slate-400">Inception-V3 Deep CNN Feature Extraction (2048-dim vector)</p>
                </div>
              )}

              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>

            {selectedFile && (
              <div className="mt-4 flex items-center justify-between">
                <span className="text-xs text-slate-400 truncate max-w-[220px]">
                  File: {selectedFile.name}
                </span>
                <button
                  onClick={handleRunClassification}
                  disabled={loading}
                  className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-400 text-slate-950 font-bold text-xs hover:opacity-95 transition-all shadow-md shadow-emerald-500/20 disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Extracting Inception Features...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Run Inception-V3 Classification</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Option B: Specimen Gallery Presets */}
          <div className="glass-card rounded-2xl p-6 border border-slate-800">
            <h3 className="text-sm font-bold text-slate-200 mb-1 flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-teal-400" />
              <span>Option B: Panel Defense Specimen Presets (1-Click Test)</span>
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Instant live demonstration specimens pre-configured across Swedish, Flavia, and Philippine leaf species.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {presetSpecimens.map((preset) => (
                <button
                  key={preset.name}
                  onClick={() => handlePresetSelect(preset)}
                  disabled={loading}
                  className={`group relative p-3.5 rounded-xl border text-left transition-all ${
                    activePreset === preset.name
                      ? 'bg-emerald-500/10 border-emerald-500 text-white ring-2 ring-emerald-500/30'
                      : 'bg-slate-950/50 border-slate-800/80 hover:border-slate-700 text-slate-300'
                  }`}
                >
                  <div className="flex items-start justify-between mb-1">
                    <span className="font-bold text-xs text-white group-hover:text-emerald-400 transition-colors">
                      {preset.label}
                    </span>
                    <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-800">
                      {preset.dataset}
                    </span>
                  </div>
                  <div className="text-xs font-medium text-emerald-400 italic mb-1">
                    {preset.name}
                  </div>
                  <p className="text-[11px] text-slate-400 line-clamp-1">
                    {preset.description}
                  </p>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Output Card */}
        <div className="lg:col-span-5">
          <div className="glass-card rounded-2xl p-6 border border-slate-800 sticky top-24">
            <h3 className="text-sm font-bold text-slate-200 mb-4 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Cpu className="w-4 h-4 text-emerald-400" />
                <span>Classification Output</span>
              </div>
              <span className="text-xs font-mono uppercase bg-slate-900 px-2 py-0.5 rounded text-emerald-400 border border-slate-800">
                {algorithm.toUpperCase()} Model
              </span>
            </h3>

            {loading ? (
              <div className="py-16 flex flex-col items-center justify-center text-center space-y-4">
                <div className="w-12 h-12 rounded-full border-2 border-emerald-500/20 border-t-emerald-500 animate-spin flex items-center justify-center">
                  <Leaf className="w-5 h-5 text-emerald-400" />
                </div>
                <div>
                  <p className="text-xs font-bold text-slate-200">Running Inception-V3 Deep Feature Extraction...</p>
                  <p className="text-[11px] text-slate-400 mt-1">Inception CNN Bottleneck Layer Representation (2048-dim vector)</p>
                </div>
              </div>
            ) : error ? (
              <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-start space-x-3">
                <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold">Prediction Failed</div>
                  <div className="mt-0.5">{error}</div>
                </div>
              </div>
            ) : result ? (
              <div className="space-y-6">
                {/* Species Badge */}
                <div className="p-4 rounded-xl bg-gradient-to-br from-slate-900 to-slate-950 border border-emerald-500/30">
                  <div className="text-[11px] font-semibold text-emerald-400 uppercase tracking-wider mb-1">
                    Predicted Species
                  </div>
                  <div className="text-xl font-extrabold text-white mb-1 capitalize">
                    {result.predictedClass}
                  </div>
                  <div className="text-xs text-slate-400 flex items-center space-x-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Inception-V3 Deep Vector Verified by Tribuo</span>
                  </div>
                </div>

                {/* Confidence Meter */}
                <div>
                  <div className="flex items-center justify-between text-xs mb-1.5">
                    <span className="text-slate-400 font-medium">Confidence Score</span>
                    <span className="font-bold text-emerald-400 font-mono">
                      {(result.confidenceScore * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden p-0.5 border border-slate-800">
                    <div
                      className="bg-gradient-to-r from-emerald-500 to-teal-300 h-full rounded-full transition-all duration-1000"
                      style={{ width: `${Math.min(100, Math.max(10, result.confidenceScore * 100))}%` }}
                    />
                  </div>
                </div>

                {/* Feature Statistics */}
                <div className="grid grid-cols-2 gap-3 pt-4 border-t border-slate-800">
                  <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold mb-1 flex items-center space-x-1">
                      <Layers className="w-3 h-3 text-emerald-400" />
                      <span>Selected Features</span>
                    </div>
                    <div className="text-base font-extrabold text-white font-mono">
                      {result.featureCount} <span className="text-xs text-slate-400 font-normal">/ 2048</span>
                    </div>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold mb-1 flex items-center space-x-1">
                      <Sparkles className="w-3 h-3 text-teal-400" />
                      <span>Feature Pruning</span>
                    </div>
                    <div className="text-base font-extrabold text-teal-400 font-mono">
                      {(((2048 - result.featureCount) / 2048) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="py-12 flex flex-col items-center justify-center text-center space-y-3 text-slate-400">
                <Leaf className="w-10 h-10 text-slate-700" />
                <p className="text-xs">
                  Upload an image on the left or select a 1-click specimen preset to view predictions.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
