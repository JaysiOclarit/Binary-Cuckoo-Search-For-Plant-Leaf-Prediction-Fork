import React, { useState, useEffect } from 'react';
import { Database, Search, Tag, MapPin, Stethoscope, Sparkles } from 'lucide-react';
import { PlantSpecies } from '../types';

interface BotanicalEncyclopediaProps {
  onFetchPlants: () => Promise<PlantSpecies[]>;
}

export const BotanicalEncyclopedia: React.FC<BotanicalEncyclopediaProps> = ({ onFetchPlants }) => {
  const [plants, setPlants] = useState<PlantSpecies[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filterDataset, setFilterDataset] = useState<string>('all');

  useEffect(() => {
    onFetchPlants().then(setPlants).catch(console.error);
  }, []);

  const filteredPlants = plants.filter((plant) => {
    const matchesSearch =
      plant.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      plant.scientificName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      plant.family.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesDataset =
      filterDataset === 'all' || plant.dataset.toLowerCase() === filterDataset.toLowerCase();
    return matchesSearch && matchesDataset;
  });

  return (
    <div className="space-y-8">
      {/* Title & Filters */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center space-x-2">
            <Database className="w-6 h-6 text-emerald-400" />
            <span>Botanical Encyclopedia & Specimen Catalog</span>
          </h2>
          <p className="text-slate-400 text-sm mt-1">
            Explore plant leaf species across Swedish, Flavia, and Philippine botanical datasets.
          </p>
        </div>

        <div className="flex items-center space-x-3 self-start">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search species, family..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-slate-900 text-xs text-slate-200 pl-9 pr-4 py-2 rounded-xl border border-slate-800 focus:outline-none focus:border-emerald-500 w-48 lg:w-64"
            />
          </div>

          <select
            value={filterDataset}
            onChange={(e) => setFilterDataset(e.target.value)}
            className="bg-slate-900 text-xs text-slate-200 px-3 py-2 rounded-xl border border-slate-800 focus:outline-none focus:border-emerald-500"
          >
            <option value="all">All Datasets</option>
            <option value="swedish">Swedish Dataset</option>
            <option value="flavia">Flavia Dataset</option>
            <option value="philippine">Philippine Dataset</option>
          </select>
        </div>
      </div>

      {/* Species Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredPlants.map((plant) => (
          <div
            key={plant.name}
            className="glass-card rounded-2xl p-6 border border-slate-800 hover:border-emerald-500/40 transition-all group hover:-translate-y-1"
          >
            <div className="flex items-start justify-between mb-3">
              <div>
                <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  {plant.dataset} Dataset
                </span>
                <h3 className="text-lg font-bold text-white group-hover:text-emerald-400 transition-colors mt-2">
                  {plant.name}
                </h3>
                <div className="text-xs italic text-emerald-400 font-medium">
                  {plant.scientificName}
                </div>
              </div>
            </div>

            <p className="text-xs text-slate-300 mb-4 line-clamp-3 leading-relaxed">
              {plant.description}
            </p>

            <div className="space-y-2 pt-3 border-t border-slate-800/80 text-xs">
              <div className="flex items-center text-slate-400 space-x-2">
                <Tag className="w-3.5 h-3.5 text-slate-500" />
                <span>Family: <strong className="text-slate-200 font-semibold">{plant.family}</strong></span>
              </div>

              <div className="flex items-center text-slate-400 space-x-2">
                <MapPin className="w-3.5 h-3.5 text-slate-500" />
                <span>Region: <strong className="text-slate-200 font-semibold">{plant.region}</strong></span>
              </div>

              <div className="flex items-start text-slate-400 space-x-2 pt-1">
                <Stethoscope className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                <div className="flex flex-wrap gap-1">
                  {plant.uses.map((use) => (
                    <span key={use} className="text-[10px] bg-slate-900 text-slate-300 px-2 py-0.5 rounded border border-slate-800">
                      {use}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
