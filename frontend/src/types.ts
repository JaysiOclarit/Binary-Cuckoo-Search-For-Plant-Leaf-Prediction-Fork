export interface PredictionResult {
  predictedClass: string;
  confidenceScore: number;
  dataset: string;
  algorithm: string;
  featureCount: number;
}

export interface SubspaceProfilePoint {
  category: string;
  BCS: number;
  GBCS: number;
}

export interface ComparisonResult {
  dataset: string;
  bcsPredictedClass: string;
  bcsConfidence: number;
  bcsFeatureCount: number;
  bcsReductionRatio: number;
  gbcsPredictedClass: string;
  gbcsConfidence: number;
  gbcsFeatureCount: number;
  gbcsReductionRatio: number;
  winner: string;
  radarProfile?: SubspaceProfilePoint[];
}

export interface AnalyticsMetric {
  dataset: string;
  algorithm: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  featuresSelected: number;
  reductionRatio: number;
}

export interface ConvergencePoint {
  iteration: number;
  gbcsFitness: number;
  bcsFitness: number;
}
