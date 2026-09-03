package WrapperCuckooSearchForFS.org.Evaluation;

import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.classification.Label;

/**
 * Utility for evaluating comparative metrics between Baseline BCS and Proposed GBCS models.
 * <p>
 * Computes average Pearson feature correlation (rho_avg), feature reduction ratio (FRR),
 * and formats side-by-side comparison statistics for research paper documentation.
 * </p>
 * Reference: Section 2.4.4 & 2.5 of Proposed Paper (Fernandez et al., 2025).
 */
public final class ComparativeEvaluator {

    private ComparativeEvaluator() {}

    /**
     * Calculates the average pairwise absolute Pearson correlation coefficient (rho_avg)
     * among the features present in a dataset.
     */
    public static double computeAverageFeatureCorrelation(Dataset<Label> dataset) {
        ImmutableFeatureMap fmap = new ImmutableFeatureMap(dataset.getFeatureMap());
        int numFeatures = fmap.size();
        int numInstances = dataset.size();

        if (numFeatures <= 1 || numInstances <= 1) {
            return 0.0D;
        }

        double[][] featureData = new double[numFeatures][numInstances];
        int instanceIdx = 0;
        for (Example<Label> example : dataset) {
            for (int f = 0; f < numFeatures; f++) {
                Feature feature = example.lookup(fmap.get(f).getName());
                featureData[f][instanceIdx] = (feature != null) ? feature.getValue() : 0.0D;
            }
            instanceIdx++;
        }

        double[] means = new double[numFeatures];
        double[] stds = new double[numFeatures];
        for (int f = 0; f < numFeatures; f++) {
            double sum = 0.0;
            for (int i = 0; i < numInstances; i++) {
                sum += featureData[f][i];
            }
            means[f] = sum / numInstances;

            double sqSum = 0.0;
            for (int i = 0; i < numInstances; i++) {
                double diff = featureData[f][i] - means[f];
                sqSum += diff * diff;
            }
            stds[f] = Math.sqrt(sqSum);
        }

        double sumAbsCorr = 0.0;
        int pairCount = 0;
        for (int i = 0; i < numFeatures; i++) {
            for (int j = i + 1; j < numFeatures; j++) {
                if (stds[i] == 0 || stds[j] == 0) {
                    continue;
                }
                double covariance = 0.0;
                for (int k = 0; k < numInstances; k++) {
                    covariance += (featureData[i][k] - means[i]) * (featureData[j][k] - means[j]);
                }
                double r = covariance / (stds[i] * stds[j]);
                if (Double.isNaN(r)) r = 0.0D;
                sumAbsCorr += Math.abs(r);
                pairCount++;
            }
        }

        return pairCount > 0 ? (sumAbsCorr / pairCount) : 0.0D;
    }

    /**
     * Data structure holding comparative execution results for one algorithm.
     */
    public record AlgorithmResult(
            String algorithmName,
            long fsDurationMs,
            int totalOriginalFeatures,
            int selectedFeatures,
            double reductionRatioPercent,
            double averageCorrelation,
            int optimalConvergenceIteration,
            double accuracy,
            double precision,
            double recall,
            double f1Score
    ) {}
}
