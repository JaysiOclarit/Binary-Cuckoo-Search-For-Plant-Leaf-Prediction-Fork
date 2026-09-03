package WrapperCuckooSearchForFS.org.Evaluation;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.FeatureSelector;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.dataset.SelectedFeatureDataset;
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.provenance.FeatureSetProvenance;

import java.util.ArrayList;
import java.util.List;

/**
 * Proposed Correlation-Aware Fitness Function for Genetic Binary Cuckoo Search (GBCS).
 * <p>
 * Evaluates feature subsets using a multi-objective function:
 * Fitness = w1 * Acc + w2 * (1 - |S|/|F|) + w3 * (1 - rho_avg)
 * where:
 * - w1 = 0.8 (Classification accuracy weight via 10-fold CV with KNN)
 * - w2 = 0.1 (Feature reduction weight)
 * - w3 = 0.1 (Feature independence / low correlation weight)
 * - rho_avg = average absolute Pearson correlation coefficient among selected features
 * </p>
 * Reference: Section 2.4.4 of Proposed Paper (Fernandez, Oclarit, Yos, & Vilchez, 2025).
 */
public final class CorrelationAwareFitnessFunction {
    private final Trainer<Label> trainer;
    private final double w1;
    private final double w2;
    private final double w3;
    
    // Cache for precomputed correlation matrix of the dataset
    private double[][] correlationMatrix;
    private Dataset<Label> cachedDataset;

    /**
     * Constructor with default weights (w1=0.8, w2=0.1, w3=0.1) as specified in Proposed Paper.
     * @param trainer The KNN trainer used for wrapper evaluation
     */
    public CorrelationAwareFitnessFunction(Trainer<Label> trainer) {
        this(trainer, 0.8D, 0.1D, 0.1D);
    }

    /**
     * Constructor with custom weights for multi-objective optimization.
     * @param trainer The trainer used for wrapper evaluation
     * @param w1 Weight for classification accuracy
     * @param w2 Weight for feature subset size reduction
     * @param w3 Weight for feature correlation penalty (independence reward)
     */
    public CorrelationAwareFitnessFunction(Trainer<Label> trainer, double w1, double w2, double w3) {
        this.trainer = trainer;
        this.w1 = w1;
        this.w2 = w2;
        this.w3 = w3;
    }

    /**
     * Computes the Correlation-Aware Fitness score for a candidate binary solution mask.
     * @param optimizer The feature selector optimizer
     * @param dataset The dataset being evaluated
     * @param Fmap The feature map of the dataset
     * @param solution The binary feature selection mask (1 for selected, 0 for excluded)
     * @return The multi-objective fitness score
     */
    public <T extends FeatureSelector<Label>> double EvaluateSolution(T optimizer, Dataset<Label> dataset, ImmutableFeatureMap Fmap, int[] solution) {
        // 1. Ensure Pearson correlation matrix is precomputed for the dataset
        ensureCorrelationMatrix(dataset, Fmap);

        // 2. Extract selected features dataset and 10-fold CV Accuracy
        SelectedFeatureSet sfs = getSFS(optimizer, dataset, Fmap, solution);
        List<Integer> selectedIndices = new ArrayList<>();
        for (int i = 0; i < solution.length; i++) {
            if (solution[i] == 1) {
                selectedIndices.add(i);
            }
        }

        // If no features are selected, fitness is 0
        if (selectedIndices.isEmpty()) {
            return 0.0D;
        }

        SelectedFeatureDataset<Label> selectedFeatureDataset = new SelectedFeatureDataset<>(dataset, sfs);
        CrossValidation<Label, LabelEvaluation> crossValidation = new CrossValidation<>(trainer, selectedFeatureDataset, new LabelEvaluator(), 10);
        double avgAccuracy = 0D;
        for (Pair<LabelEvaluation, Model<Label>> ACC : crossValidation.evaluate()) {
            avgAccuracy += ACC.getA().accuracy();
        }
        avgAccuracy /= crossValidation.getK();

        // 3. Compute Feature Reduction Ratio: (1 - |S| / |F|)
        int sizeOfSubset = selectedIndices.size();
        int sizeOfDataset = Fmap.size();
        double reductionRatio = 1.0D - ((double) sizeOfSubset / sizeOfDataset);

        // 4. Compute Average Absolute Pearson Correlation (rho_avg) among selected feature pairs
        double rhoAvg = computeAverageCorrelation(selectedIndices);

        // 5. Final Multi-Objective Fitness
        // Fitness = w1 * Acc + w2 * (1 - |S|/|F|) + w3 * (1 - rho_avg)
        double fitness = (w1 * avgAccuracy) + (w2 * reductionRatio) + (w3 * (1.0D - rhoAvg));
        return fitness;
    }

    /**
     * Precomputes the full feature-by-feature Pearson Correlation Matrix once per dataset.
     */
    private synchronized void ensureCorrelationMatrix(Dataset<Label> dataset, ImmutableFeatureMap Fmap) {
        if (cachedDataset == dataset && correlationMatrix != null) {
            return;
        }

        int numFeatures = Fmap.size();
        int numInstances = dataset.size();
        double[][] featureData = new double[numFeatures][numInstances];

        // Extract feature columns
        int instanceIdx = 0;
        for (Example<Label> example : dataset) {
            for (int f = 0; f < numFeatures; f++) {
                org.tribuo.Feature feature = example.lookup(Fmap.get(f).getName());
                featureData[f][instanceIdx] = (feature != null) ? feature.getValue() : 0.0D;
            }
            instanceIdx++;
        }

        // Compute means and standard deviations
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

        // Compute pairwise Pearson correlation
        correlationMatrix = new double[numFeatures][numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            correlationMatrix[i][i] = 1.0D;
            for (int j = i + 1; j < numFeatures; j++) {
                if (stds[i] == 0 || stds[j] == 0) {
                    correlationMatrix[i][j] = 0.0D;
                    correlationMatrix[j][i] = 0.0D;
                    continue;
                }
                double covariance = 0.0;
                for (int k = 0; k < numInstances; k++) {
                    covariance += (featureData[i][k] - means[i]) * (featureData[j][k] - means[j]);
                }
                double r = covariance / (stds[i] * stds[j]);
                if (Double.isNaN(r)) r = 0.0D;
                correlationMatrix[i][j] = r;
                correlationMatrix[j][i] = r;
            }
        }

        cachedDataset = dataset;
    }

    /**
     * Calculates the average absolute Pearson correlation (rho_avg) for a subset of selected feature indices.
     */
    private double computeAverageCorrelation(List<Integer> selectedIndices) {
        int n = selectedIndices.size();
        if (n <= 1) {
            return 0.0D; // No pairs exist, correlation penalty is 0 (independence reward is 1.0)
        }

        double sumAbsCorr = 0.0D;
        int pairCount = 0;
        for (int i = 0; i < n; i++) {
            int f1 = selectedIndices.get(i);
            for (int j = i + 1; j < n; j++) {
                int f2 = selectedIndices.get(j);
                sumAbsCorr += Math.abs(correlationMatrix[f1][f2]);
                pairCount++;
            }
        }

        return pairCount > 0 ? (sumAbsCorr / pairCount) : 0.0D;
    }

    /**
     * Constructs the SelectedFeatureSet for Tribuo.
     */
    public <T extends FeatureSelector<Label>> SelectedFeatureSet getSFS(T optimizer, Dataset<Label> dataset, ImmutableFeatureMap Fmap, int[] solution) {
        List<String> names = new ArrayList<>();
        List<Double> scores = new ArrayList<>();
        for (int i = 0; i < solution.length; i++) {
            if (solution[i] == 1) {
                names.add(Fmap.get(i).getName());
                scores.add(1D);
            }
        }
        FeatureSetProvenance provenance = new FeatureSetProvenance(SelectedFeatureSet.class.getName(),
                dataset.getProvenance(),
                optimizer.getProvenance());
        return new SelectedFeatureSet(names, scores, optimizer.isOrdered(), provenance);
    }
}
