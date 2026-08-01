package WrapperCuckooSearchForFS.org.Optimizers;

import WrapperCuckooSearchForFS.org.Discreeting.TransferFunction;
import WrapperCuckooSearchForFS.org.Evaluation.FitnessFunction;
import org.tribuo.Dataset;
import org.tribuo.FeatureSelector;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.common.nearest.KNNModel;
import org.tribuo.common.nearest.KNNTrainer;
import org.tribuo.math.distance.L2Distance;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.provenance.FeatureSelectorProvenance;
import org.tribuo.provenance.impl.FeatureSelectorProvenanceImpl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Select features based on Cuckoo Search algorithm with binary transfer functions, KNN classifier and 10-fold cross validation
 * <p>
 * see:
 * <pre>
 * Xin-She Yang and Suash Deb.
 * "Cuckoo Search via L´evy Flights", 2010.
 *
 * L. A. M. Pereira et al.
 * "A Binary Cuckoo Search and its Application for Feature Selection", 2014.
 * </pre>
 */
public  final class CuckooSearchOptimizer implements FeatureSelector<Label> {
    private final TransferFunction transferFunction;
    private final double stepSizeScaling;
    private final double lambda;
    private final double worstNestProbability;
    private final double delta;
    private final int populationSize;
    private int [][] setOfSolutions;
    private final FitnessFunction FN;
    private final int maxIteration;
    private final SplittableRandom rng;
    private final int seed;

    /**
     * The default constructor for feature selection based on Cuckoo Search Algorithm
     */
    public CuckooSearchOptimizer() {
        this.transferFunction = TransferFunction.V2;
        this.populationSize = 50;
        KNNTrainer<Label> KnnTrainer =  new KNNTrainer<>(1,
                new L2Distance(),
                Runtime.getRuntime().availableProcessors(),
                new VotingCombiner(),
                KNNModel.Backend.THREADPOOL,
                NeighboursQueryFactoryType.BRUTE_FORCE);
        FN = new FitnessFunction(KnnTrainer);
        this.stepSizeScaling = 2d;
        this.lambda = 2d;
        this.worstNestProbability = 0.1d;
        this.delta = 1.5d;
        this.maxIteration = 30;
        this.seed = 12345;
        this.rng = new SplittableRandom(seed);
    }

    /**
     * Constructs the wrapper feature selection based on cuckoo search algorithm
     * @param trainer The used trainer in the evaluation process
     * @param transferFunction The transfer function to convert continuous values to binary ones
     * @param populationSize The size of the solution in the initial population
     * @param maxIteration The number of times that is used to enhance generation
     * @param seed This seed is required for the SplittableRandom
     */
    public CuckooSearchOptimizer(Trainer<Label> trainer, TransferFunction transferFunction, int populationSize, int maxIteration, int seed) {
        this.transferFunction = transferFunction;
        this.populationSize = populationSize;
        FN = new FitnessFunction(trainer);
        this.stepSizeScaling = 2d;
        this.lambda = 2d;
        this.worstNestProbability = 0.1d;
        this.delta = 1.5d;
        this.maxIteration = maxIteration;
        this.seed = seed;
        this.rng = new SplittableRandom(seed);
    }

    /**
     * Constructs the wrapper feature selection based on cuckoo search algorithm
     * @param trainer The used trainer in the evaluation process
     * @param transferFunction The transfer function to convert continuous values to binary ones
     * @param populationSize The size of the solution in the initial population
     * @param stepSizeScaling The cuckoo step size
     * @param lambda The lambda of the levy flight function
     * @param worstNestProbability The fraction of the nests to be abandoned
     * @param delta The delta that is used in the abandon nest function
     * @param maxIteration The number of times that is used to enhance generation
     * @param seed This seed is required for the SplittableRandom
     */
    public CuckooSearchOptimizer(Trainer<Label> trainer, TransferFunction transferFunction, int populationSize, double stepSizeScaling, double lambda, double worstNestProbability, double delta, int maxIteration, int seed) {
        this.transferFunction = transferFunction;
        this.populationSize = populationSize;
        FN = new FitnessFunction(trainer);
        this.stepSizeScaling = stepSizeScaling;
        this.lambda = lambda;
        this.worstNestProbability = worstNestProbability;
        this.delta = delta;
        this.maxIteration = maxIteration;
        this.seed = seed;
        this.rng = new SplittableRandom(seed);
    }
    /**
     * This method is used to generate the initial population (set of solutions)
     * @param totalNumberOfFeatures The number of features in the given dataset
     * @return The population of subsets of selected features
     */
    private int[][] GeneratePopulation(int totalNumberOfFeatures) {
        setOfSolutions = new int[this.populationSize][totalNumberOfFeatures];
        for (int[] subSet : setOfSolutions) {
            int[] values = new int[subSet.length];
            for (int i = 0; i < values.length; i++) {
                values[i] = rng.nextInt(2);
            }
            System.arraycopy(values, 0, subSet, 0, setOfSolutions[0].length);
        }
        return setOfSolutions;
    }

    /**
     * Does this feature selection algorithm return an ordered feature set?
     *
     * @return True if the set is ordered.
     */
    @Override
    public boolean isOrdered() {
        return true;
    }

    /**
     * Selects features according to this selection algorithm from the specified dataset.
     * @param dataset The dataset to use.
     * @return A selected feature set.
     */
    @Override
    public SelectedFeatureSet select(Dataset<Label> dataset) {
        ImmutableFeatureMap FMap = new ImmutableFeatureMap(dataset.getFeatureMap());
        setOfSolutions = GeneratePopulation(dataset.getFeatureMap().size());
        List<CuckooSearchFeatureSet> subSet_fScores = new ArrayList<>();
        SelectedFeatureSet selectedFeatureSet = null;
        for (int i = 0; i < maxIteration; i++) {
            final int iter = i + 1;
            for (int solution = 0; solution < setOfSolutions.length; solution++) {
                AtomicInteger subSet = new AtomicInteger(solution);
                // Update the solution based on the levy flight function
                int[] evolvedSolution = Arrays.stream(setOfSolutions[subSet.get()]).map(x -> (int) transferFunction.
                        applyAsDouble(x + stepSizeScaling * Math.pow(iter, -lambda))).toArray();
                int[] randomCuckoo = setOfSolutions[rng.nextInt(setOfSolutions.length)];
                keepBestAfterEvaluation(dataset, FMap, evolvedSolution, randomCuckoo);
                // Update the solution based on the abandon nest function
                if (rng.nextDouble() < worstNestProbability) {
                    int r1 = rng.nextInt(setOfSolutions.length);
                    int r2 = rng.nextInt(setOfSolutions.length);
                    double randDelta = rng.nextDouble() * delta;
                    for (int j = 0; j < setOfSolutions[subSet.get()].length; j++) {
                        evolvedSolution[j] = (int) transferFunction.
                                applyAsDouble(setOfSolutions[subSet.get()][j] +
                                randDelta * (setOfSolutions[r1][j] - setOfSolutions[r2][j]));
                    }
                    keepBestAfterEvaluation(dataset, FMap, evolvedSolution, setOfSolutions[subSet.get()]);
                }
            }
            subSet_fScores.clear();
            Arrays.stream(setOfSolutions).map(subSet -> new CuckooSearchFeatureSet(subSet,
                    FN.EvaluateSolution(this, dataset, FMap, subSet))).
                    forEach(subSet_fScores::add);
            subSet_fScores.sort(Comparator.comparing(CuckooSearchFeatureSet::score).reversed());
            
            // Track per-iteration convergence metrics
            int[] bestSubSet = subSet_fScores.get(0).subSet;
            double bestScore = subSet_fScores.get(0).score;
            int selectedCount = 0;
            for (int bit : bestSubSet) if (bit == 1) selectedCount++;
            double redRatio = (1.0 - ((double) selectedCount / dataset.getFeatureMap().size())) * 100.0;
            convergenceHistory.add(new ConvergenceStep(iter, bestScore, selectedCount, redRatio));

            selectedFeatureSet = FN.getSFS(this, dataset, FMap, bestSubSet);
        }
        return selectedFeatureSet;
    }

    private final List<ConvergenceStep> convergenceHistory = new ArrayList<>();

    /**
     * Returns the iteration-by-iteration convergence history.
     */
    public List<ConvergenceStep> getConvergenceHistory() {
        return new ArrayList<>(convergenceHistory);
    }

    /**
     * Exports the convergence metrics to a CSV file for documentation and analysis.
     */
    public void exportConvergenceCSV(String filePath) throws java.io.IOException {
        try (java.io.PrintWriter pw = new java.io.PrintWriter(new java.io.FileWriter(filePath))) {
            pw.println("Iteration,BestFitnessScore,SelectedFeatures,ReductionRatioPercent");
            for (ConvergenceStep step : convergenceHistory) {
                pw.printf("%d,%.6f,%d,%.2f%%\n", step.iteration(), step.bestFitness(), step.selectedFeatures(), step.reductionRatioPercent());
            }
        }
    }

    /**
     * Finds the iteration number where the peak/optimal fitness score was first reached.
     */
    public int getOptimalConvergenceIteration() {
        if (convergenceHistory.isEmpty()) return -1;
        double maxFitness = -1.0;
        int optIter = 1;
        for (ConvergenceStep step : convergenceHistory) {
            if (step.bestFitness() > maxFitness) {
                maxFitness = step.bestFitness();
                optIter = step.iteration();
            }
        }
        return optIter;
    }

    @Override
    public FeatureSelectorProvenance getProvenance() {
        return new FeatureSelectorProvenanceImpl(this);
    }

    /**
     * @param dataset The dataset to use
     * @param FMap The map of selected features
     * @param alteredSolution The modified solution
     * @param oldSolution The old solution
     */
    private void keepBestAfterEvaluation(Dataset<Label> dataset, ImmutableFeatureMap FMap, int[] alteredSolution, int[] oldSolution) {
        double scoreOfModifiedSolution = FN.EvaluateSolution(this, dataset, FMap, alteredSolution);
        double scoreOfSolution = FN.EvaluateSolution(this, dataset, FMap, oldSolution);
        if (scoreOfModifiedSolution > scoreOfSolution) {
            System.arraycopy(alteredSolution, 0, oldSolution, 0, alteredSolution.length);
        }
    }

    /**
     * Record holding per-iteration convergence statistics.
     */
    public record ConvergenceStep(int iteration, double bestFitness, int selectedFeatures, double reductionRatioPercent) { }

    /**
     * This record is used to hold subset of features with its corresponding fitness score
     */
    record CuckooSearchFeatureSet(int[] subSet, double score) { }
}