package WrapperCuckooSearchForFS.org.Optimizers;

import WrapperCuckooSearchForFS.org.Discreeting.TransferFunction;
import WrapperCuckooSearchForFS.org.Evaluation.CorrelationAwareFitnessFunction;
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
 * Proposed Genetic Binary Cuckoo Search (GBCS) Algorithm for Feature Selection.
 * <p>
 * Combines Binary Cuckoo Search with Lévy flights, Genetic Operators (Uniform
 * Crossover & Bit Flip Mutation),
 * and a Correlation-Aware Fitness Function to prevent premature convergence and
 * reduce feature redundancy.
 * </p>
 * Reference: Proposed Paper (Fernandez, Oclarit, Yos, 2025).
 */
public final class GeneticCuckooSearchOptimizer implements FeatureSelector<Label> {
    private final TransferFunction transferFunction;
    private final double stepSizeScaling;
    private final double lambda;
    private final double worstNestProbability;
    private final double delta;
    private final double crossoverRate;
    private final double mutationRate;
    private final int populationSize;
    private int[][] setOfSolutions;
    private final CorrelationAwareFitnessFunction fitnessFunction;
    private final int maxIteration;
    private final SplittableRandom rng;
    private final int seed;

    /**
     * Default constructor for GBCS using KNN (K=1, L2 distance) and default
     * parameters.
     */
    public GeneticCuckooSearchOptimizer() {
        this.transferFunction = TransferFunction.V2;
        this.populationSize = 30;
        KNNTrainer<Label> knnTrainer = new KNNTrainer<>(1,
                new L2Distance(),
                Runtime.getRuntime().availableProcessors(),
                new VotingCombiner(),
                KNNModel.Backend.THREADPOOL,
                NeighboursQueryFactoryType.BRUTE_FORCE);
        this.fitnessFunction = new CorrelationAwareFitnessFunction(knnTrainer);
        this.stepSizeScaling = 2d;
        this.lambda = 2d;
        this.worstNestProbability = 0.1d;
        this.delta = 1.5d;
        this.crossoverRate = 0.8d;
        this.mutationRate = 0.02d;
        this.maxIteration = 20;
        this.seed = 12345;
        this.rng = new SplittableRandom(seed);
    }

    /**
     * Constructor for GBCS with custom wrapper trainer and core GBCS parameters.
     */
    public GeneticCuckooSearchOptimizer(Trainer<Label> trainer,
            TransferFunction transferFunction,
            int populationSize,
            double stepSizeScaling,
            double lambda,
            double worstNestProbability,
            double delta,
            double crossoverRate,
            double mutationRate,
            int maxIteration,
            int seed) {
        this.transferFunction = transferFunction;
        this.populationSize = populationSize;
        this.fitnessFunction = new CorrelationAwareFitnessFunction(trainer);
        this.stepSizeScaling = stepSizeScaling;
        this.lambda = lambda;
        this.worstNestProbability = worstNestProbability;
        this.delta = delta;
        this.crossoverRate = crossoverRate;
        this.mutationRate = mutationRate;
        this.maxIteration = maxIteration;
        this.seed = seed;
        this.rng = new SplittableRandom(seed);
    }

    /**
     * Generates initial random binary population.
     */
    private int[][] generatePopulation(int totalNumberOfFeatures) {
        setOfSolutions = new int[this.populationSize][totalNumberOfFeatures];
        for (int[] subSet : setOfSolutions) {
            for (int i = 0; i < subSet.length; i++) {
                subSet[i] = rng.nextInt(2);
            }
        }
        return setOfSolutions;
    }

    @Override
    public boolean isOrdered() {
        return true;
    }

    @Override
    public SelectedFeatureSet select(Dataset<Label> dataset) {
        ImmutableFeatureMap FMap = new ImmutableFeatureMap(dataset.getFeatureMap());
        int numFeatures = FMap.size();
        setOfSolutions = generatePopulation(numFeatures);

        List<GBCSFeatureSet> solutionScores = new ArrayList<>();
        SelectedFeatureSet selectedFeatureSet = null;

        for (int i = 0; i < maxIteration; i++) {
            final int iter = i + 1;

            for (int solutionIdx = 0; solutionIdx < setOfSolutions.length; solutionIdx++) {
                int[] currentNest = setOfSolutions[solutionIdx];

                // 1. Global Search: Lévy Flight Update
                int[] evolvedSolution = Arrays.stream(currentNest)
                        .map(x -> (int) transferFunction.applyAsDouble(x + stepSizeScaling * Math.pow(iter, -lambda)))
                        .toArray();

                // 2. Genetic Operators: Uniform Crossover
                if (rng.nextDouble() < crossoverRate) {
                    int partnerIdx = rng.nextInt(setOfSolutions.length);
                    int[] partnerNest = setOfSolutions[partnerIdx];
                    for (int j = 0; j < numFeatures; j++) {
                        if (rng.nextBoolean()) {
                            evolvedSolution[j] = partnerNest[j];
                        }
                    }
                }

                // 3. Genetic Operators: Bit Flip Mutation
                for (int j = 0; j < numFeatures; j++) {
                    if (rng.nextDouble() < mutationRate) {
                        evolvedSolution[j] = evolvedSolution[j] == 1 ? 0 : 1; // Flip bit
                    }
                }

                // Compare with random nest and keep best
                int[] randomCuckoo = setOfSolutions[rng.nextInt(setOfSolutions.length)];
                keepBestAfterEvaluation(dataset, FMap, evolvedSolution, randomCuckoo);

                // 4. Local Search: Abandon Worst Nests (Random Walk)
                if (rng.nextDouble() < worstNestProbability) {
                    int r1 = rng.nextInt(setOfSolutions.length);
                    int r2 = rng.nextInt(setOfSolutions.length);
                    double randDelta = rng.nextDouble() * delta;
                    for (int j = 0; j < numFeatures; j++) {
                        evolvedSolution[j] = (int) transferFunction.applyAsDouble(
                                currentNest[j] + randDelta * (setOfSolutions[r1][j] - setOfSolutions[r2][j]));
                    }
                    keepBestAfterEvaluation(dataset, FMap, evolvedSolution, currentNest);
                }
            }

            // Evaluate current population fitness scores
            solutionScores.clear();
            for (int[] solution : setOfSolutions) {
                double score = fitnessFunction.EvaluateSolution(this, dataset, FMap, solution);
                solutionScores.add(new GBCSFeatureSet(solution, score));
            }

            solutionScores.sort(Comparator.comparing(GBCSFeatureSet::score).reversed());

            // Track per-iteration convergence metrics
            int[] bestSubSet = solutionScores.get(0).subSet;
            double bestScore = solutionScores.get(0).score;
            int selectedCount = 0;
            for (int bit : bestSubSet)
                if (bit == 1)
                    selectedCount++;
            double redRatio = (1.0 - ((double) selectedCount / numFeatures)) * 100.0;
            convergenceHistory.add(new ConvergenceStep(iter, bestScore, selectedCount, redRatio));

            selectedFeatureSet = fitnessFunction.getSFS(this, dataset, FMap, bestSubSet);
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
                pw.printf("%d,%.6f,%d,%.2f%%\n", step.iteration(), step.bestFitness(), step.selectedFeatures(),
                        step.reductionRatioPercent());
            }
        }
    }

    /**
     * Finds the iteration number where the peak/optimal fitness score was first
     * reached.
     */
    public int getOptimalConvergenceIteration() {
        if (convergenceHistory.isEmpty())
            return -1;
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

    private void keepBestAfterEvaluation(Dataset<Label> dataset, ImmutableFeatureMap FMap, int[] alteredSolution,
            int[] oldSolution) {
        double scoreOfModifiedSolution = fitnessFunction.EvaluateSolution(this, dataset, FMap, alteredSolution);
        double scoreOfSolution = fitnessFunction.EvaluateSolution(this, dataset, FMap, oldSolution);
        if (scoreOfModifiedSolution > scoreOfSolution) {
            System.arraycopy(alteredSolution, 0, oldSolution, 0, alteredSolution.length);
        }
    }

    /**
     * Record holding per-iteration convergence statistics.
     */
    public record ConvergenceStep(int iteration, double bestFitness, int selectedFeatures,
            double reductionRatioPercent) {
    }

    record GBCSFeatureSet(int[] subSet, double score) {
    }
}
