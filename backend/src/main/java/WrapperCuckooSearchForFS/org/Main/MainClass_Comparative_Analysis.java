package WrapperCuckooSearchForFS.org.Main;

import WrapperCuckooSearchForFS.org.Discreeting.TransferFunction;
import WrapperCuckooSearchForFS.org.Evaluation.ComparativeEvaluator;
import WrapperCuckooSearchForFS.org.Evaluation.ComparativeEvaluator.AlgorithmResult;
import WrapperCuckooSearchForFS.org.Optimizers.CuckooSearchOptimizer;
import WrapperCuckooSearchForFS.org.Optimizers.GeneticCuckooSearchOptimizer;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.fm.FMClassificationTrainer;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.common.nearest.KNNModel;
import org.tribuo.common.nearest.KNNTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.dataset.SelectedFeatureDataset;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.math.distance.L2Distance;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.util.Util;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Paths;

/**
 * Main Execution Pipeline for Comparative Analysis between Baseline BCS and Proposed GBCS.
 * <p>
 * Evaluates both feature selection algorithms under identical dataset split conditions and
 * compares accuracy, precision, recall, F1-score, feature reduction ratio (FRR), average
 * feature correlation (rho_avg), and convergence speedup.
 * </p>
 * Reference: Section 2.5 of Proposed Paper (Fernandez et al., 2025).
 */
public class MainClass_Comparative_Analysis {

    public static void main(String[] args) throws IOException {
        String dataPath = Paths.get("Entire Data Folder", "Original Dataset", "Swedish Leaf data.csv").toString();
        System.out.println("=================================================");
        System.out.println("Starting Comparative Analysis Experiment");
        System.out.println("Loading Dataset: " + dataPath);
        System.out.println("=================================================");

        var dataSource = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(dataPath), "Class");
        var dataSplitting = new TrainTestSplitter<>(dataSource, 0.6, Trainer.DEFAULT_SEED);
        var trainData = new MutableDataset<>(dataSplitting.getTrain());
        var testData = new MutableDataset<>(dataSplitting.getTest());

        int totalOriginalFeatures = trainData.getFeatureMap().size();

        // Shared KNN trainer for feature selection wrapper evaluation
        var knnTrainer = new KNNTrainer<>(1,
                new L2Distance(),
                Runtime.getRuntime().availableProcessors(),
                new VotingCombiner(),
                KNNModel.Backend.THREADPOOL,
                NeighboursQueryFactoryType.BRUTE_FORCE);

        // Shared FM + Bagging Trainer (10 weak learners) for final classification evaluation
        var fmTrainer = new FMClassificationTrainer(new Hinge(),
                new AdaGrad(0.1, 0.5),
                50,
                Trainer.DEFAULT_SEED,
                10,
                0.2D);
        var baggingTrainer = new BaggingTrainer<>(fmTrainer,
                new VotingCombiner(),
                10,
                Trainer.DEFAULT_SEED);

        // =========================================================================
        // 1. RUN BASELINE BINARY CUCKOO SEARCH (BCS)
        // =========================================================================
        System.out.println("\n[1/2] Running Baseline Binary Cuckoo Search (BCS)...");
        var bcsOptimizer = new CuckooSearchOptimizer(knnTrainer,
                TransferFunction.V2,
                30, 2.0d, 2.0d, 0.1d, 1.5d, 20, 12345);

        long bcsStart = System.currentTimeMillis();
        var bcsSFS = bcsOptimizer.select(trainData);
        long bcsEnd = System.currentTimeMillis();

        var bcsTrainSFDS = new SelectedFeatureDataset<>(trainData, bcsSFS);
        var bcsTestSFDS = new SelectedFeatureDataset<>(testData, bcsSFS);

        double bcsAvgCorr = ComparativeEvaluator.computeAverageFeatureCorrelation(bcsTrainSFDS);
        int bcsOptIter = bcsOptimizer.getOptimalConvergenceIteration();
        bcsOptimizer.exportConvergenceCSV("BCS_Convergence_History.csv");

        // Train FM + Bagging Ensemble on BCS features
        var bcsModel = baggingTrainer.train(bcsTrainSFDS);
        var bcsEval = new LabelEvaluator().evaluate(bcsModel, bcsTestSFDS);

        int bcsSelCount = bcsTrainSFDS.getFeatureMap().size();
        AlgorithmResult bcsResult = new AlgorithmResult(
                "Baseline BCS",
                (bcsEnd - bcsStart),
                totalOriginalFeatures,
                bcsSelCount,
                (1.0 - ((double) bcsSelCount / totalOriginalFeatures)) * 100.0,
                bcsAvgCorr,
                bcsOptIter,
                bcsEval.accuracy() * 100.0,
                bcsEval.macroAveragedPrecision() * 100.0,
                bcsEval.macroAveragedRecall() * 100.0,
                bcsEval.macroAveragedF1() * 100.0
        );

        // =========================================================================
        // 2. RUN PROPOSED GENETIC BINARY CUCKOO SEARCH (GBCS)
        // =========================================================================
        System.out.println("\n[2/2] Running Proposed Genetic Binary Cuckoo Search (GBCS)...");
        var gbcsOptimizer = new GeneticCuckooSearchOptimizer(knnTrainer,
                TransferFunction.V2,
                30, 2.0d, 2.0d, 0.1d, 1.5d, 0.8d, 0.02d, 20, 12345);

        long gbcsStart = System.currentTimeMillis();
        var gbcsSFS = gbcsOptimizer.select(trainData);
        long gbcsEnd = System.currentTimeMillis();

        var gbcsTrainSFDS = new SelectedFeatureDataset<>(trainData, gbcsSFS);
        var gbcsTestSFDS = new SelectedFeatureDataset<>(testData, gbcsSFS);

        double gbcsAvgCorr = ComparativeEvaluator.computeAverageFeatureCorrelation(gbcsTrainSFDS);
        int gbcsOptIter = gbcsOptimizer.getOptimalConvergenceIteration();
        gbcsOptimizer.exportConvergenceCSV("GBCS_Convergence_History.csv");

        // Train FM + Bagging Ensemble on GBCS features
        var gbcsModel = baggingTrainer.train(gbcsTrainSFDS);
        var gbcsEval = new LabelEvaluator().evaluate(gbcsModel, gbcsTestSFDS);

        int gbcsSelCount = gbcsTrainSFDS.getFeatureMap().size();
        AlgorithmResult gbcsResult = new AlgorithmResult(
                "Proposed GBCS",
                (gbcsEnd - gbcsStart),
                totalOriginalFeatures,
                gbcsSelCount,
                (1.0 - ((double) gbcsSelCount / totalOriginalFeatures)) * 100.0,
                gbcsAvgCorr,
                gbcsOptIter,
                gbcsEval.accuracy() * 100.0,
                gbcsEval.macroAveragedPrecision() * 100.0,
                gbcsEval.macroAveragedRecall() * 100.0,
                gbcsEval.macroAveragedF1() * 100.0
        );

        // =========================================================================
        // 3. GENERATE COMPARATIVE REPORT & EXPORT RESULTS
        // =========================================================================
        int speedupIterations = bcsOptIter - gbcsOptIter;
        double accuracyDiff = gbcsResult.accuracy() - bcsResult.accuracy();
        double frrDiff = gbcsResult.reductionRatioPercent() - bcsResult.reductionRatioPercent();
        double corrDiff = bcsResult.averageCorrelation() - gbcsResult.averageCorrelation();

        printAndSaveComparison(bcsResult, gbcsResult, speedupIterations, accuracyDiff, frrDiff, corrDiff);
    }

    private static void printAndSaveComparison(AlgorithmResult bcs,
                                                AlgorithmResult gbcs,
                                                int speedupIter,
                                                double accDiff,
                                                double frrDiff,
                                                double corrDiff) throws IOException {

        StringBuilder report = new StringBuilder();
        report.append("=================================================================================\n");
        report.append("               COMPARATIVE ANALYSIS REPORT: BASELINE BCS vs PROPOSED GBCS        \n");
        report.append("=================================================================================\n\n");

        report.append(String.format("%-32s | %-20s | %-20s\n", "Metric / Evaluation Parameter", "Baseline BCS", "Proposed GBCS (Ours)"));
        report.append("---------------------------------------------------------------------------------\n");
        report.append(String.format("%-32s | %-20s | %-20s\n", "FS Execution Time", Util.formatDuration(0, bcs.fsDurationMs()), Util.formatDuration(0, gbcs.fsDurationMs())));
        report.append(String.format("%-32s | %-20d | %-20d\n", "Original Dataset Features", bcs.totalOriginalFeatures(), gbcs.totalOriginalFeatures()));
        report.append(String.format("%-32s | %-20d | %-20d\n", "Selected Feature Subset Size", bcs.selectedFeatures(), gbcs.selectedFeatures()));
        report.append(String.format("%-32s | %-20.2f%% | %-20.2f%%\n", "Feature Reduction Ratio (FRR)", bcs.reductionRatioPercent(), gbcs.reductionRatioPercent()));
        report.append(String.format("%-32s | %-20.4f | %-20.4f\n", "Avg Feature Correlation (rho_avg)", bcs.averageCorrelation(), gbcs.averageCorrelation()));
        report.append(String.format("%-32s | Iteration %-10d | Iteration %-10d\n", "Optimal Convergence Iteration", bcs.optimalConvergenceIteration(), gbcs.optimalConvergenceIteration()));
        report.append("---------------------------------------------------------------------------------\n");
        report.append(String.format("%-32s | %-20.2f%% | %-20.2f%%\n", "Test Classification Accuracy", bcs.accuracy(), gbcs.accuracy()));
        report.append(String.format("%-32s | %-20.2f%% | %-20.2f%%\n", "Macro-Averaged Precision", bcs.precision(), gbcs.precision()));
        report.append(String.format("%-32s | %-20.2f%% | %-20.2f%%\n", "Macro-Averaged Recall", bcs.recall(), gbcs.recall()));
        report.append(String.format("%-32s | %-20.2f%% | %-20.2f%%\n", "Macro-Averaged F1-Score", bcs.f1Score(), gbcs.f1Score()));
        report.append("=================================================================================\n\n");

        report.append("KEY HIGHLIGHTS & IMPROVEMENTS (GBCS vs BCS):\n");
        report.append(String.format("  - Accuracy Improvement: %+.2f%%\n", accDiff));
        report.append(String.format("  - Feature Reduction Improvement: %+.2f%%\n", frrDiff));
        report.append(String.format("  - Feature Redundancy Reduction: %+.4f lower average correlation\n", corrDiff));
        report.append(String.format("  - Convergence Speedup: %d iterations faster stabilization\n", speedupIter));
        report.append("=================================================================================\n");

        // 1. Output to console
        System.out.println(report.toString());

        // 2. Save text report to file
        try (PrintWriter pw = new PrintWriter(new FileWriter("BCS_vs_GBCS_Comparison_Report.txt"))) {
            pw.print(report.toString());
        }

        // 3. Save CSV summary for automated graphing/tables
        try (PrintWriter csv = new PrintWriter(new FileWriter("Comparative_Results.csv"))) {
            csv.println("Metric,Baseline_BCS,Proposed_GBCS,Difference");
            csv.printf("ClassificationAccuracy,%.4f,%.4f,%+.4f\n", bcs.accuracy(), gbcs.accuracy(), accDiff);
            csv.printf("Precision,%.4f,%.4f,%+.4f\n", bcs.precision(), gbcs.precision(), gbcs.precision() - bcs.precision());
            csv.printf("Recall,%.4f,%.4f,%+.4f\n", bcs.recall(), gbcs.recall(), gbcs.recall() - bcs.recall());
            csv.printf("F1Score,%.4f,%.4f,%+.4f\n", bcs.f1Score(), gbcs.f1Score(), gbcs.f1Score() - bcs.f1Score());
            csv.printf("SelectedFeatures,%d,%d,%d\n", bcs.selectedFeatures(), gbcs.selectedFeatures(), gbcs.selectedFeatures() - bcs.selectedFeatures());
            csv.printf("ReductionRatioPercent,%.2f,%.2f,%+.2f\n", bcs.reductionRatioPercent(), gbcs.reductionRatioPercent(), frrDiff);
            csv.printf("AverageCorrelation,%.4f,%.4f,%+.4f\n", bcs.averageCorrelation(), gbcs.averageCorrelation(), -corrDiff);
            csv.printf("OptimalConvergenceIteration,%d,%d,%d\n", bcs.optimalConvergenceIteration(), gbcs.optimalConvergenceIteration(), -speedupIter);
        }

        System.out.println("SUCCESS: Results saved to 'BCS_vs_GBCS_Comparison_Report.txt' and 'Comparative_Results.csv'");
    }
}
