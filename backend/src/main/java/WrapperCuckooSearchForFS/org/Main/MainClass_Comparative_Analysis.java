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

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Automated Batch Execution Pipeline for Comparative Analysis between Baseline BCS and Proposed GBCS.
 * <p>
 * Scans the Original Dataset directory, runs live feature selection with both algorithms on each raw dataset
 * (Swedish, Flavia, Philippine), trains FM + Bagging ensembles under identical 60/40 splits, and exports 
 * individual & final summary comparative reports.
 * </p>
 * Reference: Section 2.5 of Proposed Paper (Fernandez et al., 2025).
 */
public class MainClass_Comparative_Analysis {

    public record BatchComparativePair(
            String datasetName,
            AlgorithmResult bcsResult,
            AlgorithmResult gbcsResult
    ) {}

    public static void main(String[] args) throws IOException {
        System.out.println("=================================================================");
        System.out.println("   AUTOMATED BATCH COMPARATIVE ANALYSIS PIPELINE                 ");
        System.out.println("   (Baseline BCS vs Proposed GBCS across all raw datasets)       ");
        System.out.println("=================================================================\n");

        // 1. Locate Raw Datasets Directory
        File originalDataDir = new File(Paths.get("Entire Data Folder", "Original Dataset").toString());
        if (!originalDataDir.exists() || !originalDataDir.isDirectory()) {
            originalDataDir = new File(Paths.get("backend", "Entire Data Folder", "Original Dataset").toString());
        }

        if (!originalDataDir.exists() || !originalDataDir.isDirectory()) {
            System.err.println("ERROR: Cannot locate 'Entire Data Folder/Original Dataset' directory!");
            return;
        }

        File[] datasetFiles = originalDataDir.listFiles((dir, name) -> 
            name.toLowerCase().endsWith(".csv") && 
            !name.toLowerCase().contains("train") && 
            !name.toLowerCase().contains("test") &&
            !name.toLowerCase().contains("after") &&
            !name.toLowerCase().contains("split")
        );

        if (datasetFiles == null || datasetFiles.length == 0) {
            System.err.println("ERROR: No raw dataset CSV files found in: " + originalDataDir.getAbsolutePath());
            return;
        }

        // Shared Trainers
        var knnTrainer = new KNNTrainer<>(1,
                new L2Distance(),
                Runtime.getRuntime().availableProcessors(),
                new VotingCombiner(),
                KNNModel.Backend.THREADPOOL,
                NeighboursQueryFactoryType.BRUTE_FORCE);

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

        List<BatchComparativePair> batchResults = new ArrayList<>();

        // 2. Loop through all raw datasets
        for (File datasetFile : datasetFiles) {
            String fileName = datasetFile.getName();
            String datasetPrefix = fileName.replace(" Leaf data.csv", "")
                                           .replace(" dataset.csv", "")
                                           .replace(".csv", "").trim();

            System.out.println("\n-----------------------------------------------------------------");
            System.out.println("Processing Batch Comparative Analysis for: " + datasetPrefix);
            System.out.println("Path: " + datasetFile.getAbsolutePath());
            System.out.println("-----------------------------------------------------------------");

            var dataSource = new CSVLoader<>(new LabelFactory()).loadDataSource(datasetFile.toPath(), "Class");
            var dataSplitting = new TrainTestSplitter<>(dataSource, 0.6, Trainer.DEFAULT_SEED);
            var trainData = new MutableDataset<>(dataSplitting.getTrain());
            var testData = new MutableDataset<>(dataSplitting.getTest());

            int totalOriginalFeatures = trainData.getFeatureMap().size();

            // Target directory inside Entire Data Folder
            File targetDir = new File(datasetFile.getParentFile().getParentFile(), datasetPrefix + " Comparative Results");
            if (!targetDir.exists()) {
                targetDir.mkdirs();
            }

            // -------------------------------------------------------------------------
            // A. RUN BASELINE BINARY CUCKOO SEARCH (BCS)
            // -------------------------------------------------------------------------
            System.out.println("\n [1/2] Running Baseline Binary Cuckoo Search (BCS)...");
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
            bcsOptimizer.exportConvergenceCSV(targetDir.toPath().resolve(datasetPrefix + "_BCS_Convergence_History.csv").toString());

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

            // -------------------------------------------------------------------------
            // B. RUN PROPOSED GENETIC BINARY CUCKOO SEARCH (GBCS)
            // -------------------------------------------------------------------------
            System.out.println("\n [2/2] Running Proposed Genetic Binary Cuckoo Search (GBCS)...");
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
            gbcsOptimizer.exportConvergenceCSV(targetDir.toPath().resolve(datasetPrefix + "_GBCS_Convergence_History.csv").toString());

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

            // Save individual dataset reports
            printAndSaveComparison(datasetPrefix, bcsResult, gbcsResult, targetDir);

            batchResults.add(new BatchComparativePair(datasetPrefix, bcsResult, gbcsResult));
        }

        // 3. Print Final Batch Summary Table
        printFinalBatchSummaryTable(batchResults);
    }

    private static void printAndSaveComparison(String datasetPrefix,
                                                AlgorithmResult bcs,
                                                AlgorithmResult gbcs,
                                                File targetDir) throws IOException {

        int speedupIter = bcs.optimalConvergenceIteration() - gbcs.optimalConvergenceIteration();
        double accDiff = gbcs.accuracy() - bcs.accuracy();
        double frrDiff = gbcs.reductionRatioPercent() - bcs.reductionRatioPercent();
        double corrDiff = bcs.averageCorrelation() - gbcs.averageCorrelation();

        StringBuilder report = new StringBuilder();
        report.append("=================================================================================\n");
        report.append("     COMPARATIVE ANALYSIS REPORT: BASELINE BCS vs PROPOSED GBCS (" + datasetPrefix.toUpperCase() + ")\n");
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

        System.out.println(report.toString());

        Path reportPath = targetDir.toPath().resolve(datasetPrefix + "_BCS_vs_GBCS_Comparison_Report.txt");
        try (PrintWriter pw = new PrintWriter(new FileWriter(reportPath.toFile()))) {
            pw.print(report.toString());
        }

        Path csvPath = targetDir.toPath().resolve(datasetPrefix + "_Comparative_Results.csv");
        try (PrintWriter csv = new PrintWriter(new FileWriter(csvPath.toFile()))) {
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
    }

    private static void printFinalBatchSummaryTable(List<BatchComparativePair> batchResults) {
        System.out.println("\n=========================================================================================================");
        System.out.println("   FINAL BATCH COMPARATIVE ANALYSIS SUMMARY TABLE (BASELINE BCS vs PROPOSED GBCS)");
        System.out.println("=========================================================================================================");
        System.out.printf("%-12s | %-12s | %-10s | %-10s | %-10s | %-10s | %-10s\n", 
                "Dataset", "Algorithm", "Sel Feat", "FRR (%)", "rho_avg", "Acc (%)", "F1 (%)");
        System.out.println("---------------------------------------------------------------------------------------------------------");

        for (BatchComparativePair pair : batchResults) {
            AlgorithmResult bcs = pair.bcsResult();
            AlgorithmResult gbcs = pair.gbcsResult();

            System.out.printf("%-12s | %-12s | %-10d | %-10.2f | %-10.4f | %-10.2f | %-10.2f\n",
                    pair.datasetName(), bcs.algorithmName(), bcs.selectedFeatures(), bcs.reductionRatioPercent(), bcs.averageCorrelation(), bcs.accuracy(), bcs.f1Score());
            System.out.printf("%-12s | %-12s | %-10d | %-10.2f | %-10.4f | %-10.2f | %-10.2f\n",
                    pair.datasetName(), gbcs.algorithmName(), gbcs.selectedFeatures(), gbcs.reductionRatioPercent(), gbcs.averageCorrelation(), gbcs.accuracy(), gbcs.f1Score());
            System.out.println("---------------------------------------------------------------------------------------------------------");
        }
        System.out.println("=========================================================================================================");
        System.out.println("SUCCESS: All raw datasets evaluated in batch for Comparative Analysis!");
        System.out.println("=========================================================================================================");
    }
}
