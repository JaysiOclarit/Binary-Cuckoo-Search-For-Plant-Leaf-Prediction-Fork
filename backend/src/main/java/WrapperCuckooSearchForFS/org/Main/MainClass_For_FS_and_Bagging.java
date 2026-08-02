package WrapperCuckooSearchForFS.org.Main;

import WrapperCuckooSearchForFS.org.Discreeting.TransferFunction;
import WrapperCuckooSearchForFS.org.Optimizers.CuckooSearchOptimizer;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.common.nearest.KNNModel;
import org.tribuo.common.nearest.KNNTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.data.csv.CSVSaver;
import org.tribuo.dataset.SelectedFeatureDataset;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.math.distance.L2Distance;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.util.Util;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class MainClass_For_FS_and_Bagging {
        public static void main(String[] args) throws IOException {
                // Find all raw dataset CSV files inside Entire Data Folder/Original Dataset/
                java.io.File originalDir = Paths.get("Entire Data Folder", "Original Dataset").toFile();
                java.io.File[] rawCsvFiles = originalDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".csv"));

                if (rawCsvFiles == null || rawCsvFiles.length == 0) {
                        System.err.println(
                                        "ERROR: No raw dataset CSV files found in: " + originalDir.getAbsolutePath());
                        return;
                }

                System.out.println("=================================================================");
                System.out.println("   BATCH BASELINE BINARY CUCKOO SEARCH (BCS) FEATURE SELECTION   ");
                System.out.println("=================================================================\n");

                record FSSummary(String dataset, int originalFeatures, int selectedFeatures, double reductionRatio,
                                String duration) {
                }
                List<FSSummary> summaryList = new ArrayList<>();

                for (java.io.File csvFile : rawCsvFiles) {
                        String fileName = csvFile.getName();
                        // Extract dataset prefix (e.g. "Swedish", "Flavia", "Philippine")
                        String datasetPrefix = fileName.split(" ")[0];

                        System.out.println("-----------------------------------------------------------------");
                        System.out.println("Processing Raw Dataset: " + fileName + " (" + datasetPrefix + ")");
                        System.out.println("Path: " + csvFile.getAbsolutePath());

                        var dataSource = new CSVLoader<>(new LabelFactory()).loadDataSource(csvFile.toPath(), "Class");
                        var dataSplitting = new TrainTestSplitter<>(dataSource, 0.6, Trainer.DEFAULT_SEED);
                        var trainData = new MutableDataset<>(dataSplitting.getTrain());

                        // Learner & Optimizer setup
                        var learner = new KNNTrainer<>(1,
                                        new L2Distance(),
                                        Runtime.getRuntime().availableProcessors(),
                                        new VotingCombiner(),
                                        KNNModel.Backend.THREADPOOL,
                                        NeighboursQueryFactoryType.BRUTE_FORCE);

                        var optimizer = new CuckooSearchOptimizer(learner,
                                        TransferFunction.V2,
                                        30, 2d, 2d, 0.1d, 1.5d, 20, 12345);

                        var sDate = System.currentTimeMillis();
                        var SFS = optimizer.select(trainData);
                        var eDate = System.currentTimeMillis();
                        var SFDS = new SelectedFeatureDataset<>(trainData, SFS);

                        // Ensure target folder exists inside Entire Data Folder/
                        java.io.File targetDir = Paths.get("Entire Data Folder", datasetPrefix + " After FS").toFile();
                        if (!targetDir.exists()) {
                                targetDir.mkdirs();
                        }

                        // Export per-dataset convergence history inside target folder
                        String outputCsvName = datasetPrefix + " After FS.csv";
                        java.nio.file.Path outputPath = targetDir.toPath().resolve(outputCsvName);
                        new CSVSaver().save(outputPath, SFDS, "Class");

                        String csvConvergencePath = targetDir.toPath()
                                        .resolve(datasetPrefix + "_BCS_Convergence_History.csv").toString();
                        optimizer.exportConvergenceCSV(csvConvergencePath);

                        int origFeatureCount = trainData.getFeatureMap().size();
                        int selFeatureCount = SFDS.getFeatureMap().size();
                        double reductionRatio = (1.0 - ((double) selFeatureCount / origFeatureCount)) * 100.0;
                        int optIter = optimizer.getOptimalConvergenceIteration();
                        String durationStr = Util.formatDuration(sDate, eDate);

                        summaryList.add(new FSSummary(datasetPrefix, origFeatureCount, selFeatureCount, reductionRatio,
                                        durationStr));

                        System.out.println("\nRESULTS for " + datasetPrefix + " (BCS):");
                        System.out.printf("  FS Duration: %s\n", durationStr);
                        System.out.printf("  Optimal Convergence Iteration: Iteration %d (out of 20)\n", optIter);
                        System.out.printf("  Original Feature Count: %d\n", origFeatureCount);
                        System.out.printf("  Selected Feature Count: %d\n", selFeatureCount);
                        System.out.printf("  Feature Reduction Ratio (FRR): %.2f%%\n", reductionRatio);
                        System.out.println("  Saved Dataset To: " + outputPath);
                        System.out.println("  Saved Convergence To: " + csvConvergencePath);
                }

                System.out.println(
                                "\n=========================================================================================");
                System.out.println("   FINAL BATCH BASELINE BCS FEATURE SELECTION SUMMARY TABLE");
                System.out.println(
                                "=========================================================================================");
                System.out.printf("%-15s | %-12s | %-12s | %-12s | %-15s\n", "Dataset", "Orig Features", "Sel Features",
                                "FRR (%)", "Duration");
                System.out.println(
                                "-----------------------------------------------------------------------------------------");
                for (FSSummary s : summaryList) {
                        System.out.printf("%-15s | %-12d | %-12d | %-11.2f%% | %-15s\n",
                                        s.dataset(), s.originalFeatures(), s.selectedFeatures(), s.reductionRatio(),
                                        s.duration());
                }
                System.out.println(
                                "=========================================================================================");
                System.out.println("SUCCESS: All raw datasets processed with Baseline BCS!");
                System.out.println("=================================================================");
        }
}