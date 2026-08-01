package WrapperCuckooSearchForFS.org.Main;

import WrapperCuckooSearchForFS.org.Discreeting.TransferFunction;
import WrapperCuckooSearchForFS.org.Optimizers.GeneticCuckooSearchOptimizer;
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

/**
 * Proposed Genetic Binary Cuckoo Search (GBCS) Main Execution Pipeline.
 * <p>
 * Performs feature selection using the GBCS framework (Lévy flights + Uniform
 * Crossover + Bit Flip Mutation
 * + Correlation-Aware Fitness Function) on plant leaf feature embeddings.
 * </p>
 * Reference: Proposed Paper (Fernandez, Oclarit, and Yos 2025).
 */
public class MainClass_GBCS_For_FS_and_Bagging {
    public static void main(String[] args) throws IOException {
        // Find all raw dataset CSV files inside Entire Data Folder/Original Dataset/
        java.io.File originalDir = Paths.get("Entire Data Folder", "Original Dataset").toFile();
        java.io.File[] rawCsvFiles = originalDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".csv"));

        if (rawCsvFiles == null || rawCsvFiles.length == 0) {
            System.err.println("ERROR: No raw dataset CSV files found in: " + originalDir.getAbsolutePath());
            return;
        }

        System.out.println("=================================================================");
        System.out.println("   BATCH PROPOSED GENETIC BINARY CUCKOO SEARCH (GBCS) FEATURE SELECTION   ");
        System.out.println("=================================================================\n");

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

            // KNN wrapper trainer used inside GBCS evaluation
            var learner = new KNNTrainer<>(1,
                    new L2Distance(),
                    Runtime.getRuntime().availableProcessors(),
                    new VotingCombiner(),
                    KNNModel.Backend.THREADPOOL,
                    NeighboursQueryFactoryType.BRUTE_FORCE);

            // Instantiate proposed Genetic Binary Cuckoo Search (GBCS) Optimizer
            var optimizer = new GeneticCuckooSearchOptimizer(
                    learner,
                    TransferFunction.V2,
                    30,      // population size N
                    2.0d,    // step size scaling A
                    2.0d,    // lambda
                    0.1d,    // worst nest probability pa
                    1.5d,    // delta
                    0.8d,    // crossover rate Pc
                    0.02d,   // mutation rate Pm
                    20,      // max iterations
                    12345    // seed
            );

            var sDate = System.currentTimeMillis();
            var selectedFeatureSet = optimizer.select(trainData);
            var eDate = System.currentTimeMillis();
            var selectedFeatureDataset = new SelectedFeatureDataset<>(trainData, selectedFeatureSet);

            // Ensure target folder exists inside Entire Data Folder/
            java.io.File targetDir = Paths.get("Entire Data Folder", datasetPrefix + " After GBCS-FS").toFile();
            if (!targetDir.exists()) {
                targetDir.mkdirs();
            }

            // Export per-dataset GBCS convergence history inside target folder
            String outputCsvName = datasetPrefix + " After GBCS-FS.csv";
            java.nio.file.Path outputPath = targetDir.toPath().resolve(outputCsvName);
            new CSVSaver().save(outputPath, selectedFeatureDataset, "Class");

            String csvConvergencePath = targetDir.toPath().resolve(datasetPrefix + "_GBCS_Convergence_History.csv").toString();
            optimizer.exportConvergenceCSV(csvConvergencePath);

            double reductionRatio = (1.0 - ((double) selectedFeatureDataset.size() / trainData.getFeatureMap().size())) * 100.0;
            int optIter = optimizer.getOptimalConvergenceIteration();

            System.out.println("\nRESULTS for " + datasetPrefix + " (GBCS):");
            System.out.printf("  FS Duration: %s\n", Util.formatDuration(sDate, eDate));
            System.out.printf("  Optimal Convergence Iteration: Iteration %d (out of 20)\n", optIter);
            System.out.printf("  Original Feature Count: %d\n", trainData.getFeatureMap().size());
            System.out.printf("  Selected Feature Count: %d\n", selectedFeatureDataset.size());
            System.out.printf("  Feature Reduction Ratio (FRR): %.2f%%\n", reductionRatio);
            System.out.println("  Saved Dataset To: " + outputPath);
            System.out.println("  Saved Convergence To: " + csvConvergencePath);
        }

        System.out.println("\n=================================================================");
        System.out.println("SUCCESS: All raw datasets processed with Proposed GBCS!");
        System.out.println("=================================================================");
    }
}
