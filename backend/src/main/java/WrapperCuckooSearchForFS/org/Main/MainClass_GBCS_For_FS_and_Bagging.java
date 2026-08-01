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
 * Performs feature selection using the GBCS framework (Lévy flights + Uniform Crossover + Bit Flip Mutation
 * + Correlation-Aware Fitness Function) on plant leaf feature embeddings.
 * </p>
 * Reference: Proposed Paper (Fernandez, Oclarit, Yos, & Vilchez, 2025).
 */
public class MainClass_GBCS_For_FS_and_Bagging {
    public static void main(String[] args) throws IOException {
        // Read input dataset (Swedish, Flavia, or Philippine Medicinal Plant dataset)
        var dataPath = Paths.get("Entire Data Folder", "Original Dataset", "Swedish Leaf data.csv").toString();
        System.out.println("Loading dataset from: " + dataPath);
        var dataSource = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(dataPath), "Class");

        // Split data into train/test sets
        var dataSplitting = new TrainTestSplitter<>(dataSource, 0.6, Trainer.DEFAULT_SEED);
        var trainData = new MutableDataset<>(dataSplitting.getTrain());
        var testData = new MutableDataset<>(dataSplitting.getTest());

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

        System.out.println("Starting Genetic Binary Cuckoo Search (GBCS) Feature Selection...");
        var sDate = System.currentTimeMillis();
        var selectedFeatureSet = optimizer.select(trainData);
        var eDate = System.currentTimeMillis();

        var selectedFeatureDataset = new SelectedFeatureDataset<>(trainData, selectedFeatureSet);

        // Save the resulting selected feature subset
        var outputPath = Paths.get(System.getProperty("user.dir"), "Swedish After GBCS-FS.csv");
        new CSVSaver().save(outputPath, selectedFeatureDataset, "Class");

        // Export convergence history to CSV for plotting & documentation
        String csvPath = "GBCS_Convergence_History.csv";
        optimizer.exportConvergenceCSV(csvPath);

        double reductionRatio = (1.0 - ((double) selectedFeatureDataset.size() / trainData.getFeatureMap().size())) * 100.0;
        int optIter = optimizer.getOptimalConvergenceIteration();

        System.out.println("=================================================");
        System.out.println("GBCS Feature Selection Analysis & Results:");
        System.out.printf("FS Duration: %s\n", Util.formatDuration(sDate, eDate));
        System.out.printf("Convergence Speed (Optimal Iteration): Iteration %d (out of 20)\n", optIter);
        System.out.printf("Original Feature Count: %d\n", trainData.getFeatureMap().size());
        System.out.printf("Selected Feature Subset Size: %d\n", selectedFeatureDataset.size());
        System.out.printf("Feature Reduction Ratio (FRR): %.2f%%\n", reductionRatio);
        System.out.println("Convergence History Exported To: " + csvPath);
        System.out.println("Output Dataset Saved To: " + outputPath);
        System.out.println("=================================================");
    }
}
