package WrapperCuckooSearchForFS.org.Main;

import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.fm.FMClassificationTrainer;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.util.Util;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

/**
 * Trains and saves the final Factorization Machine (FM) + Bagging Ensemble
 * Model
 * built on features selected by the Proposed Genetic Binary Cuckoo Search
 * (GBCS).
 */
public class MainClass_GBCS_Bagging {
    public static void main(String[] args) throws IOException {
        System.out.println("=================================================================");
        System.out.println("   BATCH BAGGING ENSEMBLE MODEL TRAINER (PROPOSED GBCS)          ");
        System.out.println("=================================================================\n");

        // List of candidate Proposed GBCS feature-selected dataset paths
        String[][] datasetConfigs = {
                { "Swedish", "Entire Data Folder/Swedish After GBCS-FS/Swedish After GBCS-FS.csv",
                        "Swedish After GBCS-FS.csv" },
                { "Flavia", "Entire Data Folder/Flavia After GBCS-FS/Flavia After GBCS-FS.csv",
                        "Flavia After GBCS-FS.csv" },
                { "Philippine", "Entire Data Folder/Philippine After GBCS-FS/Philippine After GBCS-FS.csv",
                        "Philippine After GBCS-FS.csv" }
        };

        for (String[] config : datasetConfigs) {
            String datasetName = config[0];
            java.io.File resolvedFile = null;

            for (int i = 1; i < config.length; i++) {
                java.io.File candidate = new java.io.File(config[i]);
                if (candidate.exists() && !candidate.isDirectory()) {
                    resolvedFile = candidate;
                    break;
                }
            }

            if (resolvedFile == null) {
                System.out.println("SKIP: No GBCS feature-selected dataset found for " + datasetName + ".");
                continue;
            }

            System.out.println("-----------------------------------------------------------------");
            System.out.println("Training Proposed GBCS Model for: " + datasetName);
            System.out.println("Path: " + resolvedFile.getAbsolutePath());

            var dataSource = new CSVLoader<>(new LabelFactory()).loadDataSource(resolvedFile.toPath(), "Class");
            var dataset = new MutableDataset<>(dataSource);

            // Configure FM Trainer
            var fmTrainer = new FMClassificationTrainer(
                    new Hinge(),
                    new AdaGrad(0.1, 0.5),
                    50,
                    Trainer.DEFAULT_SEED,
                    10,
                    0.2D);

            // Configure Bagging Ensemble Trainer (10 weak learners)
            var baggingTrainer = new BaggingTrainer<>(
                    fmTrainer,
                    new VotingCombiner(),
                    10,
                    Trainer.DEFAULT_SEED);

            var sTrain = System.currentTimeMillis();
            var trainedModel = baggingTrainer.train(dataset);
            var eTrain = System.currentTimeMillis();

            // Save trained model for web API deployment (e.g.,
            // "Swedish_Plant_Model_gbcs.ser")
            String outputModelName = datasetName + "_Plant_Model_gbcs.ser";
            java.io.File modelsDir = ModelExporter.getModelsDir();
            java.io.File targetFile = new java.io.File(modelsDir, outputModelName);

            System.out.println("Saving model to: " + targetFile.getPath());
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(targetFile))) {
                oos.writeObject(trainedModel);
                System.out.println("SUCCESS: Model saved as '" + targetFile.getPath() + "'.");
            } catch (IOException e) {
                System.err.println("Error saving model: " + e.getMessage());
            }

            var evaluator = new LabelEvaluator().evaluate(trainedModel, dataset);
            System.out.println("\nRESULTS for " + datasetName + " (Proposed GBCS Bagging):");
            System.out.println("  Training Duration: " + Util.formatDuration(sTrain, eTrain));
            System.out.printf("  Accuracy: %.2f%%\n", evaluator.accuracy() * 100.0);
            System.out.printf("  Macro Recall: %.2f%%\n", evaluator.macroAveragedRecall() * 100.0);
            System.out.printf("  Macro Precision: %.2f%%\n", evaluator.macroAveragedPrecision() * 100.0);
            System.out.printf("  Macro F1-Score: %.2f%%\n", evaluator.macroAveragedF1() * 100.0);
        }

        System.out.println("\n=================================================================");
        System.out.println("SUCCESS: All Proposed GBCS models trained and saved!");
        System.out.println("=================================================================");
    }
}
