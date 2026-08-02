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

import java.io.IOException;
import java.nio.file.Paths;

public class MainClass_Bagging {
    public static void main(String[] args) throws IOException {
        System.out.println("=================================================================");
        System.out.println("   BATCH BAGGING ENSEMBLE MODEL TRAINER (BASELINE BCS)           ");
        System.out.println("=================================================================\n");

        // List of candidate Baseline BCS feature-selected dataset paths (Excludes
        // partial train/test splits)
        String[][] datasetConfigs = {
                { "Swedish", "Entire Data Folder/Swedish After FS/Swedish After FS.csv", "Swedish After FS.csv" },
                { "Flavia", "Entire Data Folder/Flavia After FS/Flavia After FS.csv", "Flavia After FS.csv" },
                { "Philippine", "Entire Data Folder/Philippine After FS/Philippine After FS.csv",
                        "Philippine After FS.csv" }
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
                System.out.println("SKIP: No feature-selected dataset found for " + datasetName + " (Baseline BCS).");
                continue;
            }

            System.out.println("-----------------------------------------------------------------");
            System.out.println("Training Baseline BCS Model for: " + datasetName);
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

            // Configure Bagging Trainer (10 weak learners)
            var baggingTrainer = new BaggingTrainer<>(
                    fmTrainer,
                    new VotingCombiner(),
                    10,
                    Trainer.DEFAULT_SEED);

            var sTrain = System.currentTimeMillis();
            var trainedModel = baggingTrainer.train(dataset);
            var eTrain = System.currentTimeMillis();

            // Save trained model (e.g., "Swedish_Plant_Model_bcs.ser")
            String outputModelName = datasetName + "_Plant_Model_bcs.ser";
            System.out.println("Saving model to: " + outputModelName);
            try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(
                    new java.io.FileOutputStream(outputModelName))) {
                oos.writeObject(trainedModel);
                System.out.println("SUCCESS: Model saved as '" + outputModelName + "' in project root.");
            } catch (java.io.IOException e) {
                System.err.println("Error saving model: " + e.getMessage());
            }

            var evaluator = new LabelEvaluator().evaluate(trainedModel, dataset);
            System.out.println("\nRESULTS for " + datasetName + " (Baseline BCS Bagging):");
            System.out.println("  Training Duration: " + Util.formatDuration(sTrain, eTrain));
            System.out.printf("  Accuracy: %.2f%%\n", evaluator.accuracy() * 100.0);
            System.out.printf("  Macro Recall: %.2f%%\n", evaluator.macroAveragedRecall() * 100.0);
            System.out.printf("  Macro Precision: %.2f%%\n", evaluator.macroAveragedPrecision() * 100.0);
            System.out.printf("  Macro F1-Score: %.2f%%\n", evaluator.macroAveragedF1() * 100.0);
        }

        System.out.println("\n=================================================================");
        System.out.println("SUCCESS: All Baseline BCS models trained and saved!");
        System.out.println("=================================================================");
    }
}
