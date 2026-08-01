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
import java.nio.file.Paths;

/**
 * Trains and saves the final Factorization Machine (FM) + Bagging Ensemble Model 
 * built on features selected by the Proposed Genetic Binary Cuckoo Search (GBCS).
 */
public class MainClass_GBCS_Bagging {
    public static void main(String[] args) throws IOException {
        // Read dataset after GBCS feature selection
        var dataPath = Paths.get("Entire Data Folder", "Philippine After IBCS-FS", "Philippine After FS(2).csv").toString();
        System.out.println("Loading GBCS feature-selected dataset from: " + dataPath);
        var dataSource = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(dataPath), "Class");
        var dataset = new MutableDataset<>(dataSource);

        // Configure FM Classifier
        var fmTrainer = new FMClassificationTrainer(
                new Hinge(),
                new AdaGrad(0.1, 0.5),
                50,
                Trainer.DEFAULT_SEED,
                10,
                0.2D
        );

        // Configure Bagging Ensemble Trainer (10 weak learners)
        var baggingTrainer = new BaggingTrainer<>(
                fmTrainer,
                new VotingCombiner(),
                10,
                Trainer.DEFAULT_SEED
        );

        System.out.println("Training Proposed GBCS FM-Bagging Ensemble Model...");
        var sTrain = System.currentTimeMillis();
        var trainedModel = baggingTrainer.train(dataset);
        var eTrain = System.currentTimeMillis();

        // Save trained model to disk for web API deployment
        String outputModelFileName = "Philippine_GBCS_Model.ser";
        System.out.println("Saving GBCS trained model to disk as: " + outputModelFileName);
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(outputModelFileName))) {
            oos.writeObject(trainedModel);
            System.out.println("SUCCESS: Model saved as '" + outputModelFileName + "' in project root.");
        } catch (IOException e) {
            System.err.println("Error saving model: " + e.getMessage());
        }

        // Evaluate model
        var evaluator = new LabelEvaluator().evaluate(trainedModel, dataset);

        System.out.println("=========================================");
        System.out.println("Proposed GBCS Model Training Results:");
        System.out.println("Training Duration: " + Util.formatDuration(sTrain, eTrain));
        System.out.printf("Accuracy: %.2f%%\n", evaluator.accuracy() * 100.0);
        System.out.printf("Macro Recall: %.2f%%\n", evaluator.macroAveragedRecall() * 100.0);
        System.out.printf("Macro Precision: %.2f%%\n", evaluator.macroAveragedPrecision() * 100.0);
        System.out.printf("Macro F1-Score: %.2f%%\n", evaluator.macroAveragedF1() * 100.0);
        System.out.println("=========================================");
    }
}
