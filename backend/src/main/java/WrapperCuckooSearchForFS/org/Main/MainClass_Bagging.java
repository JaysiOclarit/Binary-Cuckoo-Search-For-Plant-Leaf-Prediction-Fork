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
        // read the entire dataset after the FS process
        var dataPath = Paths.get("Entire Data Folder", "Philippine After IBCS-FS", "Philippine After FS(2).csv").toString();
        var dataSource = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(dataPath), "Class");
        var Data = new MutableDataset<>(dataSource);

        // use FM classifier
        var FMTrainer = new FMClassificationTrainer(new Hinge(),
                new AdaGrad(0.1, 0.5),
                50,
                Trainer.DEFAULT_SEED,
                10,
                0.2D);

        // use bagging for ensimple learning
        var trainer = new BaggingTrainer<>(FMTrainer,
                new VotingCombiner(),
                10,
                Trainer.DEFAULT_SEED);

        // train the model
        var sTrain = System.currentTimeMillis();
        var ensembleLearningTrainer = trainer.train(Data);
        var eTrain = System.currentTimeMillis();

        // ---------------------------------------------------------
        // NEW CODE: Save the fully trained model to your hard drive
        // ---------------------------------------------------------
        System.out.println("Saving model to disk...");
        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream("Philippine_Plant_Model.ser"))) {
            oos.writeObject(ensembleLearningTrainer);
            System.out.println("Success! Model saved as 'Philippine_Plant_Model.ser' in your project root.");
        } catch (java.io.IOException e) {
            System.err.println("Error saving the model: " + e.getMessage());
        }
        // ---------------------------------------------------------

        // define evaluater to test the model to get the output
        var labelEvaluator = new LabelEvaluator().evaluate(ensembleLearningTrainer, Data);

        System.out.println("The Training_Testing duration time is : " + Util.formatDuration(sTrain, eTrain));
        System.out.println("The average accuracy is : " + labelEvaluator.accuracy());
        System.out.println("The average recall is : " + labelEvaluator.macroAveragedRecall());
        System.out.println("The average F1-Score is : " + labelEvaluator.macroAveragedF1());
        System.out.println("The average precision is : " + labelEvaluator.macroAveragedPrecision());
    }
}
