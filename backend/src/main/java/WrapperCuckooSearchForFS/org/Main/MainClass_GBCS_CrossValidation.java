package WrapperCuckooSearchForFS.org.Main;

import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.fm.FMClassificationTrainer;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.util.Util;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Paths;

/**
 * Executes K-Fold Cross-Validation for the Proposed Genetic Binary Cuckoo Search (GBCS)
 * feature-selected datasets and logs performance metrics for paper tables.
 */
public class MainClass_GBCS_CrossValidation {
    public static void main(String[] args) throws IOException {
        // Read dataset after GBCS feature selection
        var dataPath = Paths.get("Entire Data Folder", "Flavia After IBCS-FS", "Flavia After FS(2).csv").toString();
        System.out.println("Loading GBCS dataset from: " + dataPath);
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
                new org.tribuo.classification.ensemble.VotingCombiner(),
                10,
                Trainer.DEFAULT_SEED
        );

        // Manual K-Fold setting (e.g. K = 3, 5, 7, or 9)
        int k = 7;

        System.out.println("Starting GBCS Cross-Validation for K = " + k + "...");

        var crossValidation = new CrossValidation<>(baggingTrainer, dataset, new LabelEvaluator(), k);

        var avgAcc = 0D;
        var avgRecall = 0D;
        var avgF1 = 0D;
        var avgPrecision = 0D;
        var sTrain = System.currentTimeMillis();

        for (var performance : crossValidation.evaluate()) {
            avgAcc += performance.getA().accuracy();
            avgRecall += performance.getA().macroAveragedRecall();
            avgF1 += performance.getA().macroAveragedF1();
            avgPrecision += performance.getA().macroAveragedPrecision();
        }
        var eTrain = System.currentTimeMillis();

        double finalAcc = (avgAcc / crossValidation.getK()) * 100.0;
        double finalRecall = (avgRecall / crossValidation.getK()) * 100.0;
        double finalF1 = (avgF1 / crossValidation.getK()) * 100.0;
        double finalPrecision = (avgPrecision / crossValidation.getK()) * 100.0;
        String duration = Util.formatDuration(sTrain, eTrain);

        System.out.println("=========================================");
        System.out.println("Proposed GBCS Cross-Validation Results for K = " + k);
        System.out.println("Duration: " + duration);
        System.out.printf("Average Accuracy: %.2f%%\n", finalAcc);
        System.out.printf("Average Recall: %.2f%%\n", finalRecall);
        System.out.printf("Average F1-Score: %.2f%%\n", finalF1);
        System.out.printf("Average Precision: %.2f%%\n", finalPrecision);
        System.out.println("=========================================");

        // Append to CV_Manual_Results.txt for documentation
        try (FileWriter fw = new FileWriter("CV_Manual_Results.txt", true);
             PrintWriter pw = new PrintWriter(fw)) {

            pw.println("[Proposed GBCS Algorithm]");
            pw.println("=========================================");
            pw.println("Results for K = " + k);
            pw.println("Duration: " + duration);
            pw.printf("Average Accuracy: %.2f%%\n", finalAcc);
            pw.printf("Average Recall: %.2f%%\n", finalRecall);
            pw.printf("Average F1-Score: %.2f%%\n", finalF1);
            pw.printf("Average Precision: %.2f%%\n", finalPrecision);
            pw.println("=========================================\n");

            System.out.println("SUCCESS: Results appended to 'CV_Manual_Results.txt'");
        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }
    }
}
