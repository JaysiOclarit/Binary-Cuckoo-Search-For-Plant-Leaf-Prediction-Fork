package WrapperCuckooSearchForFS.org.Main;

import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.fm.FMClassificationTrainer;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.util.Util;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Paths;

public class MainClass_CrossValidation {
    public static void main(String[] args) throws IOException {
        // read the entire dataset after the FS process
        var dataPath = Paths.get("Entire Data Folder", "Flavia After IBCS-FS", "Flavia After FS(2).csv").toString();
        var dataSource = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(dataPath), "Class");
        var Data = new MutableDataset<>(dataSource);

        // use FM classifier
        var FMTrainer = new FMClassificationTrainer(new Hinge(),
                new AdaGrad(0.1, 0.5),
                50,
                Trainer.DEFAULT_SEED,
                10,
                0.2D);

        // Wrap the FM classifier in the BaggingTrainer (10 weak learners)
        var baggingTrainer = new org.tribuo.ensemble.BaggingTrainer<>(FMTrainer,
                new org.tribuo.classification.ensemble.VotingCombiner(),
                10,
                Trainer.DEFAULT_SEED);

        // =========================================================
        // MANUAL K-FOLD SETTING: Change this number to 3, 5, 7, or 9
        // =========================================================
        int k = 7;

        System.out.println("Starting Cross Validation for K = " + k + ". This may take a while...");

        // use crossvalidation with the current k
        var crossValidation = new CrossValidation<>(baggingTrainer,
                Data,
                new LabelEvaluator(),
                k);

        // get outputs
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

        // Calculate final percentages
        double finalAcc = (avgAcc / crossValidation.getK()) * 100;
        double finalRecall = (avgRecall / crossValidation.getK()) * 100;
        double finalF1 = (avgF1 / crossValidation.getK()) * 100;
        double finalPrecision = (avgPrecision / crossValidation.getK()) * 100;
        String duration = Util.formatDuration(sTrain, eTrain);

        // 1. Output to Console
        System.out.println("=========================================");
        System.out.println("Results for K = " + k);
        System.out.println("The Training_Testing duration time is : " + duration);
        System.out.println("The average accuracy is : " + finalAcc + "%");
        System.out.println("The average recall is : " + finalRecall + "%");
        System.out.println("The average F1-Score is : " + finalF1 + "%");
        System.out.println("The average precision is : " + finalPrecision + "%");

        // 2. Output to Text File
        // Setting FileWriter to 'true' ensures it appends rather than overwriting
        try (FileWriter fw = new FileWriter("CV_Manual_Results.txt", true);
                PrintWriter pw = new PrintWriter(fw)) {

            pw.println("=========================================");
            pw.println("Results for K = " + k);
            pw.println("Duration: " + duration);
            pw.println("Average Accuracy: " + finalAcc + "%");
            pw.println("Average Recall: " + finalRecall + "%");
            pw.println("Average F1-Score: " + finalF1 + "%");
            pw.println("Average Precision: " + finalPrecision + "%");
            pw.println("=========================================\n");

            System.out.println("\nSUCCESS: Results successfully appended to 'CV_Manual_Results.txt'");

        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }
    }
}