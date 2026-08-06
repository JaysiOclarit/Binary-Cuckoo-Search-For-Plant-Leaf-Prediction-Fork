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
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.util.Util;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Automated K-Fold Cross-Validation Batch Runner.
 * <p>
 * Runs K-Fold Cross Validation across multiple datasets and multiple K values
 * (K = 3, 5, 7, 9)
 * in a single execution run, and exports all results side-by-side into a CSV
 * file
 * containing the Dataset Name, Algorithm, K-Folds, Accuracy, Recall, Precision,
 * F1-Score, and Duration.
 * </p>
 */
public class MainClass_All_KFold_CrossValidation {

    public record KFoldResult(
            String datasetName,
            String algorithmName,
            int kFolds,
            double accuracy,
            double recall,
            double precision,
            double f1Score,
            String duration) {
    }

    public record DatasetTask(String datasetName, String algorithmName, String relativePath) {
    }

    public static void main(String[] args) throws IOException {
        System.out.println("=================================================================");
        System.out.println("   AUTOMATED BATCH K-FOLD CROSS-VALIDATION RUNNER                ");
        System.out.println("=================================================================\n");

        // 1. Define list of K values to evaluate
        int[] kValues = { 3, 5, 7, 9 };

        // 2. Define list of dataset tasks to evaluate (ONLY Feature-Selected Datasets)
        List<DatasetTask> tasks = new ArrayList<>();

        // Proposed GBCS Feature-Selected Datasets (Subfolders & Root)
        addIfFileExists(tasks, "Swedish", "Proposed_GBCS",
                Paths.get("Entire Data Folder", "Swedish After GBCS-FS", "Swedish After GBCS-FS.csv").toString());
        addIfFileExists(tasks, "Flavia", "Proposed_GBCS",
                Paths.get("Entire Data Folder", "Flavia After GBCS-FS", "Flavia After GBCS-FS.csv").toString());
        addIfFileExists(tasks, "Philippine", "Proposed_GBCS",
                Paths.get("Entire Data Folder", "Philippine After GBCS-FS", "Philippine After GBCS-FS.csv").toString());
        addIfFileExists(tasks, "Swedish", "Proposed_GBCS", Paths
                .get("backend", "Entire Data Folder", "Swedish After GBCS-FS", "Swedish After GBCS-FS.csv").toString());
        addIfFileExists(tasks, "Flavia", "Proposed_GBCS", Paths
                .get("backend", "Entire Data Folder", "Flavia After GBCS-FS", "Flavia After GBCS-FS.csv").toString());
        addIfFileExists(tasks, "Philippine", "Proposed_GBCS",
                Paths.get("backend", "Entire Data Folder", "Philippine After GBCS-FS", "Philippine After GBCS-FS.csv")
                        .toString());
        addIfFileExists(tasks, "Swedish", "Proposed_GBCS", "Swedish After GBCS-FS.csv");
        addIfFileExists(tasks, "Flavia", "Proposed_GBCS", "Flavia After GBCS-FS.csv");
        addIfFileExists(tasks, "Philippine", "Proposed_GBCS", "Philippine After GBCS-FS.csv");

        // Baseline BCS Feature-Selected Datasets (Subfolders & Root)
        addIfFileExists(tasks, "Swedish", "Baseline_BCS",
                Paths.get("Entire Data Folder", "Swedish After FS", "Swedish After FS.csv").toString());
        addIfFileExists(tasks, "Flavia", "Baseline_BCS",
                Paths.get("Entire Data Folder", "Flavia After FS", "Flavia After FS.csv").toString());
        addIfFileExists(tasks, "Philippine", "Baseline_BCS",
                Paths.get("Entire Data Folder", "Philippine After FS", "Philippine After FS.csv").toString());
        addIfFileExists(tasks, "Swedish", "Baseline_BCS",
                Paths.get("backend", "Entire Data Folder", "Swedish After FS", "Swedish After FS.csv").toString());
        addIfFileExists(tasks, "Flavia", "Baseline_BCS",
                Paths.get("backend", "Entire Data Folder", "Flavia After FS", "Flavia After FS.csv").toString());
        addIfFileExists(tasks, "Philippine", "Baseline_BCS", Paths
                .get("backend", "Entire Data Folder", "Philippine After FS", "Philippine After FS.csv").toString());
        addIfFileExists(tasks, "Swedish", "Baseline_BCS", "Swedish After FS.csv");
        addIfFileExists(tasks, "Flavia", "Baseline_BCS", "Flavia After FS.csv");
        addIfFileExists(tasks, "Philippine", "Baseline_BCS", "Philippine After FS.csv");

        // Legacy Subfolder Fallbacks
        addIfFileExists(tasks, "Swedish", "Baseline_BCS",
                Paths.get("Entire Data Folder", "Swedish After IBCS-FS", "Swedish train data After FS.csv").toString());
        addIfFileExists(tasks, "Swedish", "Baseline_BCS",
                Paths.get("Entire Data Folder", "Swedish After IBCS-FS", "Swedish After FS(2).csv").toString());
        addIfFileExists(tasks, "Swedish", "Baseline_BCS",
                Paths.get("Entire Data Folder", "Swedish After IBCS-FS", "Swedish After FS.csv").toString());
        addIfFileExists(tasks, "Flavia", "Baseline_BCS",
                Paths.get("Entire Data Folder", "Flavia After IBCS-FS", "Flavia train data After FS.csv").toString());
        addIfFileExists(tasks, "Flavia", "Baseline_BCS",
                Paths.get("Entire Data Folder", "Flavia After IBCS-FS", "Flavia After FS(2).csv").toString());
        addIfFileExists(tasks, "Flavia", "Baseline_BCS",
                Paths.get("Entire Data Folder", "Flavia After IBCS-FS", "Flavia After FS.csv").toString());
        addIfFileExists(tasks, "Philippine", "Baseline_BCS",
                Paths.get("Entire Data Folder", "Philippine After IBCS-FS", "Philippine After FS(2).csv").toString());
        addIfFileExists(tasks, "Philippine", "Baseline_BCS",
                Paths.get("Entire Data Folder", "Philippine After IBCS-FS", "Philippine train data After FS.csv")
                        .toString());
        addIfFileExists(tasks, "Philippine", "Baseline_BCS",
                Paths.get("Entire Data Folder", "Philippine After IBCS-FS", "Philippine After FS.csv").toString());

        if (tasks.isEmpty()) {
            System.err.println("ERROR: No valid dataset files found to process.");
            return;
        }

        List<KFoldResult> allResults = new ArrayList<>();

        // 3. Configure FM + Bagging Trainer (10 weak learners)
        var fmTrainer = new FMClassificationTrainer(
                new Hinge(),
                new AdaGrad(0.1, 0.5),
                50,
                Trainer.DEFAULT_SEED,
                10,
                0.2D);

        var baggingTrainer = new BaggingTrainer<>(
                fmTrainer,
                new VotingCombiner(),
                10,
                Trainer.DEFAULT_SEED);

        // 4. Run Cross-Validation for each dataset and each K value
        for (DatasetTask task : tasks) {
            System.out.println("-----------------------------------------------------------------");
            System.out.println("Processing Dataset: " + task.datasetName() + " (" + task.algorithmName() + ")");
            System.out.println("File: " + task.relativePath());

            var dataSource = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(task.relativePath()),
                    "Class");
            var dataset = new MutableDataset<>(dataSource);

            for (int k : kValues) {
                System.out.printf("  -> Running K = %d Cross-Validation...\n", k);
                var sTrain = System.currentTimeMillis();

                var crossValidation = new CrossValidation<>(baggingTrainer, dataset, new LabelEvaluator(), k);

                double sumAcc = 0.0;
                double sumRecall = 0.0;
                double sumPrecision = 0.0;
                double sumF1 = 0.0;

                for (var performance : crossValidation.evaluate()) {
                    sumAcc += performance.getA().accuracy();
                    sumRecall += performance.getA().macroAveragedRecall();
                    sumPrecision += performance.getA().macroAveragedPrecision();
                    sumF1 += performance.getA().macroAveragedF1();
                }

                var eTrain = System.currentTimeMillis();

                double avgAcc = (sumAcc / k) * 100.0;
                double avgRecall = (sumRecall / k) * 100.0;
                double avgPrecision = (sumPrecision / k) * 100.0;
                double avgF1 = (sumF1 / k) * 100.0;
                String duration = Util.formatDuration(sTrain, eTrain);

                KFoldResult result = new KFoldResult(
                        task.datasetName(),
                        task.algorithmName(),
                        k,
                        avgAcc,
                        avgRecall,
                        avgPrecision,
                        avgF1,
                        duration);

                allResults.add(result);

                System.out.printf(
                        "     Accuracy: %.2f%% | Recall: %.2f%% | Precision: %.2f%% | F1: %.2f%% | Duration: %s\n",
                        avgAcc, avgRecall, avgPrecision, avgF1, duration);
            }
        }

        // 5. Print Final Batch K-Fold Cross-Validation Summary Table
        System.out
                .println("\n=========================================================================================");
        System.out.println("   FINAL BATCH K-FOLD CROSS-VALIDATION SUMMARY TABLE");
        System.out.println("=========================================================================================");
        System.out.printf("%-12s | %-15s | %-7s | %-10s | %-10s | %-10s | %-10s | %-15s\n",
                "Dataset", "Algorithm", "K-Folds", "Acc (%)", "Recall (%)", "Prec (%)", "F1 (%)", "Duration");
        System.out.println("-----------------------------------------------------------------------------------------");
        for (KFoldResult r : allResults) {
            System.out.printf("%-12s | %-15s | K=%-5d | %-10.2f | %-10.2f | %-10.2f | %-10.2f | %-15s\n",
                    r.datasetName(), r.algorithmName(), r.kFolds(), r.accuracy(), r.recall(), r.precision(),
                    r.f1Score(), r.duration());
        }
        System.out.println("=========================================================================================");

        // 6. Save all results to CSV File
        // 6. Export results to CSV file in Results directory
        File resultsDir = new File("Results");
        if (!resultsDir.exists()) resultsDir = new File("../Results");
        if (!resultsDir.exists()) resultsDir.mkdirs();

        File csvFile = new File(resultsDir, "All_KFold_CrossValidation_Results.csv");
        File textLogFile = new File(resultsDir, "CV_Manual_Results.txt");

        exportToCSV(csvFile.getPath(), allResults);

        // 7. Append to CV_Manual_Results.txt text log
        exportToTextLog(textLogFile.getPath(), allResults);

        System.out.println("\n=================================================================");
        System.out.println("SUCCESS: All K-Fold Cross-Validation evaluations completed!");
        System.out.println("Results exported to CSV file: " + csvFile.getPath());
        System.out.println("Results appended to text log: " + textLogFile.getPath());
        System.out.println("=================================================================");
    }

    private static void addIfFileExists(List<DatasetTask> tasks, String datasetName, String algorithmName,
            String filePath) {
        if (filePath.toLowerCase().contains("train data") || filePath.toLowerCase().contains("test data")) {
            return; // Exclude partial train/test dataset files
        }
        File file = new File(filePath);
        if (file.exists() && !file.isDirectory()) {
            // Avoid adding duplicates for the same dataset and algorithm
            for (DatasetTask task : tasks) {
                if (task.datasetName().equalsIgnoreCase(datasetName)
                        && task.algorithmName().equalsIgnoreCase(algorithmName)) {
                    return;
                }
            }
            tasks.add(new DatasetTask(datasetName, algorithmName, filePath));
        }
    }

    private static void exportToCSV(String csvFileName, List<KFoldResult> results) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(csvFileName))) {
            pw.println(
                    "DatasetName,Algorithm,K_Folds,AccuracyPercent,RecallPercent,PrecisionPercent,F1ScorePercent,Duration");
            for (KFoldResult r : results) {
                pw.printf("%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%s\n",
                        r.datasetName(),
                        r.algorithmName(),
                        r.kFolds(),
                        r.accuracy(),
                        r.recall(),
                        r.precision(),
                        r.f1Score(),
                        r.duration());
            }
        }
    }

    private static void exportToTextLog(String logFileName, List<KFoldResult> results) throws IOException {
        try (FileWriter fw = new FileWriter(logFileName, true);
                PrintWriter pw = new PrintWriter(fw)) {

            pw.println("\n=========================================================");
            pw.println("AUTOMATED BATCH K-FOLD CROSS-VALIDATION RESULTS SUMMARY");
            pw.println("=========================================================");
            for (KFoldResult r : results) {
                pw.println("Dataset: " + r.datasetName() + " | Algorithm: " + r.algorithmName());
                pw.println("Results for K = " + r.kFolds());
                pw.println("Duration: " + r.duration());
                pw.printf("Average Accuracy: %.4f%%\n", r.accuracy());
                pw.printf("Average Recall: %.4f%%\n", r.recall());
                pw.printf("Average F1-Score: %.4f%%\n", r.f1Score());
                pw.printf("Average Precision: %.4f%%\n", r.precision());
                pw.println("---------------------------------------------------------");
            }
        }
    }
}
