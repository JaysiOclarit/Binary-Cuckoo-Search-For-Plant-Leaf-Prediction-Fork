package WrapperCuckooSearchForFS.org.Main;

import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.sgd.fm.FMClassificationTrainer;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.math.optimisers.AdaGrad;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;

public class ModelExporter {

    public static void main(String[] args) {
        System.out.println("=================================================================");
        System.out.println("         TRIBUO MODEL EXPORTER & SERIALIZER                      ");
        System.out.println("=================================================================\n");

        File modelsDir = getModelsDir();
        System.out.println("Target Model Directory: " + modelsDir.getAbsolutePath() + "\n");

        Map<String, String> modelMap = new LinkedHashMap<>();

        // Baseline (BCS)
        modelMap.put("Swedish_BCS_Model.ser", getDatasetPath("Swedish After FS", "Swedish After FS.csv"));
        modelMap.put("Flavia_BCS_Model.ser", getDatasetPath("Flavia After FS", "Flavia After FS.csv"));
        modelMap.put("Philippine_BCS_Model.ser", getDatasetPath("Philippine After FS", "Philippine After FS.csv"));

        // Proposed (GBCS)
        modelMap.put("Swedish_GBCS_Model.ser", getDatasetPath("Swedish After GBCS-FS", "Swedish After GBCS-FS.csv"));
        modelMap.put("Flavia_GBCS_Model.ser", getDatasetPath("Flavia After GBCS-FS", "Flavia After GBCS-FS.csv"));
        modelMap.put("Philippine_GBCS_Model.ser", getDatasetPath("Philippine After GBCS-FS", "Philippine After GBCS-FS.csv"));

        // Full Raw Models
        modelMap.put("Swedish_Plant_Model.ser", getDatasetPath("Original Dataset", "Swedish Leaf data.csv"));
        modelMap.put("Flavia_Plant_Model.ser", getDatasetPath("Original Dataset", "Flavia Leaf data.csv"));
        modelMap.put("Philippine_Plant_Model.ser", getDatasetPath("Original Dataset", "Philippine Leaf data.csv"));

        // Setup Trainer: FM + Bagging ensemble (Aligned with Paper: 50 Epochs)
        var fmtTrainer = new FMClassificationTrainer(
                new Hinge(),
                new AdaGrad(0.1, 0.5),
                50, // 50 epochs (aligned with Baseline & Proposed papers)
                Trainer.DEFAULT_SEED,
                10,
                0.2D
        );

        var baggingTrainer = new BaggingTrainer<>(
                fmtTrainer,
                new org.tribuo.classification.ensemble.VotingCombiner(),
                10,
                Trainer.DEFAULT_SEED
        );

        for (Map.Entry<String, String> entry : modelMap.entrySet()) {
            String modelFilename = entry.getKey();
            String csvPath = entry.getValue();

            File file = new File(csvPath);
            if (!file.exists()) {
                System.err.println("❌ CSV dataset not found for " + modelFilename + ": " + csvPath);
                continue;
            }

            File targetModelFile = new File(modelsDir, modelFilename);

            System.out.println("⏳ Training model for " + modelFilename + " from " + csvPath + "...");
            try {
                var dataSource = new CSVLoader<>(new LabelFactory()).loadDataSource(file.toPath(), "Class");
                var dataset = new MutableDataset<>(dataSource);

                Model<Label> model = baggingTrainer.train(dataset);

                try (ObjectOutputStream ois = new ObjectOutputStream(new FileOutputStream(targetModelFile))) {
                    ois.writeObject(model);
                }

                System.out.println("✅ Saved serialized model: " + targetModelFile.getPath() + " (" + model.getFeatureIDMap().size() + " features)");

            } catch (Exception e) {
                System.err.println("❌ Failed training " + modelFilename + ": " + e.getMessage());
                e.printStackTrace();
            }
        }

        System.out.println("\n🎉 All models exported successfully to " + modelsDir.getPath() + "!");
    }

    public static File getModelsDir() {
        File m1 = new File("models");
        if (m1.exists()) return m1;

        File m2 = new File("backend", "models");
        if (m2.exists()) return m2;

        if (m1.mkdirs()) return m1;
        if (m2.mkdirs()) return m2;

        return m1;
    }

    private static String getDatasetPath(String subfolder, String filename) {
        Path p1 = Paths.get("Entire Data Folder", subfolder, filename);
        if (p1.toFile().exists()) return p1.toString();

        Path p2 = Paths.get("backend", "Entire Data Folder", subfolder, filename);
        if (p2.toFile().exists()) return p2.toString();

        return p1.toString();
    }
}
