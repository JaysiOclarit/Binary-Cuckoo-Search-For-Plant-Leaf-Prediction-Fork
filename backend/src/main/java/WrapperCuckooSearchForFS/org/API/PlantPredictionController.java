package WrapperCuckooSearchForFS.org.API;

import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.impl.ArrayExample;
import org.tribuo.Example;
import org.tribuo.Feature;
import jakarta.annotation.PostConstruct;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*") // Allows your frontend to connect without CORS errors
public class PlantPredictionController {

    private final Map<String, Model<Label>> modelRegistry = new HashMap<>();

    @PostConstruct
    public void loadModels() {
        // Load Baseline (BCS) models
        loadModel("swedish_bcs", "Swedish_BCS_Model.ser");
        loadModel("flavia_bcs", "Flavia_BCS_Model.ser");
        loadModel("philippine_bcs", "Philippine_BCS_Model.ser");

        // Load Proposed (GBCS) models
        loadModel("swedish_gbcs", "Swedish_GBCS_Model.ser");
        loadModel("flavia_gbcs", "Flavia_GBCS_Model.ser");
        loadModel("philippine_gbcs", "Philippine_GBCS_Model.ser");

        // Backwards compatibility fallbacks
        loadModel("swedish", "Swedish_Plant_Model.ser");
        loadModel("flavia", "Flavia_Plant_Model.ser");
        loadModel("philippine", "Philippine_Plant_Model.ser");
    }

    private void loadModel(String key, String filepath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filepath))) {
            @SuppressWarnings("unchecked")
            Model<Label> model = (Model<Label>) ois.readObject();
            modelRegistry.put(key.toLowerCase(), model);
            System.out.println("✅ Successfully loaded model for: " + key);

            System.out.println("Expected features for " + key + ":");
            model.getFeatureIDMap().forEach(feature ->
                    System.out.println("  - " + feature.getName())
            );

        } catch (Exception e) {
            System.err.println("⚠️ Could not load model " + filepath + " for key [" + key + "]. Ensure model file exists.");
        }
    }

    // Data Transfer Objects for JSON mapping
    public record PredictionRequest(String dataset, String algorithm, Map<String, Double> features) {}
    public record PredictionResponse(String predictedClass, double confidenceScore, String dataset, String algorithm) {}

    @PostMapping("/predict")
    public ResponseEntity<?> predict(@RequestBody PredictionRequest request) {
        String targetDataset = request.dataset() != null ? request.dataset().toLowerCase() : "";
        String algorithm = request.algorithm() != null ? request.algorithm().toLowerCase() : "gbcs";

        // Try combined key first (e.g. "swedish_gbcs"), then raw dataset key
        String modelKey = targetDataset + "_" + algorithm;
        Model<Label> model = modelRegistry.get(modelKey);
        if (model == null) {
            model = modelRegistry.get(targetDataset);
        }

        if (model == null) {
            return ResponseEntity.badRequest().body("Model not found or not trained yet for key: " + modelKey + " (Available keys: " + modelRegistry.keySet() + ")");
        }

        // 1. Convert incoming JSON features to Tribuo format
        Example<Label> example = new ArrayExample<>(new Label("UNKNOWN"));
        if (request.features() != null) {
            request.features().forEach((featureName, featureValue) -> {
                example.add(new Feature(featureName, featureValue));
            });
        }

        // 2. Run the model prediction
        Prediction<Label> prediction = model.predict(example);

        // 3. Extract and return results
        String label = prediction.getOutput().getLabel();
        double score = prediction.getOutput().getScore();

        return ResponseEntity.ok(new PredictionResponse(label, score, targetDataset, algorithm));
    }
}