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
        // Loads the models into memory when the server starts
        // Ensure these .ser files are placed in the root directory of your project (same level as pom.xml)
        loadModel("swedish", "Swedish_Plant_Model.ser");
        loadModel("flavia", "Flavia_Plant_Model.ser");
        loadModel("philippine", "Philippine_Plant_Model.ser");
    }

    private void loadModel(String key, String filepath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filepath))) {
            @SuppressWarnings("unchecked")
            Model<Label> model = (Model<Label>) ois.readObject();
            modelRegistry.put(key, model);
            System.out.println("✅ Successfully loaded model for: " + key);

            // --- ADD THESE LINES TO REVEAL EXPECTED FEATURES ---
            System.out.println("Expected features for " + key + ":");
            model.getFeatureIDMap().forEach(feature ->
                    System.out.println("  - " + feature.getName())
            );
            // ---------------------------------------------------

        } catch (Exception e) {
            System.err.println("⚠️ Could not load model " + filepath + ". Have you run the Bagging script to save it yet?");
        }
    }

    // Data Transfer Objects for JSON mapping
    public record PredictionRequest(String dataset, Map<String, Double> features) {}
    public record PredictionResponse(String predictedClass, double confidenceScore, String dataset) {}

    @PostMapping("/predict")
    public ResponseEntity<?> predict(@RequestBody PredictionRequest request) {
        String targetDataset = request.dataset().toLowerCase();
        Model<Label> model = modelRegistry.get(targetDataset);

        if (model == null) {
            return ResponseEntity.badRequest().body("Model not found or not trained yet for dataset: " + targetDataset);
        }

        // 1. Convert incoming JSON features to Tribuo format
        Example<Label> example = new ArrayExample<>(new Label("UNKNOWN"));

        // --- FIXED CODE ---
        request.features().forEach((featureName, featureValue) -> {
            example.add(new Feature(featureName, featureValue));
        });
        // ------------------

        // 2. Run the model prediction
        Prediction<Label> prediction = model.predict(example);

        // 3. Extract and return results
        String label = prediction.getOutput().getLabel();
        double score = prediction.getOutput().getScore();

        return ResponseEntity.ok(new PredictionResponse(label, score, targetDataset));
    }
}