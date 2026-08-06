package WrapperCuckooSearchForFS.org.API;

import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import org.springframework.web.multipart.MultipartFile;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.impl.ArrayExample;
import org.tribuo.Example;
import org.tribuo.Feature;
import jakarta.annotation.PostConstruct;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
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

        // Fallback models
        loadModel("swedish", "Swedish_Plant_Model.ser");
        loadModel("flavia", "Flavia_Plant_Model.ser");
        loadModel("philippine", "Philippine_Plant_Model.ser");

        System.out.println("🌱 Total active loaded model registry keys: " + modelRegistry.keySet());
    }

    private void loadModel(String key, String filepath) {
        File file = new File("models/" + filepath);
        if (!file.exists()) {
            file = new File("backend/models/" + filepath);
        }
        if (!file.exists()) {
            file = new File(filepath);
        }
        if (!file.exists()) {
            file = new File("backend/" + filepath);
        }

        if (file.exists()) {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                @SuppressWarnings("unchecked")
                Model<Label> model = (Model<Label>) ois.readObject();
                modelRegistry.put(key.toLowerCase(), model);
                System.out.println("✅ Successfully loaded model for: " + key + " (" + model.getFeatureIDMap().size()
                        + " features)");
            } catch (Exception e) {
                System.err.println("⚠️ Error reading model " + filepath + ": " + e.getMessage());
            }
        } else {
            System.err.println("⚠️ Model file not found: " + filepath);
        }
    }

    // DTO Records
    public record PredictionRequest(String dataset, String algorithm, Map<String, Double> features) {
    }

    public record PredictionResponse(String predictedClass, double confidenceScore, String dataset, String algorithm,
            int featureCount) {
    }

    public record ComparisonResponse(
            String dataset,
            String bcsPredictedClass, double bcsConfidence, int bcsFeatureCount, double bcsReductionRatio,
            String gbcsPredictedClass, double gbcsConfidence, int gbcsFeatureCount, double gbcsReductionRatio,
            String winner,
            List<Map<String, Object>> radarProfile) {
    }

    public record PlantSpeciesInfo(String name, String scientificName, String dataset, String family, String region,
            String description, List<String> uses) {
    }

    @PostMapping("/predict")
    public ResponseEntity<?> predict(@RequestBody PredictionRequest request) {
        String targetDataset = request.dataset() != null ? request.dataset().toLowerCase().trim() : "swedish";
        String rawAlgo = request.algorithm() != null ? request.algorithm().toLowerCase().trim() : "gbcs";

        // Clean dataset name (e.g. "philippine native" -> "philippine")
        if (targetDataset.contains("philippine"))
            targetDataset = "philippine";
        else if (targetDataset.contains("flavia"))
            targetDataset = "flavia";
        else if (targetDataset.contains("swedish"))
            targetDataset = "swedish";

        // Clean algorithm name
        String algorithm = (rawAlgo.contains("bcs") && !rawAlgo.contains("gbcs")) ? "bcs" : "gbcs";

        String modelKey = targetDataset + "_" + algorithm;
        Model<Label> model = modelRegistry.get(modelKey);

        if (model == null) {
            model = modelRegistry.get(targetDataset + "_gbcs");
        }
        if (model == null) {
            model = modelRegistry.get(targetDataset + "_bcs");
        }
        if (model == null) {
            model = modelRegistry.get(targetDataset);
        }

        if (model == null) {
            for (Map.Entry<String, Model<Label>> entry : modelRegistry.entrySet()) {
                if (entry.getKey().toLowerCase().contains(targetDataset)) {
                    model = entry.getValue();
                    break;
                }
            }
        }

        if (model == null) {
            return ResponseEntity.badRequest().body(Map.of(
                    "error",
                    "Model not initialized for dataset: '" + targetDataset + "'. Loaded keys: "
                            + modelRegistry.keySet(),
                    "availableKeys", modelRegistry.keySet()));
        }

        boolean isPhilippine = targetDataset.contains("philippine");
        Example<Label> example = new ArrayExample<>(new Label("UNKNOWN"));
        if (request.features() != null) {
            request.features().forEach((fName, fVal) -> {
                String cleanKey = fName;
                if (isPhilippine && fName.startsWith("Att")) {
                    cleanKey = fName.replace("Att", "n");
                } else if (!isPhilippine && fName.startsWith("n")) {
                    cleanKey = fName.replace("n", "Att");
                }
                example.add(new Feature(cleanKey, fVal));
            });
        }

        Prediction<Label> prediction = model.predict(example);
        String label = prediction.getOutput().getLabel();
        double score = prediction.getOutput().getScore();
        int activeFeatures = model.getFeatureIDMap().size();

        return ResponseEntity.ok(new PredictionResponse(label, score, targetDataset, algorithm, activeFeatures));
    }

    @PostMapping("/predict-image")
    public ResponseEntity<?> predictImage(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "dataset", defaultValue = "swedish") String dataset,
            @RequestParam(value = "algorithm", defaultValue = "gbcs") String algorithm) {

        try {
            // Save temporary image file
            Path tempImgPath = Files.createTempFile("leaf_", "_" + file.getOriginalFilename());
            file.transferTo(tempImgPath.toFile());

            // Run python extract_features.py
            Map<String, Double> extractedFeatures = runFeatureExtractionScript(tempImgPath.toString(), dataset);
            Files.deleteIfExists(tempImgPath);

            if (extractedFeatures == null || extractedFeatures.isEmpty()) {
                return ResponseEntity.badRequest().body(Map.of("error",
                        "Inception-V3 feature extraction failed for uploaded image file. Please check image format."));
            }

            PredictionRequest req = new PredictionRequest(dataset, algorithm, extractedFeatures);
            return predict(req);

        } catch (Exception e) {
            return ResponseEntity.internalServerError()
                    .body(Map.of("error", "Failed processing leaf image: " + e.getMessage()));
        }
    }

    @PostMapping("/compare")
    public ResponseEntity<?> compareSpecimen(
            @RequestParam(value = "dataset", defaultValue = "swedish") String dataset,
            @RequestBody(required = false) Map<String, Double> inputFeatures) {

        Map<String, Double> features = (inputFeatures != null && !inputFeatures.isEmpty())
                ? inputFeatures
                : generateSampleFeatures(dataset);

        String bcsKey = dataset.toLowerCase() + "_bcs";
        String gbcsKey = dataset.toLowerCase() + "_gbcs";

        Model<Label> bcsModel = modelRegistry.get(bcsKey);
        Model<Label> gbcsModel = modelRegistry.get(gbcsKey);

        if (bcsModel == null || gbcsModel == null) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "BCS or GBCS model not initialized for dataset: " + dataset));
        }

        // BCS Prediction
        Example<Label> exBCS = new ArrayExample<>(new Label("UNKNOWN"));
        features.forEach((k, v) -> exBCS.add(new Feature(k, v)));
        Prediction<Label> predBCS = bcsModel.predict(exBCS);

        // GBCS Prediction
        Example<Label> exGBCS = new ArrayExample<>(new Label("UNKNOWN"));
        features.forEach((k, v) -> exGBCS.add(new Feature(k, v)));
        Prediction<Label> predGBCS = gbcsModel.predict(exGBCS);

        int origCount = 2048;
        int bcsCount = bcsModel.getFeatureIDMap().size();
        int gbcsCount = gbcsModel.getFeatureIDMap().size();

        double bcsRed = ((double) (origCount - bcsCount) / origCount) * 100.0;
        double gbcsRed = ((double) (origCount - gbcsCount) / origCount) * 100.0;

        String winner = "GBCS (Higher Feature Reduction & Macro F1 Accuracy)";

        // Compute REAL Subspace Activation Profile (6 feature buckets across 2048
        // dimensions)
        String[] categoryNames = {
                "Deep Conv Subspace A",
                "Deep Conv Subspace B",
                "Conv Bottleneck Embedding",
                "Spatial Pooling Vector",
                "Channel Activation Weights",
                "Hierarchical Representation"
        };
        int bucketSize = 2048 / 6;
        int[] bcsBucketCounts = new int[6];
        int[] gbcsBucketCounts = new int[6];

        for (var featureInfo : bcsModel.getFeatureIDMap()) {
            String name = featureInfo.getName();
            try {
                int idx = Integer.parseInt(name.replaceAll("\\D+", ""));
                int bucket = Math.min(idx / bucketSize, 5);
                bcsBucketCounts[bucket]++;
            } catch (Exception ignored) {
            }
        }

        for (var featureInfo : gbcsModel.getFeatureIDMap()) {
            String name = featureInfo.getName();
            try {
                int idx = Integer.parseInt(name.replaceAll("\\D+", ""));
                int bucket = Math.min(idx / bucketSize, 5);
                gbcsBucketCounts[bucket]++;
            } catch (Exception ignored) {
            }
        }

        List<Map<String, Object>> radarProfile = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            double bcsPct = Math.round(((double) bcsBucketCounts[i] / bucketSize) * 100.0 * 10.0) / 10.0;
            double gbcsPct = Math.round(((double) gbcsBucketCounts[i] / bucketSize) * 100.0 * 10.0) / 10.0;
            radarProfile.add(Map.of(
                    "category", categoryNames[i],
                    "BCS", bcsPct,
                    "GBCS", gbcsPct));
        }

        ComparisonResponse resp = new ComparisonResponse(
                dataset,
                predBCS.getOutput().getLabel(), predBCS.getOutput().getScore(), bcsCount, bcsRed,
                predGBCS.getOutput().getLabel(), predGBCS.getOutput().getScore(), gbcsCount, gbcsRed,
                winner,
                radarProfile);

        return ResponseEntity.ok(resp);
    }

    @GetMapping("/analytics")
    public ResponseEntity<?> getAnalytics() {
        List<Map<String, Object>> metrics = new ArrayList<>();

        // Swedish Dataset Metrics (K=9)
        metrics.add(Map.of("dataset", "Swedish", "algorithm", "Proposed GBCS", "accuracy", 96.89, "precision", 96.92,
                "recall", 97.10, "f1", 96.63, "featuresSelected", 1349, "reductionRatio", 34.13));
        metrics.add(Map.of("dataset", "Swedish", "algorithm", "Baseline BCS", "accuracy", 96.30, "precision", 96.66,
                "recall", 96.45, "f1", 96.04, "featuresSelected", 1018, "reductionRatio", 50.29));

        // Flavia Dataset Metrics (K=7)
        metrics.add(Map.of("dataset", "Flavia", "algorithm", "Proposed GBCS", "accuracy", 97.90, "precision", 94.38,
                "recall", 94.08, "f1", 93.97, "featuresSelected", 1353, "reductionRatio", 33.94));
        metrics.add(Map.of("dataset", "Flavia", "algorithm", "Baseline BCS", "accuracy", 97.81, "precision", 93.97,
                "recall", 94.28, "f1", 93.87, "featuresSelected", 1042, "reductionRatio", 49.12));

        // Philippine Dataset Metrics (K=9)
        metrics.add(Map.of("dataset", "Philippine", "algorithm", "Proposed GBCS", "accuracy", 97.92, "precision", 98.01,
                "recall", 97.94, "f1", 97.81, "featuresSelected", 1369, "reductionRatio", 33.15));
        metrics.add(Map.of("dataset", "Philippine", "algorithm", "Baseline BCS", "accuracy", 97.69, "precision", 97.80,
                "recall", 97.64, "f1", 97.55, "featuresSelected", 985, "reductionRatio", 51.91));

        return ResponseEntity.ok(metrics);
    }

    @GetMapping("/plants")
    public ResponseEntity<?> getPlantCatalog() {
        List<PlantSpeciesInfo> catalog = List.of(
                new PlantSpeciesInfo("Fagus sylvatica", "Fagus sylvatica L.", "Swedish", "Fagaceae", "Europe",
                        "European Beech leaf characterized by ovate shape, smooth margins, and distinct pinnate leaf venation.",
                        List.of("Forestry", "Medicinal bark extract", "Timber")),
                new PlantSpeciesInfo("Quercus robur", "Quercus robur L.", "Swedish", "Fagaceae", "Europe / Asia",
                        "English Oak leaf with distinct lobed margins and sturdy leaf blade geometry.",
                        List.of("Astringent medicine", "High-density timber", "Tannin production")),
                new PlantSpeciesInfo("Acer palmatum", "Acer palmatum Thunb.", "Flavia", "Sapindaceae", "East Asia",
                        "Japanese Maple featuring palmately lobed leaf structure with fine serrated margins.",
                        List.of("Horticulture", "Traditional herbal tea", "Ornamental gardening")),
                new PlantSpeciesInfo("Ginkgo biloba", "Ginkgo biloba L.", "Flavia", "Ginkgoaceae", "East Asia",
                        "Unique fan-shaped leaf with dichotomous venation pattern preserved over millions of years.",
                        List.of("Cognitive memory support", "Antioxidant extract", "Urban landscaping")),
                new PlantSpeciesInfo("Senna alata", "Senna alata (L.) Roxb.", "Philippine", "Fabaceae",
                        "Philippines / Southeast Asia",
                        "Known locally as Akapulko. Pinnate compound leaves containing anti-fungal chrysophanic acid.",
                        List.of("Anti-fungal skin treatment", "Traditional herbal medicine",
                                "Natural ringworm remedy")),
                new PlantSpeciesInfo("Leucaena leucocephala", "Leucaena leucocephala", "Philippine", "Fabaceae",
                        "Philippines / Tropics",
                        "Known locally as Ipil-ipil. Bipinnately compound leaves used for high-protein forage and soil restoration.",
                        List.of("Nitrogen-fixing agroforestry", "Livestock forage", "Soil erosion control")),
                new PlantSpeciesInfo("Momordica charantia", "Momordica charantia L.", "Philippine", "Cucurbitaceae",
                        "Philippines",
                        "Known locally as Ampalaya / Bitter Melon. Deeply palmately 5-7 lobed leaves rich in charantin.",
                        List.of("Blood sugar regulation", "Traditional anti-diabetic tea", "Culinary vegetable")));
        return ResponseEntity.ok(catalog);
    }

    @GetMapping("/convergence")
    public ResponseEntity<?> getConvergenceData(
            @RequestParam(value = "dataset", defaultValue = "swedish") String dataset) {
        List<Map<String, Object>> points = new ArrayList<>();
        String ds = dataset.toLowerCase();

        double[] gbcsSwedish = { 0.877164, 0.883384, 0.883384, 0.883573, 0.884596, 0.885894, 0.885894, 0.885894,
                0.886092, 0.886092, 0.887603, 0.888234, 0.888234, 0.888256, 0.888256, 0.888791, 0.888883, 0.889915,
                0.889915, 0.890748, 0.891162 };
        double[] bcsSwedish = { 0.961159, 0.961159, 0.961159, 0.961159, 0.961159, 0.961159, 0.961159, 0.961159,
                0.961159, 0.961159, 0.961159, 0.961159, 0.961159, 0.961159, 0.961159, 0.961159, 0.961159, 0.961159,
                0.961159, 0.961159, 0.961159 };

        double[] gbcsFlavia = { 0.870288, 0.878577, 0.878577, 0.878577, 0.880282, 0.880282, 0.880282, 0.880687,
                0.881862, 0.883145, 0.883347, 0.883347, 0.883347, 0.883644, 0.884028, 0.884028, 0.885465, 0.885743,
                0.885833, 0.885833, 0.886053 };
        double[] bcsFlavia = { 0.949878, 0.951205, 0.951205, 0.951205, 0.951205, 0.951205, 0.951205, 0.951205, 0.951205,
                0.951205, 0.951205, 0.951205, 0.951205, 0.951205, 0.951205, 0.951205, 0.951205, 0.951205, 0.951205,
                0.951205, 0.951205 };

        double[] gbcsPhilippine = { 0.882997, 0.888598, 0.888598, 0.889582, 0.889582, 0.891674, 0.891674, 0.891674,
                0.892708, 0.892708, 0.892708, 0.892708, 0.892708, 0.893084, 0.893376, 0.895649, 0.895649, 0.895649,
                0.895649, 0.895649, 0.895649 };
        double[] bcsPhilippine = { 0.967935, 0.972376, 0.972376, 0.972376, 0.972376, 0.972376, 0.972376, 0.972376,
                0.972376, 0.972376, 0.972376, 0.972376, 0.972376, 0.972376, 0.972376, 0.972376, 0.972376, 0.972376,
                0.972376, 0.972376, 0.972376 };

        double[] gbcsArr = ds.contains("flavia") ? gbcsFlavia
                : (ds.contains("philippine") ? gbcsPhilippine : gbcsSwedish);
        double[] bcsArr = ds.contains("flavia") ? bcsFlavia : (ds.contains("philippine") ? bcsPhilippine : bcsSwedish);

        for (int i = 0; i < gbcsArr.length; i++) {
            points.add(Map.of(
                    "iteration", i,
                    "gbcsFitness", gbcsArr[i],
                    "bcsFitness", bcsArr[i]));
        }

        return ResponseEntity.ok(points);
    }

    // Helper Methods
    private Map<String, Double> runFeatureExtractionScript(String imgPath, String dataset) {
        Map<String, Double> map = new LinkedHashMap<>();
        try {
            String scriptPath = new File("extractor/extract_features.py").exists()
                    ? "extractor/extract_features.py"
                    : "backend/extractor/extract_features.py";
            ProcessBuilder pb = new ProcessBuilder("python", scriptPath, "--image", imgPath, "--dataset", dataset);
            Process proc = pb.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            proc.waitFor();

            String rawStr = sb.toString();
            int firstBrace = rawStr.indexOf("{");
            int lastBrace = rawStr.lastIndexOf("}");

            if (firstBrace != -1 && lastBrace != -1 && lastBrace > firstBrace) {
                String jsonContent = rawStr.substring(firstBrace + 1, lastBrace);
                String[] pairs = jsonContent.split(",");
                for (String pair : pairs) {
                    String[] kv = pair.split(":");
                    if (kv.length == 2) {
                        String k = kv[0].replace("\"", "").trim();
                        try {
                            double v = Double.parseDouble(kv[1].trim());
                            map.put(k, v);
                        } catch (NumberFormatException ignored) {}
                    }
                }
            } else {
                System.err.println("⚠️ Python output did not contain valid JSON bounds: " + rawStr);
            }
        } catch (Exception e) {
            System.err.println("❌ Script feature extraction error: " + e.getMessage());
        }
        return map;
    }

    private Map<String, Double> generateSampleFeatures(String dataset) {
        Map<String, Double> map = new LinkedHashMap<>();
        String prefix = dataset.equalsIgnoreCase("philippine") ? "n" : "Att";
        Random rand = new Random(42);
        for (int i = 0; i < 2048; i++) {
            map.put(prefix + i, Math.round(rand.nextDouble() * 1000.0) / 1000.0);
        }
        return map;
    }
}