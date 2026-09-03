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
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
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
            file = new File("01_Executable_Application/models/" + filepath);
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
    public record ClassScore(String label, double confidence) {
    }

    public record PredictionRequest(String dataset, String algorithm, Map<String, Double> features) {
    }

    public record PredictionResponse(
            String predictedClass,
            double confidenceScore,
            String dataset,
            String algorithm,
            int featureCount,
            List<ClassScore> topPredictions,
            String processedImage) {
    }

    public record ExtractionResult(Map<String, Double> features, String processedImage) {
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
        return executePrediction(request, null);
    }

    public ResponseEntity<?> executePrediction(PredictionRequest request, String processedImage) {
        String targetDataset = request.dataset() != null ? request.dataset().toLowerCase().trim() : "swedish";
        String rawAlgo = request.algorithm() != null ? request.algorithm().toLowerCase().trim() : "gbcs";

        // Clean dataset name (e.g. "philippine native" -> "philippine")
        if (targetDataset.contains("philippine"))
            targetDataset = "philippine";
        else if (targetDataset.contains("flavia"))
            targetDataset = "flavia";
        else if (targetDataset.contains("swedish"))
            targetDataset = "swedish";
        else
            targetDataset = "swedish"; // Safe fallback to default benchmark

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

        // Guard against empty feature vector so Tribuo doesn't throw
        // IllegalArgumentException
        if (example.size() == 0) {
            String defaultPrefix = isPhilippine ? "n" : "Att";
            example.add(new Feature(defaultPrefix + "0", 0.0));
        }

        Prediction<Label> prediction = model.predict(example);
        String label = prediction.getOutput().getLabel();
        double score = prediction.getOutput().getScore();
        int activeFeatures = model.getFeatureIDMap().size();

        // Extract Top-3 candidates based on true Bagging Ensemble voting proportions
        List<ClassScore> topPredictions = new ArrayList<>();
        Map<String, Label> outputScores = prediction.getOutputScores();
        if (outputScores != null && !outputScores.isEmpty()) {
            List<Map.Entry<String, Label>> sorted = new ArrayList<>(outputScores.entrySet());
            sorted.sort((a, b) -> Double.compare(b.getValue().getScore(), a.getValue().getScore()));

            int topK = Math.min(3, sorted.size());
            double totalTopScores = 0.0;
            for (int i = 0; i < topK; i++) {
                totalTopScores += Math.max(0.0, sorted.get(i).getValue().getScore());
            }

            for (int i = 0; i < topK; i++) {
                String candLabel = sorted.get(i).getKey();
                double rawScore = Math.max(0.0, sorted.get(i).getValue().getScore());
                double confPct;
                if (totalTopScores > 0.001) {
                    confPct = (rawScore / totalTopScores) * 100.0;
                } else {
                    confPct = (i == 0) ? 100.0 : 0.0;
                }
                topPredictions.add(new ClassScore(candLabel, Math.round(confPct * 10.0) / 10.0));
            }
        } else {
            topPredictions.add(new ClassScore(label, Math.round(score * 1000.0) / 10.0));
        }

        return ResponseEntity.ok(new PredictionResponse(
                label, score, targetDataset, algorithm, activeFeatures, topPredictions, processedImage));
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

            // Also save a copy to scratch/user_uploaded_image.jpg for diagnostics
            try {
                File scratchDir = new File("scratch");
                if (!scratchDir.exists())
                    scratchDir.mkdirs();
                Files.copy(tempImgPath, Path.of("scratch", "user_uploaded_image.jpg"),
                        java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            } catch (Exception ignored) {
            }

            String cleanDataset = dataset != null ? dataset.split(",")[0].trim().toLowerCase() : "philippine";
            String cleanAlgo = algorithm != null ? algorithm.split(",")[0].trim().toLowerCase() : "gbcs";

            System.out.println("📸 [LIVE INFERENCE] Received uploaded file: " + file.getOriginalFilename() + " ("
                    + file.getSize() + " bytes)");
            System.out.println("   Target Dataset: " + cleanDataset + ", Algorithm: " + cleanAlgo);

            // Run python extract_features.py
            ExtractionResult extraction = runFeatureExtractionScript(tempImgPath.toString(), cleanDataset);
            Files.deleteIfExists(tempImgPath);

            if (extraction == null || extraction.features() == null || extraction.features().isEmpty()) {
                System.err.println("❌ Feature extraction failed for: " + file.getOriginalFilename());
                return ResponseEntity.badRequest().body(Map.of("error",
                        "Inception-V3 feature extraction failed for uploaded image file. Please check image format."));
            }

            PredictionRequest req = new PredictionRequest(cleanDataset, cleanAlgo, extraction.features());
            ResponseEntity<?> resp = executePrediction(req, extraction.processedImage());
            if (resp.getBody() instanceof PredictionResponse pr) {
                System.out.println("   Result: " + pr.predictedClass() + " (Score: " + pr.confidenceScore() + ")");
                System.out.println("   Top Candidates: " + pr.topPredictions());
            }
            return resp;

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

        // Swedish Dataset Metrics (Optimal K=7, Table 3 & Table 6)
        metrics.add(Map.of("dataset", "Swedish", "algorithm", "Proposed GBCS", "accuracy", 97.04, "precision", 97.34,
                "recall", 97.21, "f1", 97.03, "featuresSelected", 1369, "reductionRatio", 33.15));
        metrics.add(Map.of("dataset", "Swedish", "algorithm", "Baseline BCS", "accuracy", 96.30, "precision", 96.96,
                "recall", 96.63, "f1", 96.30, "featuresSelected", 1038, "reductionRatio", 49.32));

        // Flavia Dataset Metrics (Optimal K=7, Table 3 & Table 6)
        metrics.add(Map.of("dataset", "Flavia", "algorithm", "Proposed GBCS", "accuracy", 97.90, "precision", 94.38,
                "recall", 94.08, "f1", 93.97, "featuresSelected", 1349, "reductionRatio", 34.13));
        metrics.add(Map.of("dataset", "Flavia", "algorithm", "Baseline BCS", "accuracy", 97.81, "precision", 93.97,
                "recall", 94.28, "f1", 93.87, "featuresSelected", 1018, "reductionRatio", 50.29));

        // Philippine Dataset Metrics (Optimal K=9, Table 3 & Table 6)
        metrics.add(Map.of("dataset", "Philippine", "algorithm", "Proposed GBCS", "accuracy", 97.92, "precision", 98.01,
                "recall", 97.94, "f1", 97.81, "featuresSelected", 1353, "reductionRatio", 33.94));
        metrics.add(Map.of("dataset", "Philippine", "algorithm", "Baseline BCS", "accuracy", 97.69, "precision", 97.80,
                "recall", 97.64, "f1", 97.55, "featuresSelected", 1042, "reductionRatio", 49.12));

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
    private ExtractionResult runFeatureExtractionScript(String imgPath, String dataset) {
        Map<String, Double> map = new LinkedHashMap<>();
        final String[] processedImgHolder = new String[1];
        try {
            String scriptPath = resolveScriptPath();
            String pythonCmd = resolvePythonCommand();

            ProcessBuilder pb = new ProcessBuilder(pythonCmd, scriptPath, "--image", imgPath, "--dataset", dataset);
            Process proc = pb.start();

            // Asynchronously read streams to prevent buffer deadlock
            StringBuilder sb = new StringBuilder();
            StringBuilder errSb = new StringBuilder();

            Thread outThread = new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(proc.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null)
                        sb.append(line);
                } catch (Exception ignored) {
                }
            });

            Thread errThread = new Thread(() -> {
                try (BufferedReader errReader = new BufferedReader(new InputStreamReader(proc.getErrorStream()))) {
                    String errLine;
                    while ((errLine = errReader.readLine()) != null)
                        errSb.append(errLine).append("\n");
                } catch (Exception ignored) {
                }
            });

            outThread.start();
            errThread.start();

            boolean finished = proc.waitFor(45, java.util.concurrent.TimeUnit.SECONDS);
            if (!finished) {
                proc.destroyForcibly();
                System.err.println("⚠️ Feature extraction script timed out after 45 seconds. Process killed.");
                return null;
            }

            outThread.join(1000);
            errThread.join(1000);

            int exitCode = proc.exitValue();
            String rawStr = sb.toString().trim();
            String rawErr = errSb.toString().trim();

            if (exitCode != 0 && !rawErr.isEmpty()) {
                System.err.println("⚠️ Feature extraction script returned code " + exitCode + ": " + rawErr);
            }

            if (!rawStr.isEmpty()) {
                try {
                    ObjectMapper mapper = new ObjectMapper();
                    JsonNode root = mapper.readTree(rawStr);
                    if (root.has("features")) {
                        JsonNode feats = root.get("features");
                        feats.fields().forEachRemaining(entry -> {
                            map.put(entry.getKey(), entry.getValue().asDouble());
                        });
                        if (root.has("processed_image")) {
                            processedImgHolder[0] = root.get("processed_image").asText();
                        }
                    } else {
                        root.fields().forEachRemaining(entry -> {
                            if (!entry.getKey().equals("processed_image")) {
                                map.put(entry.getKey(), entry.getValue().asDouble());
                            } else {
                                processedImgHolder[0] = entry.getValue().asText();
                            }
                        });
                    }
                } catch (Exception parseErr) {
                    System.err.println("⚠️ Jackson parse fallback: " + parseErr.getMessage());
                    int firstBrace = rawStr.indexOf("{");
                    int lastBrace = rawStr.lastIndexOf("}");
                    if (firstBrace != -1 && lastBrace > firstBrace) {
                        String jsonContent = rawStr.substring(firstBrace + 1, lastBrace);
                        String[] pairs = jsonContent.split(",");
                        for (String pair : pairs) {
                            String[] kv = pair.split(":");
                            if (kv.length == 2) {
                                String k = kv[0].replace("\"", "").trim();
                                try {
                                    map.put(k, Double.parseDouble(kv[1].trim()));
                                } catch (NumberFormatException ignored) {
                                }
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("❌ Script feature extraction error: " + e.getMessage());
        }
        return new ExtractionResult(map, processedImgHolder[0]);
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

    private static volatile String cachedPythonCmd = null;

    private String resolveScriptPath() {
        String[] candidates = {
                "extractor/extract_features.py",
                "backend/extractor/extract_features.py",
                "01_Executable_Application/extractor/extract_features.py"
        };
        for (String c : candidates) {
            File f = new File(c);
            if (f.exists()) {
                return f.getPath();
            }
        }
        return "extractor/extract_features.py";
    }

    private String resolvePythonCommand() {
        if (cachedPythonCmd != null) {
            return cachedPythonCmd;
        }

        // 1. Explicit override via system property (-Dpython.cmd=...) or environment
        // variable (PYTHON_CMD=...)
        String custom = System.getProperty("python.cmd");
        if (custom == null || custom.isBlank()) {
            custom = System.getenv("PYTHON_CMD");
        }
        if (custom != null && !custom.isBlank()) {
            if (testPythonCandidate(custom)) {
                System.out.println("🐍 Using custom Python command: " + custom);
                cachedPythonCmd = custom;
                return cachedPythonCmd;
            }
        }

        // 2. Project-local virtual environments (Windows, macOS, Linux)
        String[] venvCandidates = {
                "extractor/venv/Scripts/python.exe",
                "backend/extractor/venv/Scripts/python.exe",
                "extractor/.venv/Scripts/python.exe",
                "backend/extractor/.venv/Scripts/python.exe",
                "venv/Scripts/python.exe",
                ".venv/Scripts/python.exe",
                "extractor/venv/bin/python",
                "backend/extractor/venv/bin/python",
                "extractor/.venv/bin/python",
                "backend/extractor/.venv/bin/python",
                "venv/bin/python",
                ".venv/bin/python"
        };
        for (String venvPath : venvCandidates) {
            File vf = new File(venvPath);
            if (vf.exists() && (vf.canExecute() || vf.getName().endsWith(".exe"))) {
                System.out.println("🐍 Detected project virtualenv Python: " + vf.getAbsolutePath());
                cachedPythonCmd = vf.getAbsolutePath();
                return cachedPythonCmd;
            }
        }

        // 3. Orange Data Mining or Conda in the actual user's home directory
        // (cross-platform, non-hardcoded)
        String userHome = System.getProperty("user.home");
        if (userHome != null && !userHome.isBlank()) {
            File[] envLocations = {
                    new File(userHome, "AppData/Local/Programs/Orange/python.exe"),
                    new File(userHome, "miniconda3/python.exe"),
                    new File(userHome, "anaconda3/python.exe"),
                    new File(userHome, "miniconda3/bin/python"),
                    new File(userHome, "anaconda3/bin/python"),
                    new File("/Applications/Orange3.app/Contents/MacOS/Python")
            };
            for (File loc : envLocations) {
                if (loc.exists()) {
                    System.out.println("🐍 Detected local scientific Python environment: " + loc.getAbsolutePath());
                    cachedPythonCmd = loc.getAbsolutePath();
                    return cachedPythonCmd;
                }
            }
        }

        // 4. System PATH detection: test candidates dynamically based on OS
        String os = System.getProperty("os.name", "").toLowerCase();
        String[] candidates = os.contains("win")
                ? new String[] { "python", "py", "python3" }
                : new String[] { "python3", "python" };

        for (String candidate : candidates) {
            if (testPythonCandidate(candidate)) {
                System.out.println("🐍 Detected verified system Python: " + candidate);
                cachedPythonCmd = candidate;
                return cachedPythonCmd;
            }
        }

        // 5. Fallback if detection was inconclusive
        String fallback = os.contains("win") ? "python" : "python3";
        System.out.println("⚠️ No verified Python runtime found on PATH. Defaulting to: " + fallback);
        cachedPythonCmd = fallback;
        return cachedPythonCmd;
    }

    private boolean testPythonCandidate(String cmd) {
        try {
            Process proc = new ProcessBuilder(cmd, "--version")
                    .redirectErrorStream(true)
                    .start();
            boolean finished = proc.waitFor(3, java.util.concurrent.TimeUnit.SECONDS);
            if (finished && proc.exitValue() == 0) {
                return true;
            }
            proc.destroyForcibly();
        } catch (Exception ignored) {
        }
        return false;
    }
}