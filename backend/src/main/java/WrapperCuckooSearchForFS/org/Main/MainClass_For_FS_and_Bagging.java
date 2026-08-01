package WrapperCuckooSearchForFS.org.Main;

import WrapperCuckooSearchForFS.org.Discreeting.TransferFunction;
import WrapperCuckooSearchForFS.org.Optimizers.CuckooSearchOptimizer;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.common.nearest.KNNModel;
import org.tribuo.common.nearest.KNNTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.data.csv.CSVSaver;
import org.tribuo.dataset.SelectedFeatureDataset;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.math.distance.L2Distance;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.util.Util;

import java.io.IOException;
import java.nio.file.Paths;

public class MainClass_For_FS_and_Bagging {
        public static void main(String[] args) throws IOException {
                // read the data
                var dataPath = Paths.get("Entire Data Folder", "Original Dataset", "Swedish Leaf data.csv").toString();
                var dataSource = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(dataPath), "Class");

                var dataSplitting = new TrainTestSplitter<>(dataSource, 0.6, Trainer.DEFAULT_SEED);
                var trainData = new MutableDataset<>(dataSplitting.getTrain());
                var testData = new MutableDataset<>(dataSplitting.getTest());

                // use the feature selection optimizer based on the given learner
                var learner = new KNNTrainer<>(1,
                                new L2Distance(),
                                Runtime.getRuntime().availableProcessors(),
                                new VotingCombiner(),
                                KNNModel.Backend.THREADPOOL,
                                NeighboursQueryFactoryType.BRUTE_FORCE);

                // use IBCS for FS
                var optimizer = new CuckooSearchOptimizer(learner,
                                TransferFunction.V2,
                                30,
                                2d,
                                2d,
                                0.1d,
                                1.5d,
                                20,
                                12345);

                /*
                 * // use mRMR filter-based FS
                 * var sDate = System.currentTimeMillis();
                 * var SFS = new mRMR(500,
                 * 10,
                 * Runtime.getRuntime().availableProcessors())
                 * .select(trainPart);
                 * var eDate = System.currentTimeMillis();
                 * var SFDS = new SelectedFeatureDataset<>(trainPart, SFS);
                 */

                var sDate = System.currentTimeMillis();
                var SFS = optimizer.select(trainData);
                var eDate = System.currentTimeMillis();
                var SFDS = new SelectedFeatureDataset<>(trainData, SFS);

                // Export convergence history to CSV for plotting & documentation
                String csvPath = "BCS_Convergence_History.csv";
                optimizer.exportConvergenceCSV(csvPath);

                // Save the selected subset of features
                new CSVSaver().save(Paths.get(System.getProperty("user.dir") + "\\Swedish After FS.csv"),
                                SFDS,
                                "Class");

                double reductionRatio = (1.0 - ((double) SFDS.size() / trainData.getFeatureMap().size())) * 100.0;
                int optIter = optimizer.getOptimalConvergenceIteration();

                System.out.println("=================================================");
                System.out.println("BCS Feature Selection Analysis & Results:");
                System.out.printf("FS Duration: %s\n", Util.formatDuration(sDate, eDate));
                System.out.printf("Convergence Speed (Optimal Iteration): Iteration %d (out of 20)\n", optIter);
                System.out.printf("Original Feature Count: %d\n", trainData.getFeatureMap().size());
                System.out.printf("Selected Feature Subset Size: %d\n", SFDS.size());
                System.out.printf("Feature Reduction Ratio (FRR): %.2f%%\n", reductionRatio);
                System.out.println("Convergence History Exported To: " + csvPath);
                System.out.println("=================================================");

                /*
                 * Here you store the data after feature selection (FS) for the training part
                 * only.
                 * To get the values or columns of the features in the generated training set
                 * after FS,
                 * you should use the TableSaw library or another library to drop the unwanted
                 * features
                 * and keep the ones that match the training part.
                 */
        }
}