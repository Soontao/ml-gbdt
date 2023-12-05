import * as irisDataSet from "ml-dataset-iris";
import { GradientBoostingClassifier } from "../src/GradientBoostingClassifier";

describe("GradientBoostingClassifier", () => {
  it("should train and predict correctly", () => {
    const params = {
      learningRate: 0.1,
      nEstimators: 3,
      subsample: 0.8,
      maxDepth: 2,
    };

    const gradientBoosting = new GradientBoostingClassifier(params);

    // Mock training data
    const trainData = [
      [1, 2],
      [2, 3],
      [3, 4],
    ];

    // Mock labels
    const labels = [0, 1, 0];

    gradientBoosting.train(trainData, labels);

    // Mock test features
    const testFeatures = [4, 5];
    const prediction = gradientBoosting.predict(testFeatures);

    // Perform assertions based on your expectations
    expect(prediction).toEqual(0);

    // You may add more assertions based on your requirements
  });

  it("should calculate feature importances correctly", () => {
    const params = {
      nEstimators: 10,
      maxDepth: 5,
    };

    const trainingSet = irisDataSet.getNumbers();
    const predictions = irisDataSet.getClasses().map((elem) => irisDataSet.getDistinctClasses().indexOf(elem));

    const gradientBoosting = new GradientBoostingClassifier(params);

    gradientBoosting.train(trainingSet, predictions);

    // Access feature importances
    const featureImportances = gradientBoosting.feature_importances_;

    // Perform assertions based on your expectations
    expect(featureImportances).toHaveLength(4);

    // You may add more assertions based on your requirements
  });
});
