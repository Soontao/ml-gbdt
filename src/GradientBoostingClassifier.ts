/* eslint-disable @typescript-eslint/ban-ts-comment */
/* eslint-disable camelcase */
// @ts-ignore
import { DecisionTreeClassifier } from "ml-cart";

interface GradientBoostingOptions {
  learningRate?: number;
  nEstimators?: number;
  subsample?: number;
  maxDepth?: number;
}

export class GradientBoostingClassifier {
  private learningRate: number;

  private nEstimators: number;

  private subsample: number;

  private maxDepth: number;

  private trees: DecisionTreeClassifier[];

  private featureImportances: Record<number, number>;

  constructor(options: GradientBoostingOptions) {
    this.learningRate = options.learningRate ?? 1;
    this.nEstimators = options.nEstimators ?? 5;
    this.subsample = options.subsample ?? 1;
    this.maxDepth = options.maxDepth ?? 5;
    this.trees = [];
    this.featureImportances = {};
  }

  public train(data: number[][], labels: number[]): void {
    const n = data.length;
    const predictions = new Array(n).fill(0);

    for (let i = 0; i < this.nEstimators; i++) {
      // Compute pseudo-residuals
      const residuals = labels.map((label, j) => label - predictions[j]);

      // Train a decision tree on the residuals with sub-sampling
      const sampleIndices = this.getSubsampleIndices(n);
      const sampledData = sampleIndices.map((index) => data[index]);
      const sampledResiduals = sampleIndices.map((index) => residuals[index]);

      const tree = new DecisionTreeClassifier({ maxDepth: this.maxDepth });
      tree.train(sampledData, sampledResiduals);

      // Update the predictions using the current tree
      for (let j = 0; j < n; j++) {
        const prediction = tree.predict([data[j]])[0];
        predictions[j] += this.learningRate * prediction;
      }

      // Store the tree
      this.trees.push(tree);

      // Update feature importances
      this.updateFeatureImportances(tree.root);
    }
  }

  public predict(features: number[]): number {
    return this.trees.reduce((sum, tree) => sum + this.learningRate * tree.predict([features])[0], 0);
  }

  private getSubsampleIndices(n: number): number[] {
    const subsampleSize = Math.ceil(this.subsample * n);
    return Array.from({ length: n }, (_, i) => i)
      .sort(() => Math.random() - 0.5)
      .slice(0, subsampleSize);
  }

  private updateFeatureImportances(node: any): void {
    // Check if the node is not a leaf (has a splitColumn)
    if (node.splitColumn !== undefined) {
      // Initialize or increment the feature importance count
      this.featureImportances[node.splitColumn] = (this.featureImportances[node.splitColumn] || 0) + 1;
    }

    // Recursively call the function on the left and right children if they exist
    if (node.left) {
      this.updateFeatureImportances(node.left);
    }
    if (node.right) {
      this.updateFeatureImportances(node.right);
    }
  }

  public get feature_importances_(): number[] {
    return Object.values(this.featureImportances);
  }
}
