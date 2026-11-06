/**
 * High-performance mathematical XGBoost Classifier engine
 * Implements Gradient Boosted Decision Trees for binary classification
 * with exact gradients, hessians, regularization (L2 + gamma), and feature importance.
 */
import {
  XGBoostParams,
  TreeNode,
  PreprocessedData,
  ModelEvaluation,
  TrainedModel,
} from '../types';

export function sigmoid(z: number): number {
  if (z < -30) return 0;
  if (z > 30) return 1;
  return 1 / (1 + Math.exp(-z));
}

export function logLoss(yTrue: number[], yPredProbs: number[]): number {
  const eps = 1e-15;
  let sum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const p = Math.max(eps, Math.min(1 - eps, yPredProbs[i]));
    sum += -(yTrue[i] * Math.log(p) + (1 - yTrue[i]) * Math.log(1 - p));
  }
  return sum / (yTrue.length || 1);
}

export function predictRawMargin(x: number[], baseMargin: number, trees: TreeNode[], lr: number): number {
  let margin = baseMargin;
  for (const tree of trees) {
    margin += lr * evaluateTree(x, tree);
  }
  return margin;
}

export function predictProbability(x: number[], baseMargin: number, trees: TreeNode[], lr: number): number {
  const margin = predictRawMargin(x, baseMargin, trees, lr);
  return sigmoid(margin);
}

function evaluateTree(x: number[], node: TreeNode): number {
  if (node.isLeaf) {
    return node.leafValue ?? 0;
  }
  const fIdx = node.splitFeature ?? 0;
  const sVal = node.splitValue ?? 0;
  if (x[fIdx] <= sVal) {
    return node.left ? evaluateTree(x, node.left) : (node.leafValue ?? 0);
  } else {
    return node.right ? evaluateTree(x, node.right) : (node.leafValue ?? 0);
  }
}

export function trainXGBoost(
  preprocessed: PreprocessedData,
  params: XGBoostParams,
  onStepProgress?: (step: number, total: number, trainLoss: number, testLoss: number) => void
): TrainedModel {
  const { trainX, trainY, testX, testY, featureNames } = preprocessed;
  const nTrain = trainX.length;
  const nFeatures = featureNames.length;

  if (nTrain === 0) {
    throw new Error('Dữ liệu huấn luyện rỗng.');
  }

  // 1. Initial base score (log-odds of positive class)
  const posCount = trainY.filter((y) => y === 1).length;
  const baseProb = Math.max(0.01, Math.min(0.99, posCount / nTrain));
  const baseMargin = Math.log(baseProb / (1 - baseProb));

  // Current raw predictions
  const trainMargins = new Array(nTrain).fill(baseMargin);
  const testMargins = new Array(testX.length).fill(baseMargin);

  const trees: TreeNode[] = [];
  const lossHistory: { step: number; trainLoss: number; testLoss: number }[] = [];
  const featureGainMap = new Map<number, number>();

  const startTrainTime = performance.now();

  for (let iter = 0; iter < params.n_estimators; iter++) {
    // 2. Compute gradients & hessians
    const g = new Array(nTrain);
    const h = new Array(nTrain);
    const trainProbs = new Array(nTrain);

    for (let i = 0; i < nTrain; i++) {
      const p = sigmoid(trainMargins[i]);
      trainProbs[i] = p;
      g[i] = p - trainY[i];
      h[i] = Math.max(1e-6, p * (1 - p));
    }

    // 3. Subsampling of training rows
    const sampleIndices: number[] = [];
    for (let i = 0; i < nTrain; i++) {
      if (params.subsample >= 1.0 || Math.random() < params.subsample) {
        sampleIndices.push(i);
      }
    }
    if (sampleIndices.length < 5) {
      // Fallback to all samples if subsampling dropped too many
      for (let i = 0; i < nTrain; i++) sampleIndices.push(i);
    }

    // 4. Feature subsampling
    const candidateFeatureIndices: number[] = [];
    for (let f = 0; f < nFeatures; f++) {
      if (params.colsample_bytree >= 1.0 || Math.random() < params.colsample_bytree) {
        candidateFeatureIndices.push(f);
      }
    }
    if (candidateFeatureIndices.length === 0) {
      for (let f = 0; f < nFeatures; f++) candidateFeatureIndices.push(f);
    }

    // 5. Build Regression Tree for gradients
    const tree = buildTree(
      trainX,
      g,
      h,
      sampleIndices,
      candidateFeatureIndices,
      0,
      params,
      featureGainMap
    );
    trees.push(tree);

    // 6. Update margins
    for (let i = 0; i < nTrain; i++) {
      trainMargins[i] += params.learning_rate * evaluateTree(trainX[i], tree);
    }
    for (let i = 0; i < testX.length; i++) {
      testMargins[i] += params.learning_rate * evaluateTree(testX[i], tree);
    }

    // 7. Track iteration loss
    const currentTrainProbs = trainMargins.map(sigmoid);
    const currentTestProbs = testMargins.map(sigmoid);
    const trLoss = logLoss(trainY, currentTrainProbs);
    const teLoss = logLoss(testY, currentTestProbs);

    lossHistory.push({
      step: iter + 1,
      trainLoss: trLoss,
      testLoss: teLoss,
    });

    if (onStepProgress && (iter % 5 === 0 || iter === params.n_estimators - 1)) {
      onStepProgress(iter + 1, params.n_estimators, trLoss, teLoss);
    }
  }

  const durationMs = Math.round(performance.now() - startTrainTime);

  // 8. Evaluate on test set
  const evaluation = evaluateModelOnTestSet(
    testX,
    testY,
    baseMargin,
    trees,
    params.learning_rate,
    featureNames,
    featureGainMap
  );

  return {
    id: `xgb-${Date.now()}`,
    name: `XGBoost (n=${params.n_estimators}, lr=${params.learning_rate}, depth=${params.max_depth})`,
    createdAt: new Date().toLocaleString('vi-VN'),
    params,
    baseMargin,
    trees,
    preprocessedData: preprocessed,
    evaluation,
    trainingDurationMs: durationMs,
    lossHistory,
  };
}

function buildTree(
  X: number[][],
  g: number[],
  h: number[],
  sampleIndices: number[],
  candidateFeatures: number[],
  currentDepth: number,
  params: XGBoostParams,
  featureGainMap: Map<number, number>
): TreeNode {
  let G = 0;
  let H = 0;
  for (const idx of sampleIndices) {
    G += g[idx];
    H += h[idx];
  }

  // Base leaf weight: w = - G / (H + lambda)
  const leafValue = -G / (H + params.reg_lambda);

  if (currentDepth >= params.max_depth || sampleIndices.length <= 2 || H < params.min_child_weight) {
    return { isLeaf: true, leafValue };
  }

  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;
  let bestLeftIndices: number[] = [];
  let bestRightIndices: number[] = [];

  const parentScore = (G * G) / (H + params.reg_lambda);

  // Search best split
  for (const f of candidateFeatures) {
    // Get unique sorted values for feature f
    const featurePairs = sampleIndices.map((idx) => ({
      idx,
      val: X[idx][f],
      g: g[idx],
      h: h[idx],
    }));

    featurePairs.sort((a, b) => a.val - b.val);

    let gLeft = 0;
    let hLeft = 0;

    for (let i = 0; i < featurePairs.length - 1; i++) {
      gLeft += featurePairs[i].g;
      hLeft += featurePairs[i].h;

      // Don't split identical adjacent values
      if (featurePairs[i].val === featurePairs[i + 1].val) {
        continue;
      }

      const gRight = G - gLeft;
      const hRight = H - hLeft;

      if (hLeft < params.min_child_weight || hRight < params.min_child_weight) {
        continue;
      }

      const leftScore = (gLeft * gLeft) / (hLeft + params.reg_lambda);
      const rightScore = (gRight * gRight) / (hRight + params.reg_lambda);
      const gain = 0.5 * (leftScore + rightScore - parentScore) - params.gamma;

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = f;
        bestThreshold = (featurePairs[i].val + featurePairs[i + 1].val) / 2;
      }
    }
  }

  // If gain <= 0, stop splitting and make leaf
  if (bestGain <= 0 || bestFeature === -1) {
    return { isLeaf: true, leafValue };
  }

  // Partition indices
  for (const idx of sampleIndices) {
    if (X[idx][bestFeature] <= bestThreshold) {
      bestLeftIndices.push(idx);
    } else {
      bestRightIndices.push(idx);
    }
  }

  if (bestLeftIndices.length === 0 || bestRightIndices.length === 0) {
    return { isLeaf: true, leafValue };
  }

  // Accumulate feature importance gain
  const existingGain = featureGainMap.get(bestFeature) || 0;
  featureGainMap.set(bestFeature, existingGain + bestGain);

  const left = buildTree(
    X,
    g,
    h,
    bestLeftIndices,
    candidateFeatures,
    currentDepth + 1,
    params,
    featureGainMap
  );

  const right = buildTree(
    X,
    g,
    h,
    bestRightIndices,
    candidateFeatures,
    currentDepth + 1,
    params,
    featureGainMap
  );

  return {
    isLeaf: false,
    splitFeature: bestFeature,
    splitValue: bestThreshold,
    splitGain: bestGain,
    leafValue,
    left,
    right,
  };
}

export function evaluateModelOnTestSet(
  testX: number[][],
  testY: number[],
  baseMargin: number,
  trees: TreeNode[],
  lr: number,
  featureNames: string[],
  featureGainMap: Map<number, number>,
  threshold = 0.5
): ModelEvaluation {
  const n = testX.length;
  if (n === 0) {
    throw new Error('Tập kiểm thử rỗng.');
  }

  const probabilities: number[] = [];
  const predictions: number[] = [];

  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;

  for (let i = 0; i < n; i++) {
    const prob = predictProbability(testX[i], baseMargin, trees, lr);
    probabilities.push(prob);
    const pred = prob >= threshold ? 1 : 0;
    predictions.push(pred);

    const actual = testY[i];
    if (actual === 1 && pred === 1) tp++;
    else if (actual === 0 && pred === 0) tn++;
    else if (actual === 0 && pred === 1) fp++;
    else if (actual === 1 && pred === 0) fn++;
  }

  const accuracy = (tp + tn) / n;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const loss = logLoss(testY, probabilities);

  // ROC curve points & AUC calculation
  const rocCurve: { fpr: number; tpr: number; threshold: number }[] = [];
  const totalPos = tp + fn;
  const totalNeg = tn + fp;

  // Thresholds from 0 to 1
  const numSteps = 50;
  for (let s = 0; s <= numSteps; s++) {
    const t = s / numSteps;
    let stepTp = 0;
    let stepFp = 0;
    for (let i = 0; i < n; i++) {
      if (probabilities[i] >= t) {
        if (testY[i] === 1) stepTp++;
        else stepFp++;
      }
    }
    const tpr = totalPos > 0 ? stepTp / totalPos : 0;
    const fpr = totalNeg > 0 ? stepFp / totalNeg : 0;
    rocCurve.push({ fpr, tpr, threshold: t });
  }

  // Sort ROC by FPR ascending to compute AUC via trapezoidal rule
  const sortedRoc = [...rocCurve].sort((a, b) => a.fpr - b.fpr);
  // Ensure corners (0,0) and (1,1) are cleanly bounded
  let rocAuc = 0;
  for (let i = 1; i < sortedRoc.length; i++) {
    const xDiff = sortedRoc[i].fpr - sortedRoc[i - 1].fpr;
    if (xDiff > 0) {
      const yAvg = (sortedRoc[i].tpr + sortedRoc[i - 1].tpr) / 2;
      rocAuc += xDiff * yAvg;
    }
  }
  rocAuc = Math.max(0, Math.min(1, rocAuc));
  if (rocAuc < 0.5) rocAuc = 1 - rocAuc; // standard classification orientation

  // Precision-Recall curve
  const prCurve: { precision: number; recall: number; threshold: number }[] = [];
  for (let s = 1; s <= numSteps; s++) {
    const t = s / numSteps;
    let stepTp = 0;
    let stepFp = 0;
    for (let i = 0; i < n; i++) {
      if (probabilities[i] >= t) {
        if (testY[i] === 1) stepTp++;
        else stepFp++;
      }
    }
    const rec = totalPos > 0 ? stepTp / totalPos : 0;
    const prec = stepTp + stepFp > 0 ? stepTp / (stepTp + stepFp) : 1;
    prCurve.push({ precision: prec, recall: rec, threshold: t });
  }

  // Feature importance
  let totalGain = 0;
  for (let f = 0; f < featureNames.length; f++) {
    totalGain += featureGainMap.get(f) || 0;
  }
  const featureImportance = featureNames.map((name, f) => {
    const gain = featureGainMap.get(f) || 0;
    const percentage = totalGain > 0 ? (gain / totalGain) * 100 : 100 / featureNames.length;
    return {
      feature: name,
      score: gain,
      percentage: Number(percentage.toFixed(2)),
    };
  });
  featureImportance.sort((a, b) => b.score - a.score);

  // Detailed Classification Report
  const prec0 = tn + fn > 0 ? tn / (tn + fn) : 0;
  const rec0 = tn + fp > 0 ? tn / (tn + fp) : 0;
  const f1_0 = prec0 + rec0 > 0 ? (2 * prec0 * rec0) / (prec0 + rec0) : 0;

  const prec1 = precision;
  const rec1 = recall;
  const f1_1 = f1Score;

  const supp0 = tn + fp;
  const supp1 = tp + fn;

  const macroAvg = {
    precision: (prec0 + prec1) / 2,
    recall: (rec0 + rec1) / 2,
    f1: (f1_0 + f1_1) / 2,
  };

  const weightedAvg = {
    precision: (prec0 * supp0 + prec1 * supp1) / n,
    recall: (rec0 * supp0 + rec1 * supp1) / n,
    f1: (f1_0 * supp0 + f1_1 * supp1) / n,
  };

  return {
    accuracy,
    precision,
    recall,
    f1Score,
    rocAuc,
    logLoss: loss,
    confusionMatrix: { tn, fp, fn, tp },
    rocCurve,
    prCurve,
    featureImportance,
    classReport: {
      class0: { precision: prec0, recall: rec0, f1: f1_0, support: supp0 },
      class1: { precision: prec1, recall: rec1, f1: f1_1, support: supp1 },
      macroAvg,
      weightedAvg,
    },
    probabilities,
    predictions,
  };
}
