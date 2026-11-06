/**
 * Types and interfaces for XGBoost ML Studio
 */

export interface Dataset {
  name: string;
  filename: string;
  headers: string[];
  rawRows: (string | number)[][];
  rowCount: number;
  colCount: number;
  columnTypes: Record<string, 'numeric' | 'categorical'>;
}

export interface ColumnStat {
  column: string;
  type: 'numeric' | 'categorical';
  count: number;
  missingCount: number;
  missingPct: number;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  q1?: number;
  median?: number;
  q3?: number;
  skewness?: number;
  uniqueValues?: number;
  categoryDistribution?: Record<string, number>;
}

export interface CorrelationMatrix {
  columns: string[];
  matrix: number[][];
}

export interface PreprocessingConfig {
  targetColumn: string;
  featureColumns: string[];
  missingStrategy: 'mean' | 'median' | 'mode' | 'drop';
  scaler: 'standard' | 'minmax' | 'none';
  encoding: 'label' | 'onehot';
  testSize: number; // e.g. 0.2
  randomSeed: number;
  stratify: boolean;
}

export interface ScalerParams {
  type: 'standard' | 'minmax' | 'none';
  means?: number[];
  stds?: number[];
  mins?: number[];
  maxs?: number[];
}

export interface PreprocessedData {
  featureNames: string[];
  targetColumn: string;
  targetClasses: (string | number)[];
  trainX: number[][];
  trainY: number[];
  testX: number[][];
  testY: number[];
  scalerParams: ScalerParams;
  rawStats: {
    totalSamples: number;
    trainSamples: number;
    testSamples: number;
    class0Count: number;
    class1Count: number;
  };
}

export interface XGBoostParams {
  n_estimators: number;
  learning_rate: number;
  max_depth: number;
  subsample: number;
  colsample_bytree: number;
  min_child_weight: number;
  gamma: number;
  reg_lambda: number;
}

export interface TreeNode {
  isLeaf: boolean;
  leafValue?: number;
  splitFeature?: number;
  splitValue?: number;
  splitGain?: number;
  left?: TreeNode;
  right?: TreeNode;
}

export interface ModelEvaluation {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  logLoss: number;
  confusionMatrix: {
    tn: number;
    fp: number;
    fn: number;
    tp: number;
  };
  rocCurve: { fpr: number; tpr: number; threshold: number }[];
  prCurve: { precision: number; recall: number; threshold: number }[];
  featureImportance: { feature: string; score: number; percentage: number }[];
  classReport: {
    class0: { precision: number; recall: number; f1: number; support: number };
    class1: { precision: number; recall: number; f1: number; support: number };
    macroAvg: { precision: number; recall: number; f1: number };
    weightedAvg: { precision: number; recall: number; f1: number };
  };
  probabilities: number[];
  predictions: number[];
}

export interface TrainedModel {
  id: string;
  name: string;
  createdAt: string;
  params: XGBoostParams;
  baseMargin: number;
  trees: TreeNode[];
  preprocessedData: PreprocessedData;
  evaluation: ModelEvaluation;
  trainingDurationMs: number;
  lossHistory: { step: number; trainLoss: number; testLoss: number }[];
}

export interface TrainingHistoryEntry {
  id: string;
  timestamp: string;
  datasetName: string;
  params: XGBoostParams;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  durationMs: number;
}

export type ActiveTab =
  | 'overview'
  | 'descriptive'
  | 'preprocessing'
  | 'training'
  | 'evaluation'
  | 'prediction'
  | 'history'
  | 'code';
