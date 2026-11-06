import test from 'node:test';
import assert from 'node:assert/strict';
import { sigmoid, logLoss, trainXGBoost, predictProbability } from './xgboostEngine';
import { PreprocessedData, XGBoostParams } from '../types';

test('sigmoid function calculates standard logistic probabilities', () => {
  assert.equal(sigmoid(0), 0.5);
  assert.ok(sigmoid(10) > 0.99);
  assert.ok(sigmoid(-10) < 0.01);
});

test('logLoss calculates cross-entropy loss correctly', () => {
  const yTrue = [1, 0, 1, 0];
  const perfectProbs = [0.999, 0.001, 0.999, 0.001];
  const badProbs = [0.001, 0.999, 0.001, 0.999];

  const lowLoss = logLoss(yTrue, perfectProbs);
  const highLoss = logLoss(yTrue, badProbs);

  assert.ok(lowLoss < 0.05);
  assert.ok(highLoss > 3.0);
});

test('XGBoost engine trains on linear separable data with high accuracy', () => {
  // Generate simple 2D dataset: class 1 if x1 + x2 > 1 else 0
  const trainX: number[][] = [];
  const trainY: number[] = [];
  const testX: number[][] = [];
  const testY: number[] = [];

  for (let i = 0; i < 100; i++) {
    const x1 = (i % 10) / 10;
    const x2 = Math.floor(i / 10) / 10;
    const label = x1 + x2 > 0.9 ? 1 : 0;
    if (i % 5 === 0) {
      testX.push([x1, x2]);
      testY.push(label);
    } else {
      trainX.push([x1, x2]);
      trainY.push(label);
    }
  }

  const preprocessed: PreprocessedData = {
    featureNames: ['x1', 'x2'],
    targetColumn: 'label',
    targetClasses: [0, 1],
    trainX,
    trainY,
    testX,
    testY,
    scalerParams: { type: 'none' },
    rawStats: {
      totalSamples: 100,
      trainSamples: 80,
      testSamples: 20,
      class0Count: 50,
      class1Count: 50,
    },
  };

  const params: XGBoostParams = {
    n_estimators: 25,
    learning_rate: 0.15,
    max_depth: 3,
    subsample: 1.0,
    colsample_bytree: 1.0,
    min_child_weight: 1,
    gamma: 0.0,
    reg_lambda: 1.0,
  };

  const model = trainXGBoost(preprocessed, params);

  assert.ok(model.trees.length === 25);
  assert.ok(model.evaluation.accuracy >= 0.85);
  assert.ok(model.evaluation.rocAuc >= 0.85);

  // Check single prediction
  const highProb = predictProbability([0.9, 0.9], model.baseMargin, model.trees, params.learning_rate);
  const lowProb = predictProbability([0.1, 0.1], model.baseMargin, model.trees, params.learning_rate);
  assert.ok(highProb > lowProb);
});
