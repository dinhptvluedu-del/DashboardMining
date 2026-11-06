/**
 * Mathematical statistics & data preprocessing pipeline
 */
import {
  Dataset,
  ColumnStat,
  CorrelationMatrix,
  PreprocessingConfig,
  PreprocessedData,
} from '../types';

export function computeColumnStats(dataset: Dataset): ColumnStat[] {
  const stats: ColumnStat[] = [];

  for (let c = 0; c < dataset.headers.length; c++) {
    const colName = dataset.headers[c];
    const colType = dataset.columnTypes[colName] || 'numeric';
    const rawVals = dataset.rawRows.map((r) => r[c]);
    const totalCount = rawVals.length;

    let missingCount = 0;
    const validNumericVals: number[] = [];
    const catFreq: Record<string, number> = {};

    for (const v of rawVals) {
      if (v === '' || v === null || v === undefined) {
        missingCount++;
      } else if (typeof v === 'number' && !isNaN(v)) {
        validNumericVals.push(v);
        catFreq[String(v)] = (catFreq[String(v)] || 0) + 1;
      } else {
        const strVal = String(v).trim();
        if (strVal === '') {
          missingCount++;
        } else {
          catFreq[strVal] = (catFreq[strVal] || 0) + 1;
        }
      }
    }

    const missingPct = totalCount > 0 ? (missingCount / totalCount) * 100 : 0;

    if (colType === 'numeric' && validNumericVals.length > 0) {
      validNumericVals.sort((a, b) => a - b);
      const n = validNumericVals.length;
      const sum = validNumericVals.reduce((acc, v) => acc + v, 0);
      const mean = sum / n;

      const variance =
        n > 1
          ? validNumericVals.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (n - 1)
          : 0;
      const std = Math.sqrt(variance);

      const min = validNumericVals[0];
      const max = validNumericVals[n - 1];
      const median = getPercentile(validNumericVals, 0.5);
      const q1 = getPercentile(validNumericVals, 0.25);
      const q3 = getPercentile(validNumericVals, 0.75);

      // Sample skewness
      let skewness = 0;
      if (n > 2 && std > 0) {
        const m3 = validNumericVals.reduce((acc, v) => acc + ((v - mean) / std) ** 3, 0);
        skewness = (n / ((n - 1) * (n - 2))) * m3;
      }

      stats.push({
        column: colName,
        type: 'numeric',
        count: n,
        missingCount,
        missingPct,
        mean,
        std,
        min,
        max,
        q1,
        median,
        q3,
        skewness,
        uniqueValues: Object.keys(catFreq).length,
      });
    } else {
      stats.push({
        column: colName,
        type: 'categorical',
        count: totalCount - missingCount,
        missingCount,
        missingPct,
        uniqueValues: Object.keys(catFreq).length,
        categoryDistribution: catFreq,
      });
    }
  }

  return stats;
}

function getPercentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const pos = (sorted.length - 1) * p;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (base + 1 < sorted.length) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

export function computeCorrelationMatrix(dataset: Dataset, numericOnly = true): CorrelationMatrix {
  const numericColumns = dataset.headers.filter((h) =>
    numericOnly ? dataset.columnTypes[h] === 'numeric' : true
  );

  // Extract clean numerical arrays
  const colVectors: Record<string, number[]> = {};
  for (const col of numericColumns) {
    const colIdx = dataset.headers.indexOf(col);
    colVectors[col] = [];
    for (const row of dataset.rawRows) {
      const v = row[colIdx];
      colVectors[col].push(typeof v === 'number' ? v : Number(v) || 0);
    }
  }

  const matrix: number[][] = [];
  for (let i = 0; i < numericColumns.length; i++) {
    const rowCorr: number[] = [];
    const colA = numericColumns[i];
    const vecA = colVectors[colA];
    const meanA = vecA.reduce((s, v) => s + v, 0) / (vecA.length || 1);

    for (let j = 0; j < numericColumns.length; j++) {
      if (i === j) {
        rowCorr.push(1.0);
        continue;
      }
      const colB = numericColumns[j];
      const vecB = colVectors[colB];
      const meanB = vecB.reduce((s, v) => s + v, 0) / (vecB.length || 1);

      let cov = 0;
      let varA = 0;
      let varB = 0;
      for (let k = 0; k < vecA.length; k++) {
        const da = vecA[k] - meanA;
        const db = vecB[k] - meanB;
        cov += da * db;
        varA += da * da;
        varB += db * db;
      }

      const denom = Math.sqrt(varA * varB);
      const corr = denom === 0 ? 0 : Math.max(-1, Math.min(1, cov / denom));
      rowCorr.push(corr);
    }
    matrix.push(rowCorr);
  }

  return {
    columns: numericColumns,
    matrix,
  };
}

export function preprocessDataset(
  dataset: Dataset,
  config: PreprocessingConfig
): PreprocessedData {
  const targetIdx = dataset.headers.indexOf(config.targetColumn);
  if (targetIdx === -1) {
    throw new Error(`Không tìm thấy cột mục tiêu: ${config.targetColumn}`);
  }

  const featureIndices = config.featureColumns.map((col) => {
    const idx = dataset.headers.indexOf(col);
    if (idx === -1) throw new Error(`Không tìm thấy cột đặc trưng: ${col}`);
    return idx;
  });

  if (featureIndices.length === 0) {
    throw new Error('Vui lòng chọn ít nhất một biến đầu vào (Feature column).');
  }

  // 1. Filter rows & Handle Missing Values
  type IntermediateRow = { features: (number | string)[]; target: number | string };
  const validRows: IntermediateRow[] = [];

  for (const rawRow of dataset.rawRows) {
    const rawTarget = rawRow[targetIdx];
    if (rawTarget === '' || rawTarget === null || rawTarget === undefined) {
      if (config.missingStrategy === 'drop') continue;
    }

    const featureVals: (number | string)[] = [];
    let hasMissingInFeatures = false;
    for (const fIdx of featureIndices) {
      const v = rawRow[fIdx];
      if (v === '' || v === null || v === undefined) {
        hasMissingInFeatures = true;
      }
      featureVals.push(v);
    }

    if (hasMissingInFeatures && config.missingStrategy === 'drop') {
      continue;
    }

    validRows.push({
      features: featureVals,
      target: rawTarget,
    });
  }

  if (validRows.length < 10) {
    throw new Error('Số lượng dữ liệu sau khi lọc quá ít (dưới 10 mẫu).');
  }

  // 2. Impute missing values if not drop
  if (config.missingStrategy !== 'drop') {
    for (let f = 0; f < config.featureColumns.length; f++) {
      const colName = config.featureColumns[f];
      const isNum = dataset.columnTypes[colName] === 'numeric';

      if (isNum) {
        const nums = validRows
          .map((r) => r.features[f])
          .filter((v): v is number => typeof v === 'number' && !isNaN(v));

        let fillVal = 0;
        if (nums.length > 0) {
          if (config.missingStrategy === 'median') {
            nums.sort((a, b) => a - b);
            fillVal = getPercentile(nums, 0.5);
          } else {
            // Mean
            fillVal = nums.reduce((s, v) => s + v, 0) / nums.length;
          }
        }

        for (const row of validRows) {
          const v = row.features[f];
          if (v === '' || v === null || v === undefined || isNaN(Number(v))) {
            row.features[f] = fillVal;
          } else {
            row.features[f] = Number(v);
          }
        }
      } else {
        // Mode for categorical
        const freqs: Record<string, number> = {};
        for (const row of validRows) {
          const v = String(row.features[f] || '').trim();
          if (v) freqs[v] = (freqs[v] || 0) + 1;
        }
        let modeVal = '';
        let maxF = 0;
        for (const [k, count] of Object.entries(freqs)) {
          if (count > maxF) {
            maxF = count;
            modeVal = k;
          }
        }
        for (const row of validRows) {
          const v = String(row.features[f] || '').trim();
          if (!v) row.features[f] = modeVal;
        }
      }
    }
  }

  // 3. Process Target into binary 0/1 classes
  const targetClassesSet = new Set<string | number>();
  for (const row of validRows) {
    targetClassesSet.add(row.target);
  }
  const targetClasses = Array.from(targetClassesSet).sort();
  if (targetClasses.length < 2) {
    throw new Error('Biến mục tiêu chỉ có 1 lớp giá trị duy nhất, không thể huấn luyện phân loại.');
  }

  // Map to 0 and 1 (treat first as 0, second as 1, or preserve 0 and 1 if numeric)
  let class0 = targetClasses[0];
  let class1 = targetClasses[1];
  if (targetClasses.includes(0) && targetClasses.includes(1)) {
    class0 = 0;
    class1 = 1;
  }

  const binaryY: number[] = validRows.map((r) => {
    return r.target === class1 || String(r.target) === '1' ? 1 : 0;
  });

  // 4. Feature Encoding
  let expandedFeatureNames: string[] = [];
  const processedXRows: number[][] = [];

  // Determine if categorical encoding is needed
  const catEncoders: {
    featureIdx: number;
    colName: string;
    isCategorical: boolean;
    categories?: string[];
  }[] = [];

  for (let f = 0; f < config.featureColumns.length; f++) {
    const colName = config.featureColumns[f];
    const isNum = dataset.columnTypes[colName] === 'numeric';
    if (isNum) {
      catEncoders.push({ featureIdx: f, colName, isCategorical: false });
      expandedFeatureNames.push(colName);
    } else {
      const distinctCats = Array.from(
        new Set(validRows.map((r) => String(r.features[f])))
      ).sort();
      catEncoders.push({
        featureIdx: f,
        colName,
        isCategorical: true,
        categories: distinctCats,
      });

      if (config.encoding === 'onehot') {
        for (const cat of distinctCats) {
          expandedFeatureNames.push(`${colName}_${cat}`);
        }
      } else {
        expandedFeatureNames.push(colName);
      }
    }
  }

  for (const r of validRows) {
    const rowVec: number[] = [];
    for (const enc of catEncoders) {
      const rawVal = r.features[enc.featureIdx];
      if (!enc.isCategorical) {
        rowVec.push(Number(rawVal) || 0);
      } else {
        const catStr = String(rawVal);
        if (config.encoding === 'onehot' && enc.categories) {
          for (const c of enc.categories) {
            rowVec.push(c === catStr ? 1 : 0);
          }
        } else {
          // Label encoding
          const code = enc.categories ? enc.categories.indexOf(catStr) : 0;
          rowVec.push(code >= 0 ? code : 0);
        }
      }
    }
    processedXRows.push(rowVec);
  }

  // 5. Train / Test Split with Stratification
  const n = processedXRows.length;
  const indices = Array.from({ length: n }, (_, i) => i);

  // Seeded pseudo-random shuffle
  let seed = config.randomSeed || 42;
  const pseudoRandom = () => {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };

  const trainIndices: number[] = [];
  const testIndices: number[] = [];

  if (config.stratify) {
    const class0Indices = indices.filter((i) => binaryY[i] === 0);
    const class1Indices = indices.filter((i) => binaryY[i] === 1);

    // Shuffle both
    shuffle(class0Indices, pseudoRandom);
    shuffle(class1Indices, pseudoRandom);

    const testCount0 = Math.max(1, Math.round(class0Indices.length * config.testSize));
    const testCount1 = Math.max(1, Math.round(class1Indices.length * config.testSize));

    testIndices.push(...class0Indices.slice(0, testCount0));
    trainIndices.push(...class0Indices.slice(testCount0));

    testIndices.push(...class1Indices.slice(0, testCount1));
    trainIndices.push(...class1Indices.slice(testCount1));
  } else {
    shuffle(indices, pseudoRandom);
    const testCount = Math.max(1, Math.round(n * config.testSize));
    testIndices.push(...indices.slice(0, testCount));
    trainIndices.push(...indices.slice(testCount));
  }

  // Build raw train and test matrices
  const rawTrainX = trainIndices.map((i) => processedXRows[i]);
  const trainY = trainIndices.map((i) => binaryY[i]);
  const rawTestX = testIndices.map((i) => processedXRows[i]);
  const testY = testIndices.map((i) => binaryY[i]);

  // 6. Feature Scaling (Fit on Train only, transform Train & Test)
  const numFeatures = expandedFeatureNames.length;
  const means: number[] = new Array(numFeatures).fill(0);
  const stds: number[] = new Array(numFeatures).fill(1);
  const mins: number[] = new Array(numFeatures).fill(0);
  const maxs: number[] = new Array(numFeatures).fill(1);

  if (config.scaler !== 'none') {
    for (let f = 0; f < numFeatures; f++) {
      const trainCol = rawTrainX.map((r) => r[f]);
      const minV = Math.min(...trainCol);
      const maxV = Math.max(...trainCol);
      mins[f] = minV;
      maxs[f] = maxV;

      const sum = trainCol.reduce((s, v) => s + v, 0);
      const mean = sum / (trainCol.length || 1);
      means[f] = mean;

      const variance =
        trainCol.reduce((s, v) => s + (v - mean) ** 2, 0) /
        Math.max(1, trainCol.length - 1);
      stds[f] = Math.sqrt(variance) || 1; // avoid divide by zero
    }
  }

  const transformRow = (row: number[]): number[] => {
    if (config.scaler === 'standard') {
      return row.map((v, f) => (v - means[f]) / stds[f]);
    }
    if (config.scaler === 'minmax') {
      return row.map((v, f) => {
        const range = maxs[f] - mins[f];
        return range === 0 ? 0 : (v - mins[f]) / range;
      });
    }
    return [...row];
  };

  const trainX = rawTrainX.map(transformRow);
  const testX = rawTestX.map(transformRow);

  const c0Count = binaryY.filter((y) => y === 0).length;
  const c1Count = binaryY.filter((y) => y === 1).length;

  return {
    featureNames: expandedFeatureNames,
    targetColumn: config.targetColumn,
    targetClasses: [class0, class1],
    trainX,
    trainY,
    testX,
    testY,
    scalerParams: {
      type: config.scaler,
      means,
      stds,
      mins,
      maxs,
    },
    rawStats: {
      totalSamples: n,
      trainSamples: trainX.length,
      testSamples: testX.length,
      class0Count: c0Count,
      class1Count: c1Count,
    },
  };
}

function shuffle<T>(array: T[], randomFn: () => number) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(randomFn() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}
