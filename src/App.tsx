/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { DashboardOverview } from './components/DashboardOverview';
import { DescriptiveStats } from './components/DescriptiveStats';
import { PreprocessingView } from './components/PreprocessingView';
import { TrainingView } from './components/TrainingView';
import { EvaluationView } from './components/EvaluationView';
import { PredictionView } from './components/PredictionView';
import { HistoryLogView } from './components/HistoryLogView';
import { PythonSourceCodeView } from './components/PythonSourceCodeView';
import { Footer } from './components/Footer';

import {
  Dataset,
  ActiveTab,
  TrainedModel,
  PreprocessedData,
  TrainingHistoryEntry,
  XGBoostParams,
} from './types';
import { parseCSV } from './utils/csvParser';
import { NAIVE_BAYES_CSV_CONTENT } from './utils/naiveBayesData';
import { computeColumnStats, computeCorrelationMatrix, preprocessDataset } from './utils/dataStats';
import { trainXGBoost } from './utils/xgboostEngine';
import { APP_PY_CODE, REQUIREMENTS_TXT_CODE } from './utils/sourceStrings';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('overview');

  // 1. Initial Dataset
  const defaultDataset = useMemo(() => {
    return parseCSV(NAIVE_BAYES_CSV_CONTENT, 'Naive-Bayes-Classification-Data.csv');
  }, []);

  const [dataset, setDataset] = useState<Dataset>(defaultDataset);

  // Hidden upload input ref
  const uploadInputRef = useRef<HTMLInputElement>(null);

  // 2. Computed Statistics & Correlation Matrix
  const stats = useMemo(() => computeColumnStats(dataset), [dataset]);
  const correlation = useMemo(() => computeCorrelationMatrix(dataset), [dataset]);

  // 3. Preprocessing state
  const [preprocessed, setPreprocessed] = useState<PreprocessedData | null>(null);

  // 4. Model state
  const [model, setModel] = useState<TrainedModel | null>(null);

  // 5. Training History state
  const [history, setHistory] = useState<TrainingHistoryEntry[]>([]);

  // Auto-bootstrap initial preprocessing & baseline model on first mount
  useEffect(() => {
    try {
      const targetCol = dataset.headers.includes('diabetes')
        ? 'diabetes'
        : dataset.headers[dataset.headers.length - 1];
      const featureCols = dataset.headers.filter((h) => h !== targetCol);

      const prep = preprocessDataset(dataset, {
        targetColumn: targetCol,
        featureColumns: featureCols,
        missingStrategy: 'mean',
        scaler: 'standard',
        encoding: 'label',
        testSize: 0.2,
        randomSeed: 42,
        stratify: true,
      });
      setPreprocessed(prep);

      // Train baseline model
      const initParams: XGBoostParams = {
        n_estimators: 60,
        learning_rate: 0.1,
        max_depth: 4,
        subsample: 0.8,
        colsample_bytree: 1.0,
        min_child_weight: 1,
        gamma: 0.0,
        reg_lambda: 1.0,
      };

      const baselineModel = trainXGBoost(prep, initParams);
      setModel(baselineModel);

      setHistory([
        {
          id: baselineModel.id,
          timestamp: baselineModel.createdAt,
          datasetName: dataset.name,
          params: initParams,
          accuracy: baselineModel.evaluation.accuracy,
          precision: baselineModel.evaluation.precision,
          recall: baselineModel.evaluation.recall,
          f1Score: baselineModel.evaluation.f1Score,
          rocAuc: baselineModel.evaluation.rocAuc,
          durationMs: baselineModel.trainingDurationMs,
        },
      ]);
    } catch (e) {
      console.warn('Auto-bootstrap baseline model:', e);
    }
  }, [dataset]);

  // Handle new custom CSV upload
  const handleUploadCSV = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const newDataset = parseCSV(text, file.name);
        setDataset(newDataset);
        setPreprocessed(null);
        setModel(null);
        setActiveTab('overview');
      } catch (err: any) {
        alert(err.message || 'Không thể đọc tệp CSV.');
      }
    };
    reader.readAsText(file);
    // Reset input value so same file can be reloaded
    e.target.value = '';
  };

  const handleResetToDefault = () => {
    setDataset(defaultDataset);
    setPreprocessed(null);
    setModel(null);
    setActiveTab('overview');
  };

  const handleModelTrained = (newModel: TrainedModel) => {
    setModel(newModel);
    setHistory((prev) => [
      {
        id: newModel.id,
        timestamp: newModel.createdAt,
        datasetName: dataset.name,
        params: newModel.params,
        accuracy: newModel.evaluation.accuracy,
        precision: newModel.evaluation.precision,
        recall: newModel.evaluation.recall,
        f1Score: newModel.evaluation.f1Score,
        rocAuc: newModel.evaluation.rocAuc,
        durationMs: newModel.trainingDurationMs,
      },
      ...prev,
    ]);
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans antialiased text-slate-800">
      {/* Hidden file uploader */}
      <input
        type="file"
        ref={uploadInputRef}
        accept=".csv"
        onChange={handleUploadCSV}
        className="hidden"
      />

      {/* Top Navbar */}
      <Navbar
        dataset={dataset}
        model={model}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onResetData={handleResetToDefault}
        onUploadClick={() => uploadInputRef.current?.click()}
      />

      {/* Main Layout: Sidebar + Content */}
      <div className="flex-1 flex flex-col md:flex-row max-w-7xl w-full mx-auto">
        <Sidebar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          dataset={dataset}
          model={model}
          preprocessed={preprocessed}
        />

        {/* Dynamic Main Content Area */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 min-w-0 overflow-y-auto">
          {activeTab === 'overview' && (
            <DashboardOverview
              dataset={dataset}
              stats={stats}
              onNavigateToPreprocessing={() => setActiveTab('preprocessing')}
            />
          )}

          {activeTab === 'descriptive' && (
            <DescriptiveStats
              dataset={dataset}
              stats={stats}
              correlation={correlation}
            />
          )}

          {activeTab === 'preprocessing' && (
            <PreprocessingView
              dataset={dataset}
              preprocessed={preprocessed}
              onPreprocessed={(data) => setPreprocessed(data)}
              onNavigateToTraining={() => setActiveTab('training')}
            />
          )}

          {activeTab === 'training' && (
            <TrainingView
              preprocessed={preprocessed}
              model={model}
              onModelTrained={handleModelTrained}
              onNavigateToEvaluation={() => setActiveTab('evaluation')}
              onNavigateToPreprocessing={() => setActiveTab('preprocessing')}
            />
          )}

          {activeTab === 'evaluation' && (
            <EvaluationView
              model={model}
              onNavigateToTraining={() => setActiveTab('training')}
              onNavigateToPrediction={() => setActiveTab('prediction')}
            />
          )}

          {activeTab === 'prediction' && (
            <PredictionView
              model={model}
              onNavigateToTraining={() => setActiveTab('training')}
            />
          )}

          {activeTab === 'history' && (
            <HistoryLogView
              history={history}
              onApplyParams={() => {}}
              onNavigateToTraining={() => setActiveTab('training')}
            />
          )}

          {activeTab === 'code' && (
            <PythonSourceCodeView
              appPyCode={APP_PY_CODE}
              requirementsTxt={REQUIREMENTS_TXT_CODE}
            />
          )}
        </main>
      </div>

      {/* Footer */}
      <Footer />
    </div>
  );
}
