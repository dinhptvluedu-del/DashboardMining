import React, { useState } from 'react';
import {
  SlidersHorizontal,
  CheckCircle2,
  ArrowRight,
  Split,
  Scale,
  Sparkles,
} from 'lucide-react';
import { Dataset, PreprocessingConfig, PreprocessedData } from '../types';
import { preprocessDataset } from '../utils/dataStats';

interface PreprocessingViewProps {
  dataset: Dataset;
  preprocessed: PreprocessedData | null;
  onPreprocessed: (data: PreprocessedData) => void;
  onNavigateToTraining: () => void;
}

export const PreprocessingView: React.FC<PreprocessingViewProps> = ({
  dataset,
  preprocessed,
  onPreprocessed,
  onNavigateToTraining,
}) => {
  // Initial configs
  const defaultTarget = dataset.headers.includes('diabetes')
    ? 'diabetes'
    : dataset.headers[dataset.headers.length - 1];

  const defaultFeatures = dataset.headers.filter((h) => h !== defaultTarget);

  const [config, setConfig] = useState<PreprocessingConfig>({
    targetColumn: defaultTarget,
    featureColumns: defaultFeatures,
    missingStrategy: 'mean',
    scaler: 'standard',
    encoding: 'label',
    testSize: 0.2,
    randomSeed: 42,
    stratify: true,
  });

  const [isProcessing, setIsProcessing] = useState(false);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleTargetChange = (newTarget: string) => {
    const updatedFeatures = config.featureColumns.filter((f) => f !== newTarget);
    if (updatedFeatures.length === 0) {
      // Pick another column as feature
      const altFeature = dataset.headers.find((h) => h !== newTarget);
      if (altFeature) updatedFeatures.push(altFeature);
    }
    setConfig({
      ...config,
      targetColumn: newTarget,
      featureColumns: updatedFeatures,
    });
  };

  const handleFeatureToggle = (feat: string) => {
    if (config.featureColumns.includes(feat)) {
      if (config.featureColumns.length <= 1) {
        setErrorMsg('Phải giữ lại ít nhất 1 biến đặc trưng đầu vào!');
        return;
      }
      setConfig({
        ...config,
        featureColumns: config.featureColumns.filter((f) => f !== feat),
      });
    } else {
      setConfig({
        ...config,
        featureColumns: [...config.featureColumns, feat],
      });
    }
    setErrorMsg(null);
  };

  const executePreprocessing = () => {
    setIsProcessing(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    try {
      const result = preprocessDataset(dataset, config);
      onPreprocessed(result);
      setSuccessMsg(
        `Tiền xử lý thành công! Tập Train: ${result.trainX.length} mẫu | Tập Test: ${result.testX.length} mẫu`
      );
    } catch (err: any) {
      setErrorMsg(err.message || 'Lỗi khi tiền xử lý dữ liệu.');
    } finally {
      setIsProcessing(false);
    }
  };

  const trainCountEstimate = Math.round(dataset.rowCount * (1 - config.testSize));
  const testCountEstimate = dataset.rowCount - trainCountEstimate;

  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">
          3. Tiền Xử Lý Dữ Liệu Đầu Vào (Data Preprocessing)
        </h1>
        <p className="text-xs text-slate-500 mt-1">
          Lựa chọn biến mục tiêu, chuẩn hóa giá trị thang đo, xử lý giá trị khuyết và phân chia tập dữ liệu Train / Test
        </p>
      </div>

      {/* Notifications */}
      {successMsg && (
        <div className="p-3.5 bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs rounded-xl flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
            <span className="font-medium">{successMsg}</span>
          </div>
          <button
            onClick={onNavigateToTraining}
            className="inline-flex items-center gap-1.5 px-3 py-1 font-semibold text-emerald-900 bg-emerald-200/70 hover:bg-emerald-300 rounded-lg transition-colors cursor-pointer"
          >
            <span>Sang bước Huấn luyện XGBoost</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {errorMsg && (
        <div className="p-3.5 bg-rose-50 border border-rose-200 text-rose-800 text-xs rounded-xl">
          {errorMsg}
        </div>
      )}

      {/* Grid: 2 columns */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Target & Feature Selection */}
        <div className="lg:col-span-5 space-y-6">
          {/* Target Selection */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
            <div>
              <label className="block text-xs font-bold text-slate-900 uppercase tracking-wider mb-1">
                1. Chọn biến mục tiêu (Target Column / Y)
              </label>
              <p className="text-xs text-slate-500 mb-2">
                Biến cần dự đoán phân loại (nhị phân 0 hoặc 1)
              </p>
              <select
                value={config.targetColumn}
                onChange={(e) => handleTargetChange(e.target.value)}
                className="w-full text-xs font-medium text-slate-800 bg-slate-50 border border-slate-200 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 cursor-pointer"
              >
                {dataset.headers.map((h) => (
                  <option key={h} value={h}>
                    {h} ({dataset.columnTypes[h]})
                  </option>
                ))}
              </select>
            </div>

            <div className="pt-2 border-t border-slate-100">
              <label className="block text-xs font-bold text-slate-900 uppercase tracking-wider mb-1">
                2. Chọn các biến đầu vào (Feature Columns / X)
              </label>
              <p className="text-xs text-slate-500 mb-2">
                Chọn ít nhất một hoặc nhiều đặc trưng số / phân loại
              </p>
              <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                {dataset.headers
                  .filter((h) => h !== config.targetColumn)
                  .map((h) => {
                    const isChecked = config.featureColumns.includes(h);
                    return (
                      <label
                        key={h}
                        className={`flex items-center justify-between p-2.5 rounded-lg border text-xs cursor-pointer transition-colors ${
                          isChecked
                            ? 'bg-blue-50/60 border-blue-300 text-slate-900'
                            : 'bg-white border-slate-200 text-slate-500 hover:bg-slate-50'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={isChecked}
                            onChange={() => handleFeatureToggle(h)}
                            className="rounded border-slate-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                          />
                          <span className="font-semibold">{h}</span>
                        </div>
                        <span className="text-[11px] text-slate-400">
                          {dataset.columnTypes[h]}
                        </span>
                      </label>
                    );
                  })}
              </div>
            </div>
          </div>

          {/* Missing value & Encoding options */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
            <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              3. Xử lý giá trị khuyết & Mã hóa
            </h2>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Xử lý Missing Values:
              </label>
              <select
                value={config.missingStrategy}
                onChange={(e) =>
                  setConfig({ ...config, missingStrategy: e.target.value as any })
                }
                className="w-full text-xs text-slate-800 bg-slate-50 border border-slate-200 rounded-lg p-2 focus:ring-2 focus:ring-blue-500/20"
              >
                <option value="mean">Điền giá trị trung bình (Mean Imputation)</option>
                <option value="median">Điền giá trị trung vị (Median Imputation)</option>
                <option value="drop">Xóa các dòng chứa giá trị khuyết (Drop NA)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Mã hóa biến phân loại (Categorical Encoding):
              </label>
              <select
                value={config.encoding}
                onChange={(e) =>
                  setConfig({ ...config, encoding: e.target.value as any })
                }
                className="w-full text-xs text-slate-800 bg-slate-50 border border-slate-200 rounded-lg p-2 focus:ring-2 focus:ring-blue-500/20"
              >
                <option value="label">Label Encoding (Ánh xạ số nguyên 0, 1, 2...)</option>
                <option value="onehot">One-Hot Encoding (Tạo cột giả định nhị phân)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Right Column: Scaling & Train/Test Split */}
        <div className="lg:col-span-7 space-y-6">
          {/* Feature Scaling Card */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
            <div className="flex items-center gap-2">
              <Scale className="w-4 h-4 text-indigo-600" />
              <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                4. Chuẩn hóa đặc trưng (Feature Scaling)
              </h2>
            </div>
            <p className="text-xs text-slate-500">
              Mặc dù cây quyết định không nhạy cảm với thang đo, việc chuẩn hóa giúp các thuật toán gradient boosting tối ưu hóa hội tụ nhanh hơn.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {[
                {
                  id: 'standard',
                  title: 'StandardScaler',
                  desc: 'Z-score (Mean = 0, Std = 1)',
                },
                {
                  id: 'minmax',
                  title: 'MinMaxScaler',
                  desc: 'Khoảng cố định [0, 1]',
                },
                {
                  id: 'none',
                  title: 'Không chuẩn hóa',
                  desc: 'Giữ nguyên giá trị gốc',
                },
              ].map((sc) => {
                const isSelected = config.scaler === sc.id;
                return (
                  <button
                    key={sc.id}
                    type="button"
                    onClick={() => setConfig({ ...config, scaler: sc.id as any })}
                    className={`p-3 rounded-lg border text-left cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-blue-50/80 border-blue-400 text-blue-900 ring-1 ring-blue-400'
                        : 'bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100'
                    }`}
                  >
                    <div className="font-semibold text-xs">{sc.title}</div>
                    <div className="text-[11px] text-slate-500 mt-0.5">{sc.desc}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Train / Test Split Card */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Split className="w-4 h-4 text-blue-600" />
                <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                  5. Phân chia Train / Test (Split Ratio)
                </h2>
              </div>
              <span className="text-xs font-mono font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                {Math.round((1 - config.testSize) * 100)}% Train / {Math.round(config.testSize * 100)}% Test
              </span>
            </div>

            {/* Slider */}
            <div className="space-y-2">
              <input
                type="range"
                min="0.1"
                max="0.4"
                step="0.05"
                value={config.testSize}
                onChange={(e) =>
                  setConfig({ ...config, testSize: parseFloat(e.target.value) })
                }
                className="w-full accent-blue-600 cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-slate-400">
                <span>90% / 10%</span>
                <span>80% / 20% (Chuẩn)</span>
                <span>70% / 30%</span>
                <span>60% / 40%</span>
              </div>
            </div>

            {/* Visual ratio bar */}
            <div className="h-4 rounded-lg overflow-hidden flex bg-slate-100 border border-slate-200">
              <div
                className="bg-blue-600 h-full flex items-center justify-center text-[10px] text-white font-medium"
                style={{ width: `${(1 - config.testSize) * 100}%` }}
              >
                Tập Train ({trainCountEstimate} mẫu)
              </div>
              <div
                className="bg-amber-500 h-full flex items-center justify-center text-[10px] text-white font-medium"
                style={{ width: `${config.testSize * 100}%` }}
              >
                Test ({testCountEstimate})
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-2">
              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Random Seed (Hạt giống ngẫu nhiên):
                </label>
                <input
                  type="number"
                  value={config.randomSeed}
                  onChange={(e) =>
                    setConfig({ ...config, randomSeed: parseInt(e.target.value) || 42 })
                  }
                  className="w-full text-xs font-mono bg-slate-50 border border-slate-200 rounded-lg p-2 focus:ring-2 focus:ring-blue-500/20"
                />
              </div>

              <div className="flex items-center pt-5">
                <label className="flex items-center gap-2 text-xs font-medium text-slate-700 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.stratify}
                    onChange={(e) =>
                      setConfig({ ...config, stratify: e.target.checked })
                    }
                    className="rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span>Phân tầng (Stratified Sampling)</span>
                </label>
              </div>
            </div>
          </div>

          {/* Action Button */}
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="text-xs text-slate-600">
              <p className="font-semibold text-slate-900">Sẵn sàng chuyển đổi dữ liệu:</p>
              <p>
                Target: <strong>{config.targetColumn}</strong> · Features:{' '}
                <strong>{config.featureColumns.join(', ')}</strong>
              </p>
            </div>

            <button
              onClick={executePreprocessing}
              disabled={isProcessing}
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-5 py-2.5 text-xs font-bold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors cursor-pointer disabled:opacity-50"
            >
              <SlidersHorizontal className="w-4 h-4" />
              <span>{isProcessing ? 'Đang xử lý...' : 'Thực hiện Tiền Xử Lý Ngay'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Preprocessed Preview Summary Card */}
      {preprocessed && (
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-emerald-600" />
            <h2 className="text-sm font-bold text-slate-900">
              Kết Quả Tiền Xử Lý Dữ Liệu Sẵn Sàng Cho XGBoost
            </h2>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-slate-500">Mẫu huấn luyện (Train):</span>
              <p className="font-bold text-slate-900 text-base">
                {preprocessed.trainX.length} dòng
              </p>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-slate-500">Mẫu kiểm thử (Test):</span>
              <p className="font-bold text-slate-900 text-base">
                {preprocessed.testX.length} dòng
              </p>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-slate-500">Số chiều đặc trưng:</span>
              <p className="font-bold text-slate-900 text-base">
                {preprocessed.featureNames.length} chiều
              </p>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-slate-500">Phương pháp chuẩn hóa:</span>
              <p className="font-bold text-slate-900 text-base uppercase">
                {preprocessed.scalerParams.type}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
