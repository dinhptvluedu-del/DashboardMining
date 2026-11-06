import React, { useState } from 'react';
import {
  Cpu,
  Play,
  RotateCcw,
  CheckCircle,
  ArrowRight,
  TrendingDown,
  Sparkles,
} from 'lucide-react';
import confetti from 'canvas-confetti';
import { PreprocessedData, XGBoostParams, TrainedModel } from '../types';
import { trainXGBoost } from '../utils/xgboostEngine';

interface TrainingViewProps {
  preprocessed: PreprocessedData | null;
  model: TrainedModel | null;
  onModelTrained: (model: TrainedModel) => void;
  onNavigateToEvaluation: () => void;
  onNavigateToPreprocessing: () => void;
}

export const TrainingView: React.FC<TrainingViewProps> = ({
  preprocessed,
  model,
  onModelTrained,
  onNavigateToEvaluation,
  onNavigateToPreprocessing,
}) => {
  const [params, setParams] = useState<XGBoostParams>({
    n_estimators: 60,
    learning_rate: 0.1,
    max_depth: 4,
    subsample: 0.8,
    colsample_bytree: 1.0,
    min_child_weight: 1,
    gamma: 0.0,
    reg_lambda: 1.0,
  });

  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [lossData, setLossData] = useState<{ step: number; trainLoss: number; testLoss: number }[]>(
    model?.lossHistory || []
  );

  const resetDefaultParams = () => {
    setParams({
      n_estimators: 60,
      learning_rate: 0.1,
      max_depth: 4,
      subsample: 0.8,
      colsample_bytree: 1.0,
      min_child_weight: 1,
      gamma: 0.0,
      reg_lambda: 1.0,
    });
  };

  const startTraining = async () => {
    if (!preprocessed) return;

    setIsTraining(true);
    setProgress(5);
    setStatusMessage('Đang khởi tạo cấu trúc cây Gradient Boosted Decision Trees...');

    // Small delay to allow UI to render progress
    await new Promise((resolve) => setTimeout(resolve, 80));

    try {
      setStatusMessage('Đang tính toán gradients & hessians trên tập Train...');
      setProgress(25);

      // We run training
      const trained = trainXGBoost(preprocessed, params, (step, total, trLoss, teLoss) => {
        const pct = Math.round(25 + (step / total) * 70);
        setProgress(pct);
        setStatusMessage(`Đang xây dựng cây thứ ${step} / ${total} (Loss: ${trLoss.toFixed(4)})...`);
      });

      setProgress(100);
      setStatusMessage('Đang đánh giá mô hình trên tập kiểm thử (Test)...');
      await new Promise((resolve) => setTimeout(resolve, 150));

      setLossData(trained.lossHistory);
      onModelTrained(trained);

      if (trained.evaluation.accuracy >= 0.85) {
        confetti({
          particleCount: 80,
          spread: 60,
          origin: { y: 0.6 },
        });
      }
    } catch (err: any) {
      alert(`Lỗi khi huấn luyện: ${err.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  if (!preprocessed) {
    return (
      <div className="bg-white p-8 rounded-xl border border-slate-200 text-center space-y-4 shadow-sm">
        <div className="w-12 h-12 rounded-full bg-amber-50 text-amber-600 flex items-center justify-center mx-auto">
          <Cpu className="w-6 h-6" />
        </div>
        <h2 className="text-base font-bold text-slate-900">
          Chưa Có Dữ Liệu Đã Tiền Xử Lý
        </h2>
        <p className="text-xs text-slate-500 max-w-md mx-auto">
          Vui lòng hoàn thành bước tiền xử lý dữ liệu trước khi tiến hành cấu hình siêu tham số và huấn luyện mô hình XGBoost.
        </p>
        <button
          onClick={onNavigateToPreprocessing}
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors cursor-pointer"
        >
          <span>Đến bước Tiền xử lý</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">
            4. Cấu Hình & Huấn Luyện Mô Hình XGBoost
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Tinh chỉnh các siêu tham số (Hyperparameters), quan sát quá trình hội tụ hàm mất mát và tối ưu hóa độ chính xác
          </p>
        </div>

        <button
          type="button"
          onClick={resetDefaultParams}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg transition-colors cursor-pointer self-start sm:self-auto"
        >
          <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
          <span>Đặt lại mặc định</span>
        </button>
      </div>

      {/* Hyperparameter Config Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {/* Param 1: n_estimators */}
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-bold text-slate-800">
              Số lượng cây (n_estimators)
            </label>
            <span className="text-xs font-mono font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
              {params.n_estimators}
            </span>
          </div>
          <p className="text-[11px] text-slate-500">
            Số vòng lặp boosting tăng cường liên tiếp
          </p>
          <input
            type="range"
            min="10"
            max="200"
            step="5"
            value={params.n_estimators}
            onChange={(e) =>
              setParams({ ...params, n_estimators: parseInt(e.target.value) })
            }
            className="w-full accent-blue-600 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>10</span>
            <span>60 (Chuẩn)</span>
            <span>200</span>
          </div>
        </div>

        {/* Param 2: learning_rate */}
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-bold text-slate-800">
              Tốc độ học (learning_rate / eta)
            </label>
            <span className="text-xs font-mono font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
              {params.learning_rate}
            </span>
          </div>
          <p className="text-[11px] text-slate-500">
            Hệ số co ngót trọng số nút lá để chống overfitting
          </p>
          <input
            type="range"
            min="0.01"
            max="0.4"
            step="0.01"
            value={params.learning_rate}
            onChange={(e) =>
              setParams({ ...params, learning_rate: parseFloat(e.target.value) })
            }
            className="w-full accent-blue-600 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>0.01</span>
            <span>0.1 (Khuyên dùng)</span>
            <span>0.4</span>
          </div>
        </div>

        {/* Param 3: max_depth */}
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-bold text-slate-800">
              Độ sâu tối đa (max_depth)
            </label>
            <span className="text-xs font-mono font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
              {params.max_depth}
            </span>
          </div>
          <p className="text-[11px] text-slate-500">
            Giới hạn số tầng rẽ nhánh của mỗi cây con
          </p>
          <input
            type="range"
            min="1"
            max="8"
            step="1"
            value={params.max_depth}
            onChange={(e) =>
              setParams({ ...params, max_depth: parseInt(e.target.value) })
            }
            className="w-full accent-blue-600 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>1 (Cây đơn giản)</span>
            <span>4 (Chuẩn)</span>
            <span>8 (Phức tạp)</span>
          </div>
        </div>

        {/* Param 4: subsample */}
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-bold text-slate-800">
              Tỷ lệ lấy mẫu hàng (subsample)
            </label>
            <span className="text-xs font-mono font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
              {(params.subsample * 100).toFixed(0)}%
            </span>
          </div>
          <p className="text-[11px] text-slate-500">
            Tỷ lệ ngẫu nhiên số dòng dữ liệu dùng cho mỗi cây
          </p>
          <input
            type="range"
            min="0.5"
            max="1.0"
            step="0.05"
            value={params.subsample}
            onChange={(e) =>
              setParams({ ...params, subsample: parseFloat(e.target.value) })
            }
            className="w-full accent-blue-600 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>50%</span>
            <span>80% (Khuyên dùng)</span>
            <span>100%</span>
          </div>
        </div>

        {/* Param 5: gamma */}
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-bold text-slate-800">
              Hệ số phạt tách nhánh (gamma)
            </label>
            <span className="text-xs font-mono font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
              {params.gamma}
            </span>
          </div>
          <p className="text-[11px] text-slate-500">
            Mức gain tối thiểu để cho phép chia nhánh mới
          </p>
          <input
            type="range"
            min="0.0"
            max="3.0"
            step="0.1"
            value={params.gamma}
            onChange={(e) =>
              setParams({ ...params, gamma: parseFloat(e.target.value) })
            }
            className="w-full accent-blue-600 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>0.0 (Không phạt)</span>
            <span>1.0</span>
            <span>3.0 (Phạt mạnh)</span>
          </div>
        </div>

        {/* Param 6: min_child_weight */}
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-bold text-slate-800">
              Trọng số tối thiểu nút con (min_child_weight)
            </label>
            <span className="text-xs font-mono font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
              {params.min_child_weight}
            </span>
          </div>
          <p className="text-[11px] text-slate-500">
            Tổng độ cong hessian tối thiểu ở mỗi nút lá
          </p>
          <input
            type="range"
            min="1"
            max="10"
            step="1"
            value={params.min_child_weight}
            onChange={(e) =>
              setParams({ ...params, min_child_weight: parseInt(e.target.value) })
            }
            className="w-full accent-blue-600 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>1</span>
            <span>5</span>
            <span>10</span>
          </div>
        </div>
      </div>

      {/* Action Button & Live Progress */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-xs text-slate-600">
            <div className="flex items-center gap-2">
              <span className="font-bold text-slate-900 text-sm">
                Sẵn sàng khởi chạy thuật toán Extreme Gradient Boosting:
              </span>
            </div>
            <p className="text-slate-500 mt-0.5">
              Tập huấn luyện: <strong>{preprocessed.trainX.length} mẫu</strong> · Tập kiểm thử: <strong>{preprocessed.testX.length} mẫu</strong>
            </p>
          </div>

          <button
            onClick={startTraining}
            disabled={isTraining}
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3 text-xs font-bold text-white bg-blue-600 hover:bg-blue-700 rounded-xl shadow-md shadow-blue-500/20 transition-all cursor-pointer disabled:opacity-50"
          >
            <Play className={`w-4 h-4 ${isTraining ? 'animate-spin' : 'fill-white'}`} />
            <span>{isTraining ? 'Đang Huấn Luyện Mô Hình...' : 'Huấn Luyện Mô Hình (Train Model)'}</span>
          </button>
        </div>

        {/* Progress bar */}
        {isTraining && (
          <div className="space-y-2 pt-2">
            <div className="flex justify-between text-xs font-medium text-slate-700">
              <span>{statusMessage}</span>
              <span>{progress}%</span>
            </div>
            <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
              <div
                className="bg-blue-600 h-full rounded-full transition-all duration-200"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>

      {/* Loss Curves & Convergence Display */}
      {lossData.length > 0 && (
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-emerald-600" />
              <div>
                <h2 className="text-sm font-bold text-slate-900">
                  Đường Cong Hàm Mất Mát (Loss History & Convergence Curve)
                </h2>
                <p className="text-xs text-slate-500">
                  Log-loss qua từng vòng lặp: Train Loss (xanh dương) vs Test Loss (hổ phách)
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4 text-xs font-medium">
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-blue-600"></span>
                <span className="text-slate-700">Train Loss</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-amber-500"></span>
                <span className="text-slate-700">Test Loss</span>
              </div>
            </div>
          </div>

          {/* SVG Line Chart */}
          <div className="h-56 w-full">
            <svg viewBox="0 0 600 200" className="w-full h-full bg-slate-50 rounded-lg p-2 border border-slate-100">
              {/* Grid lines */}
              <line x1="40" y1="20" x2="40" y2="170" stroke="#cbd5e1" />
              <line x1="40" y1="170" x2="580" y2="170" stroke="#cbd5e1" />
              <line x1="40" y1="95" x2="580" y2="95" stroke="#f1f5f9" strokeDasharray="3 3" />

              {/* Generate points */}
              {(() => {
                const maxLoss = Math.max(
                  ...lossData.map((d) => Math.max(d.trainLoss, d.testLoss)),
                  0.7
                );
                const minLoss = Math.min(
                  ...lossData.map((d) => Math.min(d.trainLoss, d.testLoss)),
                  0.05
                );
                const range = maxLoss - minLoss || 1;
                const n = lossData.length;

                const trainPts = lossData
                  .map((d, i) => {
                    const x = 50 + (i / (n - 1 || 1)) * 520;
                    const y = 160 - ((d.trainLoss - minLoss) / range) * 130;
                    return `${x},${y}`;
                  })
                  .join(' ');

                const testPts = lossData
                  .map((d, i) => {
                    const x = 50 + (i / (n - 1 || 1)) * 520;
                    const y = 160 - ((d.testLoss - minLoss) / range) * 130;
                    return `${x},${y}`;
                  })
                  .join(' ');

                return (
                  <>
                    <polyline
                      fill="none"
                      stroke="#2563eb"
                      strokeWidth="2.5"
                      points={trainPts}
                    />
                    <polyline
                      fill="none"
                      stroke="#f59e0b"
                      strokeWidth="2"
                      strokeDasharray="4 2"
                      points={testPts}
                    />
                  </>
                );
              })()}

              <text x="300" y="190" textAnchor="middle" className="text-[10px] fill-slate-500">
                Số vòng lặp Boosting (Iterations)
              </text>
              <text x="15" y="95" textAnchor="middle" transform="rotate(-90 15,95)" className="text-[10px] fill-slate-500">
                Log Loss
              </text>
            </svg>
          </div>
        </div>
      )}

      {/* Trained Model Notification & Next Step Banner */}
      {model && (
        <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl flex flex-col sm:flex-row items-center justify-between gap-4 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center flex-shrink-0">
              <CheckCircle className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-xs font-bold text-emerald-950">
                Mô Hình XGBoost Đã Được Huấn Luyện Thành Công!
              </h2>
              <p className="text-xs text-emerald-800">
                Độ chính xác (Accuracy): <strong>{(model.evaluation.accuracy * 100).toFixed(2)}%</strong> · ROC-AUC: <strong>{model.evaluation.rocAuc.toFixed(4)}</strong> · Thời gian: {model.trainingDurationMs}ms
              </p>
            </div>
          </div>

          <button
            onClick={onNavigateToEvaluation}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-bold text-white bg-emerald-600 hover:bg-emerald-700 rounded-lg shadow-sm transition-colors cursor-pointer flex-shrink-0"
          >
            <span>Xem Đánh Giá & Ma Trận Nhầm Lẫn</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
};
