import React, { useState, useRef } from 'react';
import {
  Sparkles,
  Upload,
  Download,
  CheckCircle,
  AlertTriangle,
  FileSpreadsheet,
} from 'lucide-react';
import { TrainedModel } from '../types';
import { predictProbability } from '../utils/xgboostEngine';
import { parseCSV, exportToCSV, downloadCSV } from '../utils/csvParser';

interface PredictionViewProps {
  model: TrainedModel | null;
  onNavigateToTraining: () => void;
}

export const PredictionView: React.FC<PredictionViewProps> = ({
  model,
  onNavigateToTraining,
}) => {
  const [activeMode, setActiveMode] = useState<'single' | 'batch'>('single');

  // Single mode state
  const featureNames = model?.preprocessedData.featureNames || ['glucose', 'bloodpressure'];
  const [manualInputs, setManualInputs] = useState<Record<string, number>>(() => {
    const init: Record<string, number> = {};
    featureNames.forEach((f) => {
      if (f.toLowerCase().includes('glucose')) init[f] = 45;
      else if (f.toLowerCase().includes('blood') || f.toLowerCase().includes('pressure')) init[f] = 80;
      else init[f] = 50;
    });
    return init;
  });

  const [singlePrediction, setSinglePrediction] = useState<{
    prob: number;
    pred: number;
    risk: 'Low' | 'High';
  } | null>(null);

  // Batch mode state
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [batchResults, setBatchResults] = useState<{
    headers: string[];
    rows: (string | number)[][];
    total: number;
    positiveCount: number;
  } | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);

  if (!model) {
    return (
      <div className="bg-white p-8 rounded-xl border border-slate-200 text-center space-y-4 shadow-sm">
        <div className="w-12 h-12 rounded-full bg-indigo-50 text-indigo-600 flex items-center justify-center mx-auto">
          <Sparkles className="w-6 h-6" />
        </div>
        <h2 className="text-base font-bold text-slate-900">
          Chưa Có Mô Hình Sẵn Sàng Để Dự Đoán
        </h2>
        <p className="text-xs text-slate-500 max-w-md mx-auto">
          Mô hình XGBoost cần được huấn luyện trước để trích xuất cấu trúc cây và trọng số phục vụ dự báo.
        </p>
        <button
          onClick={onNavigateToTraining}
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors cursor-pointer"
        >
          <span>Huấn luyện mô hình ngay</span>
        </button>
      </div>
    );
  }

  const transformInputVector = (rawVec: number[]): number[] => {
    const { scalerParams } = model.preprocessedData;
    if (scalerParams.type === 'standard' && scalerParams.means && scalerParams.stds) {
      return rawVec.map((v, i) => (v - scalerParams.means![i]) / scalerParams.stds![i]);
    }
    if (scalerParams.type === 'minmax' && scalerParams.mins && scalerParams.maxs) {
      return rawVec.map((v, i) => {
        const range = scalerParams.maxs![i] - scalerParams.mins![i];
        return range === 0 ? 0 : (v - scalerParams.mins![i]) / range;
      });
    }
    return rawVec;
  };

  const handlePredictSingle = (e: React.FormEvent) => {
    e.preventDefault();
    const rawVector = featureNames.map((f) => manualInputs[f] ?? 0);
    const scaledVector = transformInputVector(rawVector);

    const prob = predictProbability(
      scaledVector,
      model.baseMargin,
      model.trees,
      model.params.learning_rate
    );

    const pred = prob >= 0.5 ? 1 : 0;
    setSinglePrediction({
      prob,
      pred,
      risk: prob >= 0.5 ? 'High' : 'Low',
    });
  };

  const handleBatchFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setBatchError(null);
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const parsed = parseCSV(text, file.name);

        // Verify feature columns existence
        const missingFeatures = featureNames.filter((f) => !parsed.headers.includes(f));
        if (missingFeatures.length > 0) {
          throw new Error(
            `Tệp CSV thiếu các cột đặc trưng bắt buộc: ${missingFeatures.join(', ')}`
          );
        }

        const featureIndices = featureNames.map((f) => parsed.headers.indexOf(f));

        const resultRows: (string | number)[][] = [];
        let posCount = 0;

        for (const row of parsed.rawRows) {
          const rawVector = featureIndices.map((idx) => {
            const v = row[idx];
            return typeof v === 'number' ? v : Number(v) || 0;
          });

          const scaledVector = transformInputVector(rawVector);
          const prob = predictProbability(
            scaledVector,
            model.baseMargin,
            model.trees,
            model.params.learning_rate
          );

          const pred = prob >= 0.5 ? 1 : 0;
          if (pred === 1) posCount++;

          const probPercent = Number((prob * 100).toFixed(2));
          const conclusion = pred === 1 ? 'Dương tính (Nguy cơ cao)' : 'Âm tính (Bình thường)';

          resultRows.push([...row, probPercent, pred, conclusion]);
        }

        const resultHeaders = [
          ...parsed.headers,
          'Xac_Suat_Duong_Tinh_%',
          'Du_Doan_Nhan',
          'Ket_Luan',
        ];

        setBatchResults({
          headers: resultHeaders,
          rows: resultRows,
          total: resultRows.length,
          positiveCount: posCount,
        });
      } catch (err: any) {
        setBatchError(err.message || 'Lỗi khi xử lý tệp CSV dự đoán.');
      }
    };
    reader.readAsText(file);
  };

  const downloadBatchCSV = () => {
    if (!batchResults) return;
    const csvStr = exportToCSV(batchResults.headers, batchResults.rows);
    downloadCSV('ket_qua_du_doan_xgboost.csv', csvStr);
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">
          6. Dự Đoán / Phân Loại Dữ Liệu Bên Ngoài (Inference)
        </h1>
        <p className="text-xs text-slate-500 mt-1">
          Ứng dụng mô hình XGBoost đã tối ưu để phân loại từng cá nhân hoặc dự báo hàng loạt theo lô từ tệp CSV
        </p>
      </div>

      {/* Mode Selector Tabs */}
      <div className="flex items-center gap-2 p-1 bg-slate-100 rounded-xl max-w-md">
        <button
          onClick={() => setActiveMode('single')}
          className={`flex-1 py-2 px-3 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
            activeMode === 'single'
              ? 'bg-white text-slate-900 shadow-xs'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          Nhập liệu thủ công từng mẫu
        </button>
        <button
          onClick={() => setActiveMode('batch')}
          className={`flex-1 py-2 px-3 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
            activeMode === 'batch'
              ? 'bg-white text-slate-900 shadow-xs'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          Tải lên file CSV hàng loạt
        </button>
      </div>

      {/* MODE 1: SINGLE FORM */}
      {activeMode === 'single' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Form */}
          <div className="lg:col-span-6 bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
            <h2 className="text-sm font-bold text-slate-900">
              Nhập Thông Số Đo Lường Lâm Sàng
            </h2>
            <p className="text-xs text-slate-500">
              Nhập các chỉ số xét nghiệm để hệ thống phân tích xác suất mắc bệnh tiểu đường
            </p>

            <form onSubmit={handlePredictSingle} className="space-y-4 pt-2">
              {featureNames.map((feat) => (
                <div key={feat}>
                  <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1">
                    {feat} ({feat.toLowerCase().includes('glucose') ? 'mg/dL' : 'mmHg'})
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    required
                    value={manualInputs[feat] ?? ''}
                    onChange={(e) =>
                      setManualInputs({
                        ...manualInputs,
                        [feat]: parseFloat(e.target.value) || 0,
                      })
                    }
                    className="w-full text-xs font-mono bg-slate-50 border border-slate-200 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500"
                    placeholder={`Ví dụ: ${feat.toLowerCase().includes('glucose') ? '45' : '80'}`}
                  />
                  <span className="text-[11px] text-slate-400 mt-0.5 block">
                    Khoảng dữ liệu huấn luyện: {feat.toLowerCase().includes('glucose') ? '20 - 70' : '50 - 100'}
                  </span>
                </div>
              ))}

              <div className="pt-2">
                <button
                  type="submit"
                  className="w-full inline-flex items-center justify-center gap-2 px-5 py-2.5 text-xs font-bold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors cursor-pointer"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>Dự Đoán Kết Quả Ngay</span>
                </button>
              </div>
            </form>
          </div>

          {/* Single Output Card */}
          <div className="lg:col-span-6 bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
            <div>
              <h2 className="text-sm font-bold text-slate-900 mb-1">
                Kết Quả Phân Loại Từ XGBoost
              </h2>
              <p className="text-xs text-slate-500">
                Suy luận dựa trên tổng hợp cây quyết định và hàm sigmoid
              </p>

              {singlePrediction ? (
                <div className="mt-6 space-y-6">
                  {/* Result banner */}
                  <div
                    className={`p-4 rounded-xl border flex items-center gap-3 ${
                      singlePrediction.pred === 1
                        ? 'bg-rose-50 border-rose-200 text-rose-900'
                        : 'bg-emerald-50 border-emerald-200 text-emerald-900'
                    }`}
                  >
                    {singlePrediction.pred === 1 ? (
                      <AlertTriangle className="w-8 h-8 text-rose-600 flex-shrink-0" />
                    ) : (
                      <CheckCircle className="w-8 h-8 text-emerald-600 flex-shrink-0" />
                    )}
                    <div>
                      <div className="text-base font-bold">
                        {singlePrediction.pred === 1
                          ? 'DƯƠNG TÍNH (Nguy cơ cao - Lớp 1)'
                          : 'ÂM TÍNH (Bình thường - Lớp 0)'}
                      </div>
                      <p className="text-xs opacity-90 mt-0.5">
                        {singlePrediction.pred === 1
                          ? 'Mô hình phát hiện đặc trưng chỉ số tương đồng với nhóm bệnh nhân có nguy cơ tiểu đường.'
                          : 'Mô hình ghi nhận các chỉ số trong ngưỡng an toàn, nguy cơ mắc bệnh thấp.'}
                      </p>
                    </div>
                  </div>

                  {/* Probability Gauge Bar */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
                      <span>Xác suất nguy cơ (Probability of Class 1):</span>
                      <span className="font-mono text-sm text-blue-600">
                        {(singlePrediction.prob * 100).toFixed(2)}%
                      </span>
                    </div>

                    <div className="w-full bg-slate-100 h-4 rounded-full overflow-hidden p-0.5 border border-slate-200">
                      <div
                        className={`h-full rounded-full transition-all duration-700 ${
                          singlePrediction.prob >= 0.5
                            ? 'bg-gradient-to-r from-amber-500 to-rose-600'
                            : 'bg-gradient-to-r from-emerald-400 to-blue-500'
                        }`}
                        style={{ width: `${Math.max(2, singlePrediction.prob * 100)}%` }}
                      ></div>
                    </div>

                    <div className="flex justify-between text-[10px] text-slate-400">
                      <span>0% (An toàn tuyệt đối)</span>
                      <span>50% (Ngưỡng phân loại)</span>
                      <span>100% (Nguy cơ rất cao)</span>
                    </div>
                  </div>

                  {/* Input summary echo */}
                  <div className="bg-slate-50 p-3 rounded-lg text-xs text-slate-600 space-y-1 border border-slate-100">
                    <span className="font-bold text-slate-900 block mb-1">
                      Các thông số đã nhập:
                    </span>
                    {featureNames.map((f) => (
                      <div key={f} className="flex justify-between">
                        <span>{f}:</span>
                        <span className="font-mono font-medium text-slate-900">
                          {manualInputs[f]}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="h-64 flex flex-col items-center justify-center text-center p-6 text-slate-400">
                  <Sparkles className="w-8 h-8 mb-2 opacity-50" />
                  <p className="text-xs">
                    Vui lòng nhập các chỉ số bên cạnh và bấm nút "Dự Đoán Kết Quả Ngay".
                  </p>
                </div>
              )}
            </div>

            <p className="text-[11px] text-slate-400 pt-4 border-t border-slate-100 mt-4">
              Lưu ý: Kết quả dự báo mang tính chất tham khảo học thuật từ mô hình Machine Learning, không thay thế cho chẩn đoán y khoa chuyên nghiệp.
            </p>
          </div>
        </div>
      )}

      {/* MODE 2: BATCH CSV */}
      {activeMode === 'batch' && (
        <div className="space-y-6">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <h2 className="text-sm font-bold text-slate-900">
                  Tải Lên Tệp CSV Cần Phân Loại Hàng Loạt
                </h2>
                <p className="text-xs text-slate-500">
                  Tệp CSV cần chứa các cột: <strong>{featureNames.join(', ')}</strong>
                </p>
              </div>

              <input
                type="file"
                ref={fileInputRef}
                accept=".csv"
                onChange={handleBatchFileUpload}
                className="hidden"
              />

              <button
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors cursor-pointer self-start sm:self-auto"
              >
                <Upload className="w-4 h-4" />
                <span>Chọn tệp CSV từ máy</span>
              </button>
            </div>

            {batchError && (
              <div className="p-3 bg-rose-50 border border-rose-200 text-rose-800 text-xs rounded-lg">
                {batchError}
              </div>
            )}
          </div>

          {/* Results table */}
          {batchResults && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden space-y-4 p-5">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-2 border-b border-slate-100">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-emerald-50 text-emerald-600 rounded-lg">
                    <FileSpreadsheet className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-slate-900">
                      Đã Phân Loại Thành Công {batchResults.total} Bản Ghi
                    </h3>
                    <p className="text-xs text-slate-500">
                      Tỷ lệ dự đoán: {batchResults.positiveCount} Dương tính (
                      {((batchResults.positiveCount / batchResults.total) * 100).toFixed(1)}%) ·{' '}
                      {batchResults.total - batchResults.positiveCount} Âm tính
                    </p>
                  </div>
                </div>

                <button
                  onClick={downloadBatchCSV}
                  className="inline-flex items-center gap-2 px-4 py-2 text-xs font-bold text-white bg-emerald-600 hover:bg-emerald-700 rounded-lg shadow-sm transition-colors cursor-pointer"
                >
                  <Download className="w-4 h-4" />
                  <span>Tải Xuống Kết Quả (CSV)</span>
                </button>
              </div>

              {/* Table */}
              <div className="overflow-x-auto max-h-96">
                <table className="w-full text-left border-collapse text-xs">
                  <thead className="sticky top-0 bg-slate-50 border-b border-slate-200">
                    <tr className="text-slate-600 font-semibold">
                      <th className="py-2.5 px-3">#</th>
                      {batchResults.headers.map((h) => (
                        <th key={h} className="py-2.5 px-3 whitespace-nowrap">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-slate-700">
                    {batchResults.rows.slice(0, 100).map((row, rIdx) => {
                      const predVal = row[row.length - 2];
                      const isPositive = predVal === 1;
                      return (
                        <tr key={rIdx} className="hover:bg-slate-50/70">
                          <td className="py-2 px-3 font-mono text-slate-400 text-[11px]">
                            {rIdx + 1}
                          </td>
                          {row.map((val, cIdx) => {
                            const isConclusion = cIdx === row.length - 1;
                            const isProb = cIdx === row.length - 3;
                            return (
                              <td key={cIdx} className="py-2 px-3 whitespace-nowrap">
                                {isConclusion ? (
                                  <span
                                    className={`font-semibold ${
                                      isPositive ? 'text-rose-600' : 'text-emerald-600'
                                    }`}
                                  >
                                    {String(val)}
                                  </span>
                                ) : isProb ? (
                                  <span className="font-mono font-medium">{val}%</span>
                                ) : (
                                  <span>{String(val)}</span>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {batchResults.rows.length > 100 && (
                <p className="text-[11px] text-slate-400 text-center pt-2">
                  (Đang hiển thị 100 dòng đầu tiên. Hãy tải xuống tệp CSV để xem đầy đủ {batchResults.total} bản ghi.)
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
