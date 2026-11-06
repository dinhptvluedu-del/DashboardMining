import React from 'react';
import {
  History,
  Download,
  Award,
  ArrowRight,
  TrendingUp,
} from 'lucide-react';
import { TrainingHistoryEntry, XGBoostParams } from '../types';
import { exportToCSV, downloadCSV } from '../utils/csvParser';

interface HistoryLogViewProps {
  history: TrainingHistoryEntry[];
  onApplyParams: (params: XGBoostParams) => void;
  onNavigateToTraining: () => void;
}

export const HistoryLogView: React.FC<HistoryLogViewProps> = ({
  history,
  onApplyParams,
  onNavigateToTraining,
}) => {
  const downloadHistoryCSV = () => {
    if (history.length === 0) return;
    const headers = [
      'ID',
      'Thời gian',
      'Tập dữ liệu',
      'Số cây (n_estimators)',
      'Learning Rate',
      'Max Depth',
      'Subsample',
      'Gamma',
      'Accuracy (%)',
      'Precision (%)',
      'Recall (%)',
      'F1-Score (%)',
      'ROC-AUC',
      'Thời gian huấn luyện (ms)',
    ];

    const rows = history.map((h) => [
      h.id,
      h.timestamp,
      h.datasetName,
      h.params.n_estimators,
      h.params.learning_rate,
      h.params.max_depth,
      h.params.subsample,
      h.params.gamma,
      (h.accuracy * 100).toFixed(2),
      (h.precision * 100).toFixed(2),
      (h.recall * 100).toFixed(2),
      (h.f1Score * 100).toFixed(2),
      h.rocAuc.toFixed(4),
      h.durationMs,
    ]);

    const csvContent = exportToCSV(headers, rows);
    downloadCSV('lich_su_huan_luyen_xgboost.csv', csvContent);
  };

  // Find best model by Accuracy
  const bestEntry = history.length > 0
    ? [...history].sort((a, b) => b.accuracy - a.accuracy)[0]
    : null;

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">
            7. Lịch Sử Thực Hiện & Thử Nghiệm Mô Hình (Experiment Logging)
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Ghi nhận tự động các lần huấn luyện, so sánh độ chính xác và khôi phục bộ siêu tham số tối ưu
          </p>
        </div>

        {history.length > 0 && (
          <button
            onClick={downloadHistoryCSV}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg shadow-xs transition-colors cursor-pointer self-start sm:self-auto"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" />
            <span>Xuất Lịch Sử (CSV)</span>
          </button>
        )}
      </div>

      {/* Best Model Banner */}
      {bestEntry && (
        <div className="bg-gradient-to-r from-blue-600 to-indigo-700 rounded-xl p-5 text-white shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-white/10 flex items-center justify-center text-amber-300">
              <Award className="w-7 h-7" />
            </div>
            <div>
              <span className="text-[11px] font-semibold uppercase tracking-wider text-blue-200">
                Mô hình có hiệu suất cao nhất
              </span>
              <div className="text-lg font-bold">
                Accuracy: {(bestEntry.accuracy * 100).toFixed(2)}% · ROC-AUC: {bestEntry.rocAuc.toFixed(4)}
              </div>
              <p className="text-xs text-blue-100 mt-0.5">
                Cấu hình: n_estimators={bestEntry.params.n_estimators}, lr={bestEntry.params.learning_rate}, depth={bestEntry.params.max_depth} · ({bestEntry.timestamp})
              </p>
            </div>
          </div>

          <button
            onClick={() => {
              onApplyParams(bestEntry.params);
              onNavigateToTraining();
            }}
            className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-bold text-slate-900 bg-white hover:bg-slate-100 rounded-lg transition-colors cursor-pointer flex-shrink-0"
          >
            <span>Áp dụng cấu hình này</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* History Table */}
      {history.length === 0 ? (
        <div className="bg-white p-8 rounded-xl border border-slate-200 text-center space-y-4 shadow-sm">
          <div className="w-12 h-12 rounded-full bg-slate-100 text-slate-400 flex items-center justify-center mx-auto">
            <History className="w-6 h-6" />
          </div>
          <h2 className="text-base font-bold text-slate-900">
            Chưa Có Lần Huấn Luyện Nào Được Lưu
          </h2>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Mỗi khi bạn bấm 'Huấn luyện mô hình', các tham số và chỉ số đánh giá sẽ tự động được ghi lại tại đây để tiện theo dõi và đối sánh.
          </p>
          <button
            onClick={onNavigateToTraining}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors cursor-pointer"
          >
            <span>Huấn luyện mô hình đầu tiên</span>
          </button>
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="p-4 border-b border-slate-200 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-blue-600" />
              <h2 className="text-sm font-bold text-slate-900">
                Nhật Ký Các Lần Thử Nghiệm (Tổng cộng: {history.length} lần)
              </h2>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold">
                  <th className="py-2.5 px-3">#</th>
                  <th className="py-2.5 px-3">Thời gian</th>
                  <th className="py-2.5 px-3">Siêu tham số (n, lr, depth)</th>
                  <th className="py-2.5 px-3 text-right">Accuracy</th>
                  <th className="py-2.5 px-3 text-right">Precision</th>
                  <th className="py-2.5 px-3 text-right">Recall</th>
                  <th className="py-2.5 px-3 text-right">F1-Score</th>
                  <th className="py-2.5 px-3 text-right">ROC-AUC</th>
                  <th className="py-2.5 px-3 text-right">Thời gian</th>
                  <th className="py-2.5 px-3 text-center">Thao tác</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {history.map((entry, idx) => {
                  const isBest = bestEntry?.id === entry.id;
                  return (
                    <tr
                      key={entry.id}
                      className={`hover:bg-slate-50/70 transition-colors ${
                        isBest ? 'bg-amber-50/30' : ''
                      }`}
                    >
                      <td className="py-3 px-3 font-mono text-slate-400">
                        {history.length - idx}
                      </td>
                      <td className="py-3 px-3 font-medium text-slate-900 whitespace-nowrap">
                        <div className="flex items-center gap-1.5">
                          {isBest && (
                            <Award className="w-3.5 h-3.5 text-amber-500 flex-shrink-0" />
                          )}
                          <span>{entry.timestamp}</span>
                        </div>
                      </td>
                      <td className="py-3 px-3 font-mono text-[11px] text-slate-600">
                        n={entry.params.n_estimators}, lr={entry.params.learning_rate}, depth={entry.params.max_depth}
                      </td>
                      <td className="py-3 px-3 text-right font-mono font-bold text-blue-600">
                        {(entry.accuracy * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3 text-right font-mono">
                        {(entry.precision * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3 text-right font-mono">
                        {(entry.recall * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3 text-right font-mono font-medium">
                        {(entry.f1Score * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3 text-right font-mono text-rose-600 font-semibold">
                        {entry.rocAuc.toFixed(4)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono text-slate-400">
                        {entry.durationMs}ms
                      </td>
                      <td className="py-3 px-3 text-center">
                        <button
                          onClick={() => {
                            onApplyParams(entry.params);
                            onNavigateToTraining();
                          }}
                          className="px-2.5 py-1 text-[11px] font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 rounded border border-blue-200 transition-colors cursor-pointer"
                        >
                          Khôi phục
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
