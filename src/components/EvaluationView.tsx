import React, { useState, useMemo } from 'react';
import {
  Target,
  BarChart2,
  CheckCircle2,
  Sliders,
  Sparkles,
} from 'lucide-react';
import { TrainedModel } from '../types';
import { evaluateModelOnTestSet } from '../utils/xgboostEngine';

interface EvaluationViewProps {
  model: TrainedModel | null;
  onNavigateToTraining: () => void;
  onNavigateToPrediction: () => void;
}

export const EvaluationView: React.FC<EvaluationViewProps> = ({
  model,
  onNavigateToTraining,
  onNavigateToPrediction,
}) => {
  const [threshold, setThreshold] = useState<number>(0.5);

  // Compute live evaluation based on interactive threshold
  const dynamicEval = useMemo(() => {
    if (!model) return null;
    const { testX, testY, featureNames } = model.preprocessedData;
    const dummyGainMap = new Map<number, number>();
    model.evaluation.featureImportance.forEach((fi) => {
      const idx = featureNames.indexOf(fi.feature);
      if (idx !== -1) dummyGainMap.set(idx, fi.score);
    });

    return evaluateModelOnTestSet(
      testX,
      testY,
      model.baseMargin,
      model.trees,
      model.params.learning_rate,
      featureNames,
      dummyGainMap,
      threshold
    );
  }, [model, threshold]);

  if (!model || !dynamicEval) {
    return (
      <div className="bg-white p-8 rounded-xl border border-slate-200 text-center space-y-4 shadow-sm">
        <div className="w-12 h-12 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center mx-auto">
          <Target className="w-6 h-6" />
        </div>
        <h2 className="text-base font-bold text-slate-900">
          Chưa Có Mô Hình Nào Được Huấn Luyện
        </h2>
        <p className="text-xs text-slate-500 max-w-md mx-auto">
          Vui lòng bấm huấn luyện mô hình XGBoost ở tab '4. Huấn luyện XGBoost' để kích hoạt trang đánh giá toàn diện.
        </p>
        <button
          onClick={onNavigateToTraining}
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors cursor-pointer"
        >
          <span>Đến tab Huấn luyện</span>
        </button>
      </div>
    );
  }

  const { accuracy, precision, recall, f1Score, rocAuc, confusionMatrix, rocCurve, prCurve, featureImportance, classReport } = dynamicEval;
  const { tn, fp, fn, tp } = confusionMatrix;
  const totalTest = tn + fp + fn + tp;

  return (
    <div className="space-y-6">
      {/* Title & Quick Action */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">
            5. Đánh Giá Kết Quả Phân Loại Trên Tập Kiểm Thử (Test Set)
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Đánh giá hiệu suất tổng thể của mô hình XGBoost với N = {totalTest} mẫu kiểm nghiệm độc lập
          </p>
        </div>

        <button
          onClick={onNavigateToPrediction}
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg shadow-sm transition-colors cursor-pointer self-start sm:self-auto"
        >
          <Sparkles className="w-4 h-4" />
          <span>Dự đoán dữ liệu mới</span>
        </button>
      </div>

      {/* 5 Core Metric Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <span className="text-xs font-medium text-slate-500 block mb-1">
            Độ chính xác (Accuracy)
          </span>
          <div className="text-2xl font-bold text-blue-600">
            {(accuracy * 100).toFixed(2)}%
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            {tp + tn} / {totalTest} dự đoán đúng
          </p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <span className="text-xs font-medium text-slate-500 block mb-1">
            Độ chuẩn xác (Precision)
          </span>
          <div className="text-2xl font-bold text-emerald-600">
            {(precision * 100).toFixed(2)}%
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            TP / (TP + FP)
          </p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <span className="text-xs font-medium text-slate-500 block mb-1">
            Độ bao phủ (Recall / Sensitivity)
          </span>
          <div className="text-2xl font-bold text-indigo-600">
            {(recall * 100).toFixed(2)}%
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            TP / (TP + FN)
          </p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <span className="text-xs font-medium text-slate-500 block mb-1">
            F1-Score
          </span>
          <div className="text-2xl font-bold text-purple-600">
            {(f1Score * 100).toFixed(2)}%
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Trung bình điều hòa P & R
          </p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm col-span-2 sm:col-span-1">
          <span className="text-xs font-medium text-slate-500 block mb-1">
            ROC-AUC Score
          </span>
          <div className="text-2xl font-bold text-rose-600">
            {rocAuc.toFixed(4)}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Khả năng phân biệt 2 lớp
          </p>
        </div>
      </div>

      {/* Interactive Classification Threshold Slider */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-50 text-blue-600 rounded-lg">
            <Sliders className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs font-bold text-slate-900 block">
              Điều Chỉnh Ngưỡng Quyết Định Phân Loại (Decision Threshold)
            </span>
            <p className="text-xs text-slate-500">
              Kéo để xem sự thay đổi thời gian thực giữa Precision và Recall trên tập Test
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 w-full sm:w-72">
          <span className="text-xs font-mono text-slate-400">0.1</span>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.02"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full accent-blue-600 cursor-pointer"
          />
          <span className="text-xs font-mono font-bold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
            {threshold.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Row: Confusion Matrix + Feature Importance */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Confusion Matrix */}
        <div className="lg:col-span-6 bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-bold text-slate-900">
                Ma Trận Nhầm Lẫn (Confusion Matrix)
              </h2>
              <p className="text-xs text-slate-500">
                So sánh giữa kết quả dự đoán của XGBoost và nhãn thực tế
              </p>
            </div>
            <span className="text-xs font-mono text-slate-500 bg-slate-50 px-2 py-1 rounded border border-slate-200">
              Ngưỡng: &ge; {threshold.toFixed(2)}
            </span>
          </div>

          {/* Matrix Grid */}
          <div className="max-w-md mx-auto pt-2">
            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              <div></div>
              <div className="font-bold text-slate-700 pb-1">Dự đoán Âm (0)</div>
              <div className="font-bold text-slate-700 pb-1">Dự đoán Dương (1)</div>

              <div className="font-bold text-slate-700 flex items-center justify-end pr-2">
                Thực tế Âm (0)
              </div>
              {/* True Negative */}
              <div className="p-4 rounded-xl bg-blue-50 border border-blue-200 flex flex-col items-center justify-center">
                <span className="text-xl font-bold text-blue-900 font-mono">{tn}</span>
                <span className="text-[11px] font-semibold text-blue-700 mt-1">
                  Đúng Âm (TN)
                </span>
                <span className="text-[10px] text-blue-600/70">
                  {((tn / (tn + fp || 1)) * 100).toFixed(1)}% lớp 0
                </span>
              </div>

              {/* False Positive */}
              <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 flex flex-col items-center justify-center">
                <span className="text-xl font-bold text-rose-900 font-mono">{fp}</span>
                <span className="text-[11px] font-semibold text-rose-700 mt-1">
                  Dương giả (FP - Loại I)
                </span>
                <span className="text-[10px] text-rose-600/70">
                  Báo động nhầm
                </span>
              </div>

              <div className="font-bold text-slate-700 flex items-center justify-end pr-2">
                Thực tế Dương (1)
              </div>
              {/* False Negative */}
              <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 flex flex-col items-center justify-center">
                <span className="text-xl font-bold text-amber-900 font-mono">{fn}</span>
                <span className="text-[11px] font-semibold text-amber-700 mt-1">
                  Âm giả (FN - Loại II)
                </span>
                <span className="text-[10px] text-amber-600/70">
                  Bỏ sót bệnh
                </span>
              </div>

              {/* True Positive */}
              <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 flex flex-col items-center justify-center">
                <span className="text-xl font-bold text-emerald-900 font-mono">{tp}</span>
                <span className="text-[11px] font-semibold text-emerald-700 mt-1">
                  Đúng Dương (TP)
                </span>
                <span className="text-[10px] text-emerald-600/70">
                  {((tp / (tp + fn || 1)) * 100).toFixed(1)}% lớp 1
                </span>
              </div>
            </div>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 text-xs text-slate-600 flex items-center justify-between">
            <span>Tổng số dự đoán chính xác:</span>
            <span className="font-bold text-slate-900">
              {tp + tn} / {totalTest} ({((tp + tn) / totalTest * 100).toFixed(1)}%)
            </span>
          </div>
        </div>

        {/* Feature Importance */}
        <div className="lg:col-span-6 bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-indigo-600" />
            <div>
              <h2 className="text-sm font-bold text-slate-900">
                Độ Quan Trọng Của Đặc Trưng (Feature Importance - Gain)
              </h2>
              <p className="text-xs text-slate-500">
                Đóng góp mức tăng thông tin (Gain) của từng biến vào việc giảm log-loss
              </p>
            </div>
          </div>

          <div className="space-y-4 pt-2">
            {featureImportance.map((fi, i) => (
              <div key={fi.feature} className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded bg-slate-100 text-slate-700 font-bold flex items-center justify-center text-[10px]">
                      #{i + 1}
                    </span>
                    <span className="font-bold text-slate-800">{fi.feature}</span>
                  </div>
                  <div className="flex items-center gap-2 font-mono">
                    <span className="text-slate-500">Gain: {fi.score.toFixed(2)}</span>
                    <span className="font-bold text-indigo-600">{fi.percentage}%</span>
                  </div>
                </div>

                <div className="w-full bg-slate-100 h-3 rounded-full overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-indigo-600 h-full rounded-full transition-all duration-500"
                    style={{ width: `${Math.max(5, fi.percentage)}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 p-3 bg-slate-50 rounded-lg text-xs text-slate-600 border border-slate-100">
            <span className="font-semibold text-slate-900">Giải thích đặc trưng:</span> Trong tập dữ liệu Naive Bayes, biến <strong>{featureImportance[0]?.feature || 'glucose'}</strong> đóng vai trò then chốt nhất trong các quyết định phân nhánh của cây XGBoost.
          </div>
        </div>
      </div>

      {/* Row: ROC Curve & PR Curve */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ROC Curve */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-bold text-slate-900">
                Đường Cong ROC (Receiver Operating Characteristic)
              </h2>
              <p className="text-xs text-slate-500">
                Tỷ lệ dương tính thật (TPR) vs Tỷ lệ dương tính giả (FPR)
              </p>
            </div>
            <span className="text-xs font-bold text-rose-600 bg-rose-50 px-2 py-0.5 rounded border border-rose-200">
              AUC = {rocAuc.toFixed(3)}
            </span>
          </div>

          {/* SVG ROC Plot */}
          <div className="h-56">
            <svg viewBox="0 0 300 200" className="w-full h-full bg-slate-50 rounded-lg border border-slate-100 p-2">
              <line x1="35" y1="20" x2="35" y2="175" stroke="#cbd5e1" />
              <line x1="35" y1="175" x2="280" y2="175" stroke="#cbd5e1" />

              {/* Diagonal baseline */}
              <line x1="35" y1="175" x2="280" y2="20" stroke="#94a3b8" strokeDasharray="3 3" />

              {/* ROC Curve line */}
              {(() => {
                const points = rocCurve
                  .map((p) => {
                    const x = 35 + p.fpr * 245;
                    const y = 175 - p.tpr * 155;
                    return `${x},${y}`;
                  })
                  .join(' ');

                return (
                  <polyline
                    fill="none"
                    stroke="#2563eb"
                    strokeWidth="2.5"
                    points={points}
                  />
                );
              })()}

              <text x="157" y="193" textAnchor="middle" className="text-[10px] fill-slate-500">
                False Positive Rate (1 - Specificity)
              </text>
              <text x="14" y="97" textAnchor="middle" transform="rotate(-90 14,97)" className="text-[10px] fill-slate-500">
                True Positive Rate
              </text>
            </svg>
          </div>
        </div>

        {/* PR Curve */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-bold text-slate-900">
                Đường Cong Precision - Recall
              </h2>
              <p className="text-xs text-slate-500">
                Độ chuẩn xác theo độ bao phủ trên mọi ngưỡng
              </p>
            </div>
            <span className="text-xs font-bold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
              F1 = {(f1Score * 100).toFixed(1)}%
            </span>
          </div>

          {/* SVG PR Plot */}
          <div className="h-56">
            <svg viewBox="0 0 300 200" className="w-full h-full bg-slate-50 rounded-lg border border-slate-100 p-2">
              <line x1="35" y1="20" x2="35" y2="175" stroke="#cbd5e1" />
              <line x1="35" y1="175" x2="280" y2="175" stroke="#cbd5e1" />

              {/* PR Curve line */}
              {(() => {
                const points = prCurve
                  .map((p) => {
                    const x = 35 + p.recall * 245;
                    const y = 175 - p.precision * 155;
                    return `${x},${y}`;
                  })
                  .join(' ');

                return (
                  <polyline
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="2.5"
                    points={points}
                  />
                );
              })()}

              <text x="157" y="193" textAnchor="middle" className="text-[10px] fill-slate-500">
                Recall (Độ bao phủ)
              </text>
              <text x="14" y="97" textAnchor="middle" transform="rotate(-90 14,97)" className="text-[10px] fill-slate-500">
                Precision (Độ chuẩn xác)
              </text>
            </svg>
          </div>
        </div>
      </div>

      {/* Detailed Classification Report Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-blue-600" />
            <h2 className="text-sm font-bold text-slate-900">
              Báo Cáo Phân Loại Chi Tiết (Classification Report)
            </h2>
          </div>
          <span className="text-xs text-slate-500">
            Hỗ trợ đánh giá cân bằng Precision, Recall & F1 từng lớp
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold">
                <th className="py-2.5 px-4">Lớp nhãn (Class)</th>
                <th className="py-2.5 px-4 text-right">Precision</th>
                <th className="py-2.5 px-4 text-right">Recall</th>
                <th className="py-2.5 px-4 text-right">F1-Score</th>
                <th className="py-2.5 px-4 text-right">Số mẫu (Support)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              <tr>
                <td className="py-2.5 px-4 font-semibold text-slate-900">
                  Lớp 0 (Âm tính / Bình thường)
                </td>
                <td className="py-2.5 px-4 text-right font-mono">
                  {classReport.class0.precision.toFixed(3)}
                </td>
                <td className="py-2.5 px-4 text-right font-mono">
                  {classReport.class0.recall.toFixed(3)}
                </td>
                <td className="py-2.5 px-4 text-right font-mono font-medium text-blue-600">
                  {classReport.class0.f1.toFixed(3)}
                </td>
                <td className="py-2.5 px-4 text-right font-mono text-slate-500">
                  {classReport.class0.support}
                </td>
              </tr>
              <tr>
                <td className="py-2.5 px-4 font-semibold text-slate-900">
                  Lớp 1 (Dương tính / Nguy cơ cao)
                </td>
                <td className="py-2.5 px-4 text-right font-mono">
                  {classReport.class1.precision.toFixed(3)}
                </td>
                <td className="py-2.5 px-4 text-right font-mono">
                  {classReport.class1.recall.toFixed(3)}
                </td>
                <td className="py-2.5 px-4 text-right font-mono font-medium text-emerald-600">
                  {classReport.class1.f1.toFixed(3)}
                </td>
                <td className="py-2.5 px-4 text-right font-mono text-slate-500">
                  {classReport.class1.support}
                </td>
              </tr>
              <tr className="bg-slate-50/70 font-semibold text-slate-800 border-t border-slate-200">
                <td className="py-2.5 px-4">Macro Average (Trung bình giản đơn)</td>
                <td className="py-2.5 px-4 text-right font-mono">
                  {classReport.macroAvg.precision.toFixed(3)}
                </td>
                <td className="py-2.5 px-4 text-right font-mono">
                  {classReport.macroAvg.recall.toFixed(3)}
                </td>
                <td className="py-2.5 px-4 text-right font-mono">
                  {classReport.macroAvg.f1.toFixed(3)}
                </td>
                <td className="py-2.5 px-4 text-right font-mono">{totalTest}</td>
              </tr>
              <tr className="bg-slate-50/70 font-semibold text-slate-800">
                <td className="py-2.5 px-4">Weighted Average (Trung bình gia quyền)</td>
                <td className="py-2.5 px-4 text-right font-mono">
                  {classReport.weightedAvg.precision.toFixed(3)}
                </td>
                <td className="py-2.5 px-4 text-right font-mono">
                  {classReport.weightedAvg.recall.toFixed(3)}
                </td>
                <td className="py-2.5 px-4 text-right font-mono">
                  {classReport.weightedAvg.f1.toFixed(3)}
                </td>
                <td className="py-2.5 px-4 text-right font-mono">{totalTest}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
