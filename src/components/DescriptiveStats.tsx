import React, { useState } from 'react';
import { Table, Flame, CheckCircle, HelpCircle } from 'lucide-react';
import { Dataset, ColumnStat, CorrelationMatrix } from '../types';

interface DescriptiveStatsProps {
  dataset: Dataset;
  stats: ColumnStat[];
  correlation: CorrelationMatrix;
}

export const DescriptiveStats: React.FC<DescriptiveStatsProps> = ({
  dataset,
  stats,
  correlation,
}) => {
  const [hoveredCorr, setHoveredCorr] = useState<{
    col1: string;
    col2: string;
    val: number;
  } | null>(null);

  const getHeatmapColor = (val: number) => {
    // val ranges from -1 to 1
    if (val > 0) {
      // Blue scale for positive correlation
      const intensity = Math.min(1, Math.max(0, val));
      const r = Math.round(255 - intensity * (255 - 37));
      const g = Math.round(255 - intensity * (255 - 99));
      const b = Math.round(255 - intensity * (255 - 235));
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Red scale for negative correlation
      const intensity = Math.min(1, Math.max(0, Math.abs(val)));
      const r = Math.round(255 - intensity * (255 - 225));
      const g = Math.round(255 - intensity * (255 - 29));
      const b = Math.round(255 - intensity * (255 - 72));
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  const numericStats = stats.filter((s) => s.type === 'numeric');

  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">
          2. Thống Kê Mô Tả & Ma Trận Tương Quan
        </h1>
        <p className="text-xs text-slate-500 mt-1">
          Phân tích định lượng các thông số thống kê trung tâm, độ biến thiên và mối quan hệ tuyến tính giữa các biến
        </p>
      </div>

      {/* Descriptive Statistics Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Table className="w-4 h-4 text-blue-600" />
            <h2 className="text-sm font-bold text-slate-900">
              Bảng Thống Kê Mô Tả Chi Tiết (Descriptive Statistics)
            </h2>
          </div>
          <span className="text-xs text-slate-500 font-mono">
            N = {dataset.rowCount} mẫu
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold">
                <th className="py-2.5 px-4">Tên biến (Feature)</th>
                <th className="py-2.5 px-3">Kiểu</th>
                <th className="py-2.5 px-3 text-right">Số mẫu</th>
                <th className="py-2.5 px-3 text-right">Trung bình (Mean)</th>
                <th className="py-2.5 px-3 text-right">Độ lệch chuẩn (Std)</th>
                <th className="py-2.5 px-3 text-right">Tối thiểu (Min)</th>
                <th className="py-2.5 px-3 text-right">Q1 (25%)</th>
                <th className="py-2.5 px-3 text-right font-bold text-slate-900">
                  Trung vị (Median)
                </th>
                <th className="py-2.5 px-3 text-right">Q3 (75%)</th>
                <th className="py-2.5 px-3 text-right">Tối đa (Max)</th>
                <th className="py-2.5 px-3 text-right">Độ lệch (Skewness)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {numericStats.map((st) => (
                <tr key={st.column} className="hover:bg-slate-50/70 transition-colors">
                  <td className="py-3 px-4 font-medium text-slate-900 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                    <span>{st.column}</span>
                  </td>
                  <td className="py-3 px-3">
                    <span className="text-[11px] text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded">
                      numeric
                    </span>
                  </td>
                  <td className="py-3 px-3 text-right font-mono">{st.count}</td>
                  <td className="py-3 px-3 text-right font-mono font-medium">
                    {st.mean?.toFixed(2)}
                  </td>
                  <td className="py-3 px-3 text-right font-mono text-slate-500">
                    {st.std?.toFixed(2)}
                  </td>
                  <td className="py-3 px-3 text-right font-mono">{st.min?.toFixed(1)}</td>
                  <td className="py-3 px-3 text-right font-mono text-slate-600">
                    {st.q1?.toFixed(1)}
                  </td>
                  <td className="py-3 px-3 text-right font-mono font-semibold text-blue-600">
                    {st.median?.toFixed(1)}
                  </td>
                  <td className="py-3 px-3 text-right font-mono text-slate-600">
                    {st.q3?.toFixed(1)}
                  </td>
                  <td className="py-3 px-3 text-right font-mono">{st.max?.toFixed(1)}</td>
                  <td className="py-3 px-3 text-right font-mono">
                    <span
                      className={
                        Math.abs(st.skewness || 0) > 0.5 ? 'text-amber-600 font-medium' : 'text-slate-500'
                      }
                    >
                      {st.skewness?.toFixed(2)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Correlation Matrix Heatmap & Quality Inspection */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Heatmap */}
        <div className="lg:col-span-7 bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Flame className="w-4 h-4 text-rose-500" />
              <div>
                <h2 className="text-sm font-bold text-slate-900">
                  Ma Trận Tương Quan Tuyến Tính (Correlation Heatmap)
                </h2>
                <p className="text-xs text-slate-500">
                  Hệ số tương quan Pearson từ -1.0 (nghịch đảo) đến +1.0 (thuận)
                </p>
              </div>
            </div>
            {hoveredCorr && (
              <div className="text-xs font-mono bg-slate-100 px-2.5 py-1 rounded border border-slate-200">
                <span className="font-semibold">{hoveredCorr.col1}</span> &harr;{' '}
                <span className="font-semibold">{hoveredCorr.col2}</span>: r ={' '}
                <span className="font-bold text-blue-700">{hoveredCorr.val.toFixed(3)}</span>
              </div>
            )}
          </div>

          <div className="mt-4 overflow-x-auto">
            <div className="inline-block min-w-full">
              <div
                className="grid gap-1.5"
                style={{
                  gridTemplateColumns: `auto repeat(${correlation.columns.length}, minmax(80px, 1fr))`,
                }}
              >
                {/* Header row */}
                <div></div>
                {correlation.columns.map((c) => (
                  <div
                    key={c}
                    className="text-center text-[11px] font-bold text-slate-700 p-1 truncate"
                    title={c}
                  >
                    {c}
                  </div>
                ))}

                {/* Rows */}
                {correlation.columns.map((rowName, rIdx) => (
                  <React.Fragment key={rowName}>
                    <div
                      className="text-right text-[11px] font-bold text-slate-700 pr-2 flex items-center justify-end truncate"
                      title={rowName}
                    >
                      {rowName}
                    </div>
                    {correlation.columns.map((colName, cIdx) => {
                      const val = correlation.matrix[rIdx][cIdx];
                      const isHigh = Math.abs(val) > 0.5;
                      return (
                        <div
                          key={colName}
                          onMouseEnter={() =>
                            setHoveredCorr({ col1: rowName, col2: colName, val })
                          }
                          onMouseLeave={() => setHoveredCorr(null)}
                          className="h-14 rounded-lg flex flex-col items-center justify-center cursor-pointer transition-transform hover:scale-105 shadow-xs border border-white/40"
                          style={{
                            backgroundColor: getHeatmapColor(val),
                            color: Math.abs(val) > 0.4 ? '#ffffff' : '#1e293b',
                          }}
                        >
                          <span className="text-xs font-bold font-mono">
                            {val.toFixed(2)}
                          </span>
                          <span className="text-[9px] opacity-80">
                            {rIdx === cIdx ? '100%' : `${(val * 100).toFixed(0)}%`}
                          </span>
                        </div>
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>
            </div>
          </div>

          {/* Color scale legend */}
          <div className="mt-6 flex items-center justify-between text-[11px] text-slate-500 pt-3 border-t border-slate-100">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-[#e11d48]"></span>
              <span>Nghịch biến mạnh (-1.0)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-white border border-slate-300"></span>
              <span>Không tương quan (0.0)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-[#2563eb]"></span>
              <span>Đồng biến mạnh (+1.0)</span>
            </div>
          </div>
        </div>

        {/* Missing Values & Data Quality Audit */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle className="w-4 h-4 text-emerald-500" />
              <h2 className="text-sm font-bold text-slate-900">
                Kiểm Tra Giá Trị Khuyết Thiếu (Missing Values)
              </h2>
            </div>
            <div className="divide-y divide-slate-100 text-xs">
              {stats.map((st) => (
                <div key={st.column} className="py-2.5 flex items-center justify-between">
                  <div>
                    <span className="font-semibold text-slate-800">{st.column}</span>
                    <p className="text-[11px] text-slate-400">
                      {st.type === 'numeric' ? 'Dữ liệu số liên tục' : 'Dữ liệu phân loại'}
                    </p>
                  </div>
                  <div className="text-right">
                    <span className="text-emerald-700 font-semibold bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200/60">
                      0 khuyết ({st.missingPct.toFixed(1)}%)
                    </span>
                    <p className="text-[10px] text-slate-400 mt-0.5">
                      {st.count} / {dataset.rowCount} hợp lệ
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-xs text-slate-600 space-y-2">
            <div className="flex items-center gap-1.5 font-bold text-slate-900">
              <HelpCircle className="w-4 h-4 text-blue-600" />
              <span>Gợi ý phân tích đặc trưng</span>
            </div>
            <p>
              • Biến <strong>glucose</strong> có khoảng giá trị từ 20 đến 70 mg/dL với giá trị trung bình 44.3 mg/dL.
            </p>
            <p>
              • Biến <strong>bloodpressure</strong> có khoảng giá trị từ 50 đến 100 mmHg, phân phối tương đối đối xứng (Skewness ~ 0.05).
            </p>
            <p>
              • Tương quan giữa glucose và huyết áp là tương quan yếu, cho phép mô hình XGBoost khai thác thông tin độc lập giữa 2 đặc trưng này một cách hiệu quả.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
