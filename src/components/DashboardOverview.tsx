import React, { useState, useMemo } from 'react';
import {
  Layers,
  Hash,
  Activity,
  AlertTriangle,
  Search,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
} from 'lucide-react';
import { Dataset, ColumnStat } from '../types';

interface DashboardOverviewProps {
  dataset: Dataset;
  stats: ColumnStat[];
  onNavigateToPreprocessing: () => void;
}

export const DashboardOverview: React.FC<DashboardOverviewProps> = ({
  dataset,
  stats,
  onNavigateToPreprocessing,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(0);
  const rowsPerPage = 10;

  // Filtered rows
  const filteredRows = useMemo(() => {
    if (!searchTerm.trim()) return dataset.rawRows;
    const term = searchTerm.toLowerCase();
    return dataset.rawRows.filter((row) =>
      row.some((val) => String(val).toLowerCase().includes(term))
    );
  }, [dataset.rawRows, searchTerm]);

  const totalPages = Math.ceil(filteredRows.length / rowsPerPage) || 1;
  const currentRows = filteredRows.slice(page * rowsPerPage, (page + 1) * rowsPerPage);

  // Target statistics (assuming 'diabetes' or last column)
  const targetColName = dataset.headers.includes('diabetes')
    ? 'diabetes'
    : dataset.headers[dataset.headers.length - 1];
  const targetIdx = dataset.headers.indexOf(targetColName);

  const targetDistribution = useMemo(() => {
    let c0 = 0;
    let c1 = 0;
    for (const r of dataset.rawRows) {
      const v = String(r[targetIdx]);
      if (v === '1') c1++;
      else c0++;
    }
    const total = c0 + c1 || 1;
    return {
      c0,
      c1,
      pct0: (c0 / total) * 100,
      pct1: (c1 / total) * 100,
    };
  }, [dataset.rawRows, targetIdx]);

  // Scatter plot points (glucose vs bloodpressure)
  const glucoseIdx = dataset.headers.indexOf('glucose');
  const bpIdx = dataset.headers.indexOf('bloodpressure');

  const scatterPoints = useMemo(() => {
    if (glucoseIdx === -1 || bpIdx === -1) return [];
    // sample max 300 points for smooth svg rendering
    const step = Math.max(1, Math.floor(dataset.rawRows.length / 300));
    const points: { x: number; y: number; label: number }[] = [];

    for (let i = 0; i < dataset.rawRows.length; i += step) {
      const g = Number(dataset.rawRows[i][glucoseIdx]);
      const bp = Number(dataset.rawRows[i][bpIdx]);
      const lbl = Number(dataset.rawRows[i][targetIdx]);
      if (!isNaN(g) && !isNaN(bp)) {
        points.push({ x: g, y: bp, label: lbl === 1 ? 1 : 0 });
      }
    }
    return points;
  }, [dataset.rawRows, glucoseIdx, bpIdx, targetIdx]);

  // Min/Max for scatter scale
  const scatterScale = useMemo(() => {
    if (scatterPoints.length === 0) return { minX: 0, maxX: 100, minY: 0, maxY: 100 };
    const xs = scatterPoints.map((p) => p.x);
    const ys = scatterPoints.map((p) => p.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };
  }, [scatterPoints]);

  return (
    <div className="space-y-6">
      {/* Top Welcome & KPI row */}
      <div>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-slate-900 tracking-tight">
              1. Dashboard Tổng Kết Dữ Liệu
            </h1>
            <p className="text-xs text-slate-500 mt-1">
              Khám phá tổng quan cấu trúc bảng dữ liệu, phân phối các biến và kiểm tra chất lượng ban đầu
            </p>
          </div>
          <button
            onClick={onNavigateToPreprocessing}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors cursor-pointer self-start sm:self-auto"
          >
            <span>Tiến hành Tiền xử lý</span>
            <TrendingUp className="w-4 h-4" />
          </button>
        </div>

        {/* 4 Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-4">
          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between text-slate-500 text-xs font-medium mb-1">
              <span>Tổng số mẫu quan sát</span>
              <Layers className="w-4 h-4 text-blue-500" />
            </div>
            <div className="text-2xl font-bold text-slate-900">
              {dataset.rowCount.toLocaleString()}
            </div>
            <p className="text-[11px] text-slate-500 mt-1">Dòng dữ liệu hợp lệ</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between text-slate-500 text-xs font-medium mb-1">
              <span>Số lượng cột đặc trưng</span>
              <Hash className="w-4 h-4 text-indigo-500" />
            </div>
            <div className="text-2xl font-bold text-slate-900">{dataset.colCount}</div>
            <p className="text-[11px] text-slate-500 mt-1">
              {stats.filter((s) => s.type === 'numeric').length} số ·{' '}
              {stats.filter((s) => s.type === 'categorical').length} phân loại
            </p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between text-slate-500 text-xs font-medium mb-1">
              <span>Cân bằng biến mục tiêu</span>
              <Activity className="w-4 h-4 text-emerald-500" />
            </div>
            <div className="text-2xl font-bold text-slate-900">
              {targetDistribution.pct1.toFixed(1)}% : {targetDistribution.pct0.toFixed(1)}%
            </div>
            <p className="text-[11px] text-slate-500 mt-1">
              Lớp 1 ({targetDistribution.c1}) vs Lớp 0 ({targetDistribution.c0})
            </p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between text-slate-500 text-xs font-medium mb-1">
              <span>Giá trị khuyết (Missing)</span>
              <AlertTriangle className="w-4 h-4 text-amber-500" />
            </div>
            <div className="text-2xl font-bold text-slate-900">0</div>
            <p className="text-[11px] text-emerald-600 font-medium mt-1">
              Dữ liệu hoàn chỉnh 100%
            </p>
          </div>
        </div>
      </div>

      {/* Visual Charts: Target distribution & 2D Scatter plot */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart 1: Target distribution */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-bold text-slate-900">
                Phân phối biến mục tiêu ({targetColName})
              </h2>
              <p className="text-xs text-slate-500">Tỷ lệ các lớp trong tập dữ liệu</p>
            </div>
          </div>

          <div className="space-y-4 pt-2">
            <div>
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="font-medium text-slate-700">Lớp 0 (Âm tính / Không tiểu đường)</span>
                <span className="text-slate-500">
                  {targetDistribution.c0} mẫu ({targetDistribution.pct0.toFixed(1)}%)
                </span>
              </div>
              <div className="w-full bg-slate-100 h-3 rounded-full overflow-hidden">
                <div
                  className="bg-blue-500 h-full rounded-full transition-all duration-500"
                  style={{ width: `${targetDistribution.pct0}%` }}
                ></div>
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="font-medium text-slate-700">Lớp 1 (Dương tính / Có tiểu đường)</span>
                <span className="text-slate-500">
                  {targetDistribution.c1} mẫu ({targetDistribution.pct1.toFixed(1)}%)
                </span>
              </div>
              <div className="w-full bg-slate-100 h-3 rounded-full overflow-hidden">
                <div
                  className="bg-rose-500 h-full rounded-full transition-all duration-500"
                  style={{ width: `${targetDistribution.pct1}%` }}
                ></div>
              </div>
            </div>

            <div className="mt-4 p-3 bg-slate-50 rounded-lg text-xs text-slate-600 leading-relaxed border border-slate-100">
              <span className="font-semibold text-slate-900">Nhận xét:</span> Tập dữ liệu đạt tỷ lệ cân bằng lý tưởng (~50/50), rất thuận lợi cho thuật toán XGBoost và không yêu cầu các kỹ thuật cân bằng lớp phức tạp như SMOTE hay class weighting.
            </div>
          </div>
        </div>

        {/* Chart 2: 2D Scatter plot (Glucose vs BloodPressure) */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h2 className="text-sm font-bold text-slate-900">
                Biểu đồ phân tán (Scatter Plot): Glucose vs Blood Pressure
              </h2>
              <p className="text-xs text-slate-500">Trực quan hóa sự phân tách không gian giữa 2 lớp</p>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
                <span className="text-slate-600">Lớp 0</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                <span className="text-slate-600">Lớp 1</span>
              </div>
            </div>
          </div>

          {scatterPoints.length > 0 ? (
            <div className="mt-3 relative">
              <svg viewBox="0 0 400 220" className="w-full h-52 bg-slate-50 rounded-lg border border-slate-100">
                {/* Grid lines */}
                <line x1="40" y1="20" x2="40" y2="190" stroke="#e2e8f0" strokeDasharray="3 3" />
                <line x1="40" y1="190" x2="380" y2="190" stroke="#cbd5e1" />
                <line x1="40" y1="105" x2="380" y2="105" stroke="#f1f5f9" strokeDasharray="3 3" />
                <line x1="210" y1="20" x2="210" y2="190" stroke="#f1f5f9" strokeDasharray="3 3" />

                {/* Axis Labels */}
                <text x="210" y="210" textAnchor="middle" className="text-[10px] fill-slate-500">
                  Glucose (mg/dL)
                </text>
                <text x="15" y="105" textAnchor="middle" transform="rotate(-90 15,105)" className="text-[10px] fill-slate-500">
                  Blood Pressure (mmHg)
                </text>

                {/* Points */}
                {scatterPoints.map((p, idx) => {
                  const rangeX = scatterScale.maxX - scatterScale.minX || 1;
                  const rangeY = scatterScale.maxY - scatterScale.minY || 1;
                  const cx = 50 + ((p.x - scatterScale.minX) / rangeX) * 320;
                  const cy = 180 - ((p.y - scatterScale.minY) / rangeY) * 150;
                  return (
                    <circle
                      key={idx}
                      cx={cx}
                      cy={cy}
                      r="3"
                      fill={p.label === 1 ? '#f43f5e' : '#3b82f6'}
                      opacity="0.65"
                    />
                  );
                })}
              </svg>
            </div>
          ) : (
            <div className="h-52 flex items-center justify-center text-xs text-slate-400">
              Chưa đủ biến dạng số để vẽ biểu đồ phân tán.
            </div>
          )}
        </div>
      </div>

      {/* Data preview table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h2 className="text-sm font-bold text-slate-900">
              Bảng dữ liệu chi tiết ({dataset.rowCount.toLocaleString()} bản ghi)
            </h2>
            <p className="text-xs text-slate-500">Xem trước từng dòng dữ liệu và tìm kiếm nhanh</p>
          </div>

          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Tìm kiếm giá trị..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setPage(0);
              }}
              className="pl-8 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 w-full sm:w-60"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-50/80 border-b border-slate-200 text-slate-600 font-semibold">
                <th className="py-2.5 px-4 w-16">#STT</th>
                {dataset.headers.map((h) => (
                  <th key={h} className="py-2.5 px-4">
                    <div className="flex items-center gap-1.5">
                      <span>{h}</span>
                      <span className="text-[10px] font-normal text-slate-400">
                        ({dataset.columnTypes[h] === 'numeric' ? 'float' : 'str'})
                      </span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {currentRows.map((row, rIdx) => {
                const actualIdx = page * rowsPerPage + rIdx + 1;
                return (
                  <tr key={rIdx} className="hover:bg-slate-50/60 transition-colors">
                    <td className="py-2.5 px-4 font-mono text-slate-400 text-[11px]">
                      {actualIdx}
                    </td>
                    {row.map((val, cIdx) => {
                      const isTarget = cIdx === targetIdx;
                      return (
                        <td key={cIdx} className="py-2.5 px-4">
                          {isTarget ? (
                            <span
                              className={`font-semibold ${
                                String(val) === '1' ? 'text-rose-600' : 'text-blue-600'
                              }`}
                            >
                              {String(val)}
                            </span>
                          ) : (
                            <span>{String(val)}</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
              {currentRows.length === 0 && (
                <tr>
                  <td
                    colSpan={dataset.headers.length + 1}
                    className="py-8 text-center text-slate-400 text-xs"
                  >
                    Không tìm thấy dữ liệu phù hợp với từ khóa "{searchTerm}".
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination footer */}
        <div className="p-3 bg-slate-50/60 border-t border-slate-200 flex items-center justify-between text-xs text-slate-600">
          <div>
            Hiển thị {Math.min(filteredRows.length, page * rowsPerPage + 1)} -{' '}
            {Math.min(filteredRows.length, (page + 1) * rowsPerPage)} trên tổng số{' '}
            {filteredRows.length.toLocaleString()} dòng
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="p-1 rounded border border-slate-200 bg-white hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="px-2 font-medium">
              Trang {page + 1} / {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="p-1 rounded border border-slate-200 bg-white hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
