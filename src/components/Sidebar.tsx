import React from 'react';
import {
  LayoutDashboard,
  BarChart3,
  SlidersHorizontal,
  Cpu,
  Target,
  Sparkles,
  History,
  FileCode,
  FileSpreadsheet,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';
import { ActiveTab, Dataset, TrainedModel, PreprocessedData } from '../types';

interface SidebarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  dataset: Dataset;
  model: TrainedModel | null;
  preprocessed: PreprocessedData | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  dataset,
  model,
  preprocessed,
}) => {
  const menuItems: { id: ActiveTab; label: string; icon: React.ReactNode; badge?: string }[] = [
    {
      id: 'overview',
      label: '1. Dashboard Tổng kết',
      icon: <LayoutDashboard className="w-4 h-4" />,
    },
    {
      id: 'descriptive',
      label: '2. Thống kê mô tả',
      icon: <BarChart3 className="w-4 h-4" />,
    },
    {
      id: 'preprocessing',
      label: '3. Tiền xử lý dữ liệu',
      icon: <SlidersHorizontal className="w-4 h-4" />,
      badge: preprocessed ? 'Sẵn sàng' : undefined,
    },
    {
      id: 'training',
      label: '4. Huấn luyện XGBoost',
      icon: <Cpu className="w-4 h-4" />,
    },
    {
      id: 'evaluation',
      label: '5. Đánh giá mô hình',
      icon: <Target className="w-4 h-4" />,
      badge: model ? `${(model.evaluation.accuracy * 100).toFixed(0)}%` : undefined,
    },
    {
      id: 'prediction',
      label: '6. Dự đoán dữ liệu mới',
      icon: <Sparkles className="w-4 h-4" />,
    },
    {
      id: 'history',
      label: '7. Lịch sử thử nghiệm',
      icon: <History className="w-4 h-4" />,
    },
    {
      id: 'code',
      label: '8. Mã nguồn Python (Streamlit)',
      icon: <FileCode className="w-4 h-4" />,
      badge: 'app.py',
    },
  ];

  return (
    <aside className="w-full md:w-64 bg-slate-900 text-slate-200 flex-shrink-0 flex flex-col justify-between border-r border-slate-800">
      <div className="p-4 space-y-6">
        {/* Dataset Quick Summary Card */}
        <div className="bg-slate-800/80 rounded-xl p-3 border border-slate-700/60">
          <div className="flex items-center gap-2 mb-2">
            <FileSpreadsheet className="w-4 h-4 text-blue-400" />
            <span className="text-xs font-semibold text-slate-200 tracking-wide">
              TẬP DỮ LIỆU HIỆN TẠI
            </span>
          </div>
          <p className="text-xs font-medium text-white truncate" title={dataset.filename}>
            {dataset.filename}
          </p>
          <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-slate-400">
            <div>
              <span className="text-slate-500">Mẫu: </span>
              <span className="text-slate-200 font-semibold">{dataset.rowCount.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-slate-500">Cột: </span>
              <span className="text-slate-200 font-semibold">{dataset.colCount}</span>
            </div>
          </div>
        </div>

        {/* Navigation list */}
        <div>
          <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-2 mb-2">
            Mục lục chức năng
          </p>
          <nav className="space-y-1" aria-label="Sidebar Navigation">
            {menuItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2 text-xs font-medium rounded-lg transition-colors text-left cursor-pointer ${
                    isActive
                      ? 'bg-blue-600 text-white shadow-sm'
                      : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  <div className="flex items-center gap-2.5 truncate">
                    <span className={isActive ? 'text-white' : 'text-slate-400'}>
                      {item.icon}
                    </span>
                    <span className="truncate">{item.label}</span>
                  </div>
                  {item.badge && (
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                        isActive
                          ? 'bg-blue-700 text-blue-100'
                          : 'bg-slate-800 text-slate-400 border border-slate-700'
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Pipeline Readiness Checklist */}
        <div className="bg-slate-800/40 rounded-xl p-3 border border-slate-800 space-y-2">
          <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Tiến trình Pipeline
          </p>
          <div className="space-y-1.5 text-xs">
            <div className="flex items-center gap-2 text-slate-300">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
              <span>Dữ liệu đã tải ({dataset.rowCount} dòng)</span>
            </div>
            <div className="flex items-center gap-2 text-slate-300">
              {preprocessed ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
              ) : (
                <AlertCircle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
              )}
              <span>{preprocessed ? 'Đã tiền xử lý & phân chia' : 'Chưa tiền xử lý'}</span>
            </div>
            <div className="flex items-center gap-2 text-slate-300">
              {model ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
              ) : (
                <AlertCircle className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
              )}
              <span>{model ? `Mô hình XGBoost (Acc: ${(model.evaluation.accuracy * 100).toFixed(1)}%)` : 'Chưa huấn luyện'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Footer in Sidebar */}
      <div className="p-4 border-t border-slate-800 text-[11px] text-slate-400">
        <p className="font-medium text-slate-300">© 2026 - Dinh PT</p>
        <p className="text-slate-400">Khoa CNTT - VLU</p>
      </div>
    </aside>
  );
};
