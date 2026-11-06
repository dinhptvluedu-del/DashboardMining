import React from 'react';
import { Database, Zap, RefreshCw, Upload, FileCode2 } from 'lucide-react';
import { Dataset, TrainedModel, ActiveTab } from '../types';

interface NavbarProps {
  dataset: Dataset;
  model: TrainedModel | null;
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  onResetData: () => void;
  onUploadClick: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  dataset,
  model,
  activeTab,
  setActiveTab,
  onResetData,
  onUploadClick,
}) => {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-sm shadow-blue-500/20">
              <Zap className="w-5 h-5 text-amber-300 fill-amber-300" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-slate-900 text-lg tracking-tight">
                  XGBoost ML Studio
                </span>
                <span className="text-[11px] font-semibold uppercase tracking-wider text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200/60">
                  v2.5
                </span>
              </div>
              <p className="text-xs text-slate-500 hidden sm:block">
                Hệ thống phân tích dữ liệu & huấn luyện mô hình phân loại Naive-Bayes
              </p>
            </div>
          </div>

          {/* Center: Current Dataset info */}
          <div className="hidden md:flex items-center gap-3 text-xs bg-slate-50 border border-slate-200 px-3.5 py-1.5 rounded-lg">
            <Database className="w-4 h-4 text-slate-500" />
            <span className="font-medium text-slate-700 truncate max-w-[200px]" title={dataset.filename}>
              {dataset.filename}
            </span>
            <span className="text-slate-300">|</span>
            <span className="text-slate-600">{dataset.rowCount} mẫu</span>
            <span className="text-slate-300">·</span>
            <span className="text-slate-600">{dataset.colCount} cột</span>
          </div>

          {/* Right actions */}
          <div className="flex items-center gap-2">
            {model ? (
              <div className="hidden lg:flex items-center gap-2 text-xs bg-emerald-50 text-emerald-700 border border-emerald-200 px-2.5 py-1.5 rounded-lg">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="font-medium">Mô hình đã sẵn sàng</span>
                <span className="text-emerald-600/70">
                  (Acc: {(model.evaluation.accuracy * 100).toFixed(1)}%)
                </span>
              </div>
            ) : (
              <div className="hidden lg:flex items-center gap-1.5 text-xs text-slate-500 bg-slate-100 px-2.5 py-1.5 rounded-lg">
                <span className="w-2 h-2 rounded-full bg-slate-400"></span>
                <span>Chưa huấn luyện</span>
              </div>
            )}

            <button
              onClick={onUploadClick}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg transition-colors cursor-pointer"
              title="Tải lên tệp CSV khác"
            >
              <Upload className="w-3.5 h-3.5 text-slate-500" />
              <span className="hidden sm:inline">Tải CSV</span>
            </button>

            <button
              onClick={onResetData}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg transition-colors cursor-pointer"
              title="Đặt lại dữ liệu gốc Naive-Bayes"
            >
              <RefreshCw className="w-3.5 h-3.5 text-slate-500" />
              <span className="hidden sm:inline">Mặc định</span>
            </button>

            <button
              onClick={() => setActiveTab('code')}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
                activeTab === 'code'
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-200'
              }`}
              title="Xem mã nguồn Python Streamlit (app.py)"
            >
              <FileCode2 className="w-3.5 h-3.5" />
              <span>Mã Python</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
