import React from 'react';
import { Zap, ShieldCheck } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="mt-12 bg-white border-t border-slate-200 py-6">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-md bg-blue-600 flex items-center justify-center text-white">
              <Zap className="w-3 h-3 text-amber-300 fill-amber-300" />
            </div>
            <span className="font-semibold text-slate-800">
              XGBoost ML Studio • Naive Bayes Dataset Engine
            </span>
          </div>

          {/* Mandatory Copyright notice */}
          <div className="text-center sm:text-right">
            <p className="font-medium text-slate-800">
              © 2026 - Phát triển bởi Dinh PT (dinh.pt.vlu.edu@gmail.com). Bảo lưu mọi quyền.
            </p>
            <div className="flex items-center justify-center sm:justify-end gap-1.5 text-[11px] text-slate-400 mt-0.5">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" />
              <span>Hệ thống tuân thủ chuẩn học máy thực nghiệm & khoa học dữ liệu</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};
