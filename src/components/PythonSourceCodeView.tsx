import React, { useState } from 'react';
import {
  FileCode,
  Copy,
  Check,
  Download,
  Terminal,
  FileText,
  FileSpreadsheet,
} from 'lucide-react';
import { downloadCSV } from '../utils/csvParser';
import { NAIVE_BAYES_CSV_CONTENT } from '../utils/naiveBayesData';

interface PythonSourceCodeViewProps {
  appPyCode: string;
  requirementsTxt: string;
}

export const PythonSourceCodeView: React.FC<PythonSourceCodeViewProps> = ({
  appPyCode,
  requirementsTxt,
}) => {
  const [selectedFile, setSelectedFile] = useState<'app.py' | 'requirements.txt' | 'instructions'>('app.py');
  const [copied, setCopied] = useState(false);

  const getActiveCode = () => {
    if (selectedFile === 'app.py') return appPyCode;
    if (selectedFile === 'requirements.txt') return requirementsTxt;
    return `# Hướng dẫn cài đặt và khởi chạy:
# 1. Cài đặt các thư viện phụ thuộc:
pip install -r requirements.txt

# 2. Khởi chạy ứng dụng web Streamlit:
streamlit run app.py

# 3. Ứng dụng sẽ tự động mở trên trình duyệt tại:
# http://localhost:8501
`;
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(getActiveCode());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = (filename: string, content: string) => {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadDataset = () => {
    downloadCSV('Naive-Bayes-Classification-Data.csv', NAIVE_BAYES_CSV_CONTENT);
  };

  const activeContent = getActiveCode();
  const codeLines = activeContent.split('\n');

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">
            8. Toàn Bộ Mã Nguồn Python (Streamlit) & Hướng Dẫn Cài Đặt
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Mã nguồn hoàn chỉnh, độc lập, có thể tải về chạy trực tiếp trên máy tính cá nhân bằng lệnh <code className="bg-slate-100 px-1 py-0.5 rounded text-blue-700">streamlit run app.py</code>
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <button
            onClick={downloadDataset}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg shadow-xs transition-colors cursor-pointer"
            title="Tải tệp dữ liệu CSV chuẩn"
          >
            <FileSpreadsheet className="w-3.5 h-3.5 text-slate-500" />
            <span>Tải CSV Dữ liệu</span>
          </button>

          <button
            onClick={() => handleDownload('requirements.txt', requirementsTxt)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg shadow-xs transition-colors cursor-pointer"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" />
            <span>Tải requirements.txt</span>
          </button>

          <button
            onClick={() => handleDownload('app.py', appPyCode)}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-bold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-xs transition-colors cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Tải app.py</span>
          </button>
        </div>
      </div>

      {/* Terminal Quick Guide Banner */}
      <div className="bg-slate-900 rounded-xl p-4 text-slate-200 font-mono text-xs border border-slate-800 space-y-2 shadow-sm">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-emerald-400" />
            <span className="font-semibold text-slate-300">
              Terminal: Cài đặt & Khởi chạy Streamlit chỉ trong 2 bước
            </span>
          </div>
          <span className="text-[11px] text-slate-400">Python 3.8+</span>
        </div>
        <div className="space-y-1 text-slate-300">
          <p>
            <span className="text-emerald-400 select-none">$ </span>
            <span className="text-white">pip install -r requirements.txt</span>
          </p>
          <p>
            <span className="text-emerald-400 select-none">$ </span>
            <span className="text-white">streamlit run app.py</span>
          </p>
        </div>
      </div>

      {/* Code Viewer Panel */}
      <div className="bg-slate-950 rounded-xl border border-slate-800 overflow-hidden shadow-lg">
        {/* Top bar */}
        <div className="bg-slate-900/90 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
          {/* File tabs */}
          <div className="flex items-center gap-1">
            <button
              onClick={() => setSelectedFile('app.py')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                selectedFile === 'app.py'
                  ? 'bg-slate-800 text-blue-400 border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <FileCode className="w-3.5 h-3.5" />
              <span>app.py</span>
            </button>

            <button
              onClick={() => setSelectedFile('requirements.txt')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                selectedFile === 'requirements.txt'
                  ? 'bg-slate-800 text-blue-400 border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <FileText className="w-3.5 h-3.5" />
              <span>requirements.txt</span>
            </button>

            <button
              onClick={() => setSelectedFile('instructions')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                selectedFile === 'instructions'
                  ? 'bg-slate-800 text-blue-400 border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Terminal className="w-3.5 h-3.5" />
              <span>Hướng dẫn (README)</span>
            </button>
          </div>

          {/* Copy button */}
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700 transition-colors cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400">Đã sao chép!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Sao chép mã</span>
              </>
            )}
          </button>
        </div>

        {/* Code Content */}
        <div className="p-4 font-mono text-xs overflow-x-auto max-h-[600px] text-slate-200 leading-relaxed">
          <table className="w-full border-collapse">
            <tbody>
              {codeLines.map((line, idx) => (
                <tr key={idx} className="hover:bg-slate-900/40">
                  <td className="w-10 pr-4 text-right text-slate-600 select-none text-[11px]">
                    {idx + 1}
                  </td>
                  <td className="whitespace-pre">
                    {/* Basic syntax coloration for comments & keywords */}
                    {line.startsWith('#') || line.startsWith('"""') ? (
                      <span className="text-slate-500 italic">{line}</span>
                    ) : line.includes('import ') || line.includes('from ') ? (
                      <span className="text-purple-400">{line}</span>
                    ) : line.includes('def ') || line.includes('class ') ? (
                      <span className="text-blue-400">{line}</span>
                    ) : line.includes('st.') ? (
                      <span className="text-emerald-400">{line}</span>
                    ) : (
                      <span>{line}</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
