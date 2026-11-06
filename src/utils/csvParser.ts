/**
 * Robust CSV Parser & Serializer
 */
import { Dataset } from '../types';

export function parseCSV(rawText: string, filename = 'data.csv'): Dataset {
  const lines = rawText
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length === 0) {
    throw new Error('Tệp CSV trống hoặc không có dữ liệu hợp lệ.');
  }

  // Parse header
  const headerLine = lines[0];
  const headers = splitCSVRow(headerLine).map((h) => h.replace(/^["']|["']$/g, '').trim());

  if (headers.length === 0) {
    throw new Error('Không tìm thấy tiêu đề cột hợp lệ trong tệp CSV.');
  }

  const rawRows: (string | number)[][] = [];

  for (let i = 1; i < lines.length; i++) {
    const rawTokens = splitCSVRow(lines[i]);
    if (rawTokens.length === 0) continue;

    // Pad or trim to match headers count
    const rowValues: (string | number)[] = headers.map((_, colIdx) => {
      const valStr = rawTokens[colIdx]?.replace(/^["']|["']$/g, '').trim() ?? '';
      if (valStr === '' || valStr.toLowerCase() === 'nan' || valStr.toLowerCase() === 'null') {
        return '';
      }
      const num = Number(valStr);
      return !isNaN(num) && isFinite(num) ? num : valStr;
    });

    rawRows.push(rowValues);
  }

  // Detect column types
  const columnTypes: Record<string, 'numeric' | 'categorical'> = {};
  headers.forEach((header, colIdx) => {
    let numericCount = 0;
    let validCount = 0;
    for (const row of rawRows) {
      const val = row[colIdx];
      if (val !== '' && val !== null && val !== undefined) {
        validCount++;
        if (typeof val === 'number') {
          numericCount++;
        }
      }
    }
    // If >80% of non-empty values are numeric, treat as numeric
    columnTypes[header] = validCount > 0 && numericCount / validCount >= 0.8 ? 'numeric' : 'categorical';
  });

  return {
    name: filename.replace(/\.csv$/i, ''),
    filename,
    headers,
    rawRows,
    rowCount: rawRows.length,
    colCount: headers.length,
    columnTypes,
  };
}

function splitCSVRow(row: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < row.length; i++) {
    const char = row[i];
    if (char === '"' || char === "'") {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  result.push(current.trim());
  return result;
}

export function exportToCSV(headers: string[], rows: (string | number)[][]): string {
  const escapeCell = (val: string | number) => {
    const s = String(val ?? '');
    if (s.includes(',') || s.includes('"') || s.includes('\n')) {
      return `"${s.replace(/"/g, '""')}"`;
    }
    return s;
  };

  const headerRow = headers.map(escapeCell).join(',');
  const dataRows = rows.map((r) => r.map(escapeCell).join(',')).join('\n');
  return `${headerRow}\n${dataRows}`;
}

export function downloadCSV(filename: string, csvContent: string) {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
