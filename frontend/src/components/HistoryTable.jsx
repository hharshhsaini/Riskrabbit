import { formatDateTime, formatScore } from '../format.js'
import RiskBadge from './RiskBadge.jsx'

export default function HistoryTable({ predictions }) {
  if (!predictions?.length) {
    return <p className="text-sm text-slate-500">No analyses yet. Click Analyze above to score this pull request.</p>
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th scope="col" className="py-2 pr-4 font-medium">Date</th>
            <th scope="col" className="py-2 pr-4 text-right font-medium">Score</th>
            {/* No right padding below sm: Label is the last visible column there (Model is hidden). */}
            <th scope="col" className="py-2 font-medium sm:pr-4">Label</th>
            <th scope="col" className="hidden py-2 font-medium sm:table-cell">Model</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {predictions.map((prediction) => (
            <tr key={prediction.id}>
              <td className="whitespace-nowrap py-2 pr-4 text-slate-700">{formatDateTime(prediction.created_at)}</td>
              <td className="py-2 pr-4 text-right font-mono tabular-nums text-slate-900">{formatScore(prediction.risk_score)}</td>
              <td className="py-2 sm:pr-4">
                <RiskBadge label={prediction.risk_label} size="sm" showScore={false} />
              </td>
              <td className="hidden whitespace-nowrap py-2 font-mono text-xs text-slate-500 sm:table-cell">
                {prediction.model_version}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
