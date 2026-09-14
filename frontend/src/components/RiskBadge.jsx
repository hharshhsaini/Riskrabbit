import { formatScore } from '../format.js'

const LABEL_STYLES = {
  Low: { badge: 'bg-emerald-50 text-emerald-800 ring-emerald-600/25', dot: 'bg-emerald-500' },
  Medium: { badge: 'bg-amber-50 text-amber-800 ring-amber-600/30', dot: 'bg-amber-500' },
  High: { badge: 'bg-red-50 text-red-800 ring-red-600/25', dot: 'bg-red-500' },
}
const UNKNOWN = { badge: 'bg-slate-100 text-slate-700 ring-slate-500/20', dot: 'bg-slate-400' }

// The label is written out, so colour is never the only way to read the risk.
export default function RiskBadge({ score, label, size = 'md', showScore = true }) {
  const style = LABEL_STYLES[label] ?? UNKNOWN
  const sizing = size === 'sm' ? 'gap-1.5 px-2 py-0.5 text-xs' : 'gap-2 px-3 py-1 text-sm'
  return (
    <span className={`inline-flex items-center whitespace-nowrap rounded-full font-medium ring-1 ring-inset ${style.badge} ${sizing}`}>
      <span aria-hidden="true" className={`size-2 rounded-full ${style.dot}`} />
      <span>{label ? `${label} risk` : 'Not rated'}</span>
      {showScore && <span className="font-mono tabular-nums">{formatScore(score)}</span>}
    </span>
  )
}
