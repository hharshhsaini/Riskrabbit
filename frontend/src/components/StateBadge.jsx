const STATE_STYLES = {
  open: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20',
  merged: 'bg-violet-50 text-violet-700 ring-violet-600/20',
  closed: 'bg-slate-100 text-slate-600 ring-slate-500/20',
}

export default function StateBadge({ state }) {
  const style = STATE_STYLES[state] ?? STATE_STYLES.closed
  return (
    <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium capitalize ring-1 ring-inset ${style}`}>
      {state || 'unknown'}
    </span>
  )
}
