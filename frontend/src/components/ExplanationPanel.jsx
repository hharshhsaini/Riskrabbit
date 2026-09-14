// Stored features come back from a Postgres JSONB column, which does not keep key
// order, so rows are listed in this fixed order instead.
const FEATURE_ROWS = [
  ['files_changed', 'Files changed', 'count'],
  ['lines_added', 'Lines added', 'count'],
  ['lines_removed', 'Lines removed', 'count'],
  ['total_churn', 'Lines changed in total', 'count'],
  ['unique_dirs_touched', 'Directories touched', 'count'],
  ['commit_count', 'Commits', 'count'],
  ['avg_commit_msg_len', 'Average commit message length', 'count'],
  ['has_tests', 'Changes test files', 'flag'],
  ['test_file_ratio', 'Share of changed files that are tests', 'ratio'],
  ['touches_config', 'Changes configuration files', 'flag'],
  ['touches_ci', 'Changes CI workflow files', 'flag'],
  ['review_comment_count', 'Review comments', 'count'],
  ['pr_age_hours', 'Hours from opening to merge', 'hours'],
  ['is_weekend', 'Merged on a weekend', 'flag'],
  ['is_off_hours', 'Merged outside 09:00–18:00 UTC', 'flag'],
  ['author_pr_count', "Author's earlier merged PRs in this repository", 'count'],
]
const KNOWN = new Set(FEATURE_ROWS.map(([key]) => key))

export default function ExplanationPanel({ explanation, features }) {
  const values = features ?? {}
  const extra = Object.entries(values)
    .filter(([key, value]) => !KNOWN.has(key) && !key.startsWith('_') && typeof value === 'number')
    .map(([key]) => [key, key, 'count'])
  const rows = [...FEATURE_ROWS.filter(([key]) => key in values), ...extra]

  return (
    <div className="space-y-4">
      <p className="max-w-prose break-words text-sm leading-6 text-slate-700">
        {explanation || 'No explanation was generated for this analysis.'}
      </p>
      {rows.length === 0 && (
        <p className="text-sm text-slate-500">
          No measurements were saved with this analysis. Click Analyze again to record them.
        </p>
      )}
      {rows.length > 0 && (
        <table className="w-full max-w-xl text-sm">
          <caption className="mb-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
            Measurements used for this score
          </caption>
          <tbody className="divide-y divide-slate-100">
            {rows.map(([key, label, kind]) => (
              <tr key={key}>
                <th scope="row" className="py-1.5 pr-4 text-left font-normal text-slate-600">
                  {label}
                </th>
                <td className="py-1.5 text-right font-mono tabular-nums text-slate-900">{formatValue(values[key], kind)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

function formatValue(value, kind) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  if (kind === 'flag') return value >= 0.5 ? 'Yes' : 'No'
  if (kind === 'ratio') return `${Math.round(value * 100)}%`
  if (kind === 'hours') return value < 10 ? value.toFixed(1) : Math.round(value).toLocaleString()
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(1)
}
