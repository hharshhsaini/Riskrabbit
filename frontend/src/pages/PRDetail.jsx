import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router'
import { getPRHistory, getPullRequest, predict } from '../api.js'
import ErrorMessage from '../components/ErrorMessage.jsx'
import ExplanationPanel from '../components/ExplanationPanel.jsx'
import HistoryTable from '../components/HistoryTable.jsx'
import RiskBadge from '../components/RiskBadge.jsx'
import SkeletonRows from '../components/SkeletonRows.jsx'
import StateBadge from '../components/StateBadge.jsx'
import { formatDateTime } from '../format.js'
import { useApi } from '../useApi.js'

// Shown on a timer while the request runs. Timings follow the pipeline's measured
// stages: ~1.6 s fetching from GitHub, then scoring, then the LLM explanation.
const STAGES = [
  { afterMs: 0, text: 'Fetching PR data…' },
  { afterMs: 1500, text: 'Scoring…' },
  { afterMs: 2500, text: 'Generating explanation…' },
]

export default function PRDetail() {
  const { id } = useParams()
  const pullRequest = useApi(() => getPullRequest(id), [id])
  const history = useApi(() => getPRHistory(id), [id])
  const analysis = useAnalysis(id, history.refetch)
  const latest = analysis.result ?? history.data?.[0] ?? null

  // Without the pull request nothing else on the page can work, and the history
  // request fails for the same reason — show the one error, not two.
  if (pullRequest.error) {
    return (
      <div className="space-y-6">
        <BackButton />
        <ErrorMessage error={pullRequest.error} onRetry={pullRequest.refetch} />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <BackButton />
      <PullRequestHeader pullRequest={pullRequest} />

      <section aria-labelledby="risk-heading" className="rounded-lg border border-slate-200 bg-white p-4 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 id="risk-heading" className="text-sm font-semibold text-slate-900">
            Deployment risk
          </h2>
          <button
            type="button"
            onClick={analysis.analyze}
            disabled={analysis.analyzing || !pullRequest.data}
            aria-busy={analysis.analyzing}
            className="inline-flex items-center gap-2 rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {analysis.analyzing && <Spinner />}
            {analysis.analyzing ? 'Analyzing…' : latest ? 'Analyze again' : 'Analyze'}
          </button>
        </div>

        <div className="mt-4">
          {analysis.analyzing ? (
            <p role="status" aria-live="polite" className="flex items-center gap-2 text-sm text-slate-600">
              <Spinner />
              {analysis.stageText}
            </p>
          ) : (
            <div className="space-y-4">
              {/* A failed run keeps the last successful result visible below the error. */}
              {analysis.error && <ErrorMessage error={analysis.error} onRetry={analysis.analyze} />}
              {latest ? (
                <>
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                    <RiskBadge score={latest.risk_score} label={latest.risk_label} />
                    <span className="text-xs text-slate-500">
                      Analysed {formatDateTime(latest.created_at)} · model {latest.model_version}
                    </span>
                  </div>
                  <ExplanationPanel explanation={latest.explanation} features={latest.features} />
                </>
              ) : (
                !analysis.error && (
                  <p className="text-sm text-slate-500">
                    Not analysed yet. Click Analyze to fetch the latest data from GitHub, score the pull request and
                    explain the score.
                  </p>
                )
              )}
            </div>
          )}
        </div>
      </section>

      <section aria-labelledby="history-heading" className="rounded-lg border border-slate-200 bg-white p-4 sm:p-5">
        <h2 id="history-heading" className="mb-3 text-sm font-semibold text-slate-900">
          Analysis history
        </h2>
        {history.error ? (
          <ErrorMessage error={history.error} onRetry={history.refetch} />
        ) : history.loading && !history.data ? (
          <SkeletonRows rows={2} label="Loading analysis history" />
        ) : (
          <HistoryTable predictions={history.data} />
        )}
      </section>
    </div>
  )
}

function useAnalysis(pullRequestId, onSuccess) {
  const [stage, setStage] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  // A ref, not state: it updates immediately, so a second click in the same instant is ignored.
  const inFlight = useRef(false)
  const timers = useRef([])

  useEffect(() => () => timers.current.forEach(clearTimeout), [])

  async function analyze() {
    if (inFlight.current) return
    inFlight.current = true
    setError(null)
    setStage(0)
    timers.current = STAGES.slice(1).map(({ afterMs }, index) => setTimeout(() => setStage(index + 1), afterMs))
    try {
      const prediction = await predict(pullRequestId)
      setResult(prediction)
      onSuccess()
    } catch (analysisError) {
      setError(analysisError)
    } finally {
      timers.current.forEach(clearTimeout)
      timers.current = []
      setStage(null)
      inFlight.current = false
    }
  }

  return {
    analyze,
    analyzing: stage !== null,
    stageText: stage === null ? null : STAGES[stage].text,
    // Ignore a result that belongs to a different pull request after navigating away mid-request.
    result: result?.pull_request_id === pullRequestId ? result : null,
    error,
  }
}

function PullRequestHeader({ pullRequest }) {
  if (pullRequest.loading && !pullRequest.data) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white">
        <SkeletonRows rows={2} label="Loading pull request" />
      </div>
    )
  }
  const pr = pullRequest.data
  const repo = pr.repository
  const githubUrl = `https://github.com/${repo.owner}/${repo.name}/pull/${pr.github_pr_number}`

  return (
    <header className="rounded-lg border border-slate-200 bg-white p-4 sm:p-5">
      <div className="flex flex-wrap items-center gap-2 text-sm text-slate-500">
        <span className="break-words">
          {repo.owner}/{repo.name}
        </span>
        <span className="font-mono tabular-nums">#{pr.github_pr_number}</span>
        <StateBadge state={pr.state} />
      </div>
      <h1 className="mt-2 text-xl font-semibold text-balance break-words text-slate-900">{pr.title || 'Untitled pull request'}</h1>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-slate-500">
        <span>{pr.author ? `by ${pr.author}` : 'Author unknown'}</span>
        {pr.opened_at && <span>Opened {formatDateTime(pr.opened_at)}</span>}
        {pr.merged_at && <span>Merged {formatDateTime(pr.merged_at)}</span>}
        <a
          href={githubUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="rounded font-medium text-slate-700 underline underline-offset-2 hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900"
        >
          View on GitHub ↗
        </a>
      </div>
    </header>
  )
}

function BackButton() {
  const navigate = useNavigate()
  const location = useLocation()
  // "default" means this is the first page in the session, so there is nothing to go back to.
  const goBack = () => (location.key === 'default' ? navigate('/') : navigate(-1))
  return (
    <button
      type="button"
      onClick={goBack}
      className="rounded text-sm font-medium text-slate-600 hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900"
    >
      ← Back
    </button>
  )
}

function Spinner() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="size-4 motion-safe:animate-spin">
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="3" className="opacity-25" />
      <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}
