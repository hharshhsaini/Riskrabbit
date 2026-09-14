import { useState } from 'react'
import { Link, useSearchParams } from 'react-router'
import { connectRepo, getPullRequests, getRepos } from '../api.js'
import ErrorMessage from '../components/ErrorMessage.jsx'
import SkeletonRows from '../components/SkeletonRows.jsx'
import StateBadge from '../components/StateBadge.jsx'
import { useApi } from '../useApi.js'

// Mirrors the backend's RepositoryCreate validation.
const REPO_SEGMENT = /^[A-Za-z0-9._-]+$/
const MAX_SEGMENT_LENGTH = 100

export default function Dashboard() {
  // The selected repository lives in the URL (?repo=id), so Back from a PR returns to it.
  const [searchParams, setSearchParams] = useSearchParams()
  const selectedId = searchParams.get('repo')
  const repos = useApi(getRepos, [])
  const selectedRepo = repos.data?.find((repo) => repo.id === selectedId) ?? null

  const selectRepo = (id) => setSearchParams({ repo: id })

  return (
    <div className="space-y-6">
      <ConnectRepoForm
        onConnected={(repo) => {
          repos.refetch()
          selectRepo(repo.id)
        }}
      />
      <div className="grid items-start gap-6 lg:grid-cols-[18rem_minmax(0,1fr)]">
        <RepositoryList repos={repos} selectedId={selectedId} onSelect={selectRepo} />
        <PullRequestList repo={selectedRepo} noRepositories={Array.isArray(repos.data) && repos.data.length === 0} />
      </div>
    </div>
  )
}

function ConnectRepoForm({ onConnected }) {
  const [value, setValue] = useState('')
  const [error, setError] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit(event) {
    event.preventDefault()
    const parsed = parseRepository(value)
    if (parsed.error) {
      setError(parsed.error)
      return
    }
    setSubmitting(true)
    setError(null)
    try {
      const repo = await connectRepo(parsed.owner, parsed.name)
      setValue('')
      onConnected(repo)
    } catch (apiError) {
      setError(apiError.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} noValidate className="rounded-lg border border-slate-200 bg-white p-4 sm:p-5">
      <label htmlFor="repo-input" className="block text-sm font-semibold text-slate-900">
        Connect a public GitHub repository
      </label>
      <div className="mt-2 flex flex-col gap-2 sm:flex-row">
        <input
          id="repo-input"
          type="text"
          value={value}
          onChange={(event) => {
            setValue(event.target.value)
            if (error) setError(null)
          }}
          placeholder="owner/name, e.g. pallets/flask"
          autoComplete="off"
          spellCheck="false"
          readOnly={submitting}
          aria-invalid={error ? 'true' : undefined}
          aria-describedby={error ? 'repo-input-error' : 'repo-input-hint'}
          className="min-w-0 flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 read-only:bg-slate-50 read-only:text-slate-500 focus-visible:border-slate-900 focus-visible:outline-2 focus-visible:outline-offset-0 focus-visible:outline-slate-900 aria-invalid:border-red-500"
        />
        <button
          type="submit"
          disabled={submitting}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {submitting ? 'Connecting…' : 'Connect'}
        </button>
      </div>
      {error ? (
        <p id="repo-input-error" role="alert" className="mt-2 text-sm text-red-700">
          {error}
        </p>
      ) : (
        <p id="repo-input-hint" className="mt-2 text-sm text-slate-500">
          A GitHub URL works too.
        </p>
      )}
    </form>
  )
}

function RepositoryList({ repos, selectedId, onSelect }) {
  let content
  if (repos.error) {
    content = (
      <div className="p-4">
        <ErrorMessage error={repos.error} onRetry={repos.refetch} />
      </div>
    )
  } else if (repos.loading && !repos.data) {
    content = <SkeletonRows rows={3} label="Loading repositories" />
  } else if (!repos.data?.length) {
    content = <p className="px-4 py-6 text-sm text-slate-500">No repositories connected yet — add one above</p>
  } else {
    content = (
      <ul className="divide-y divide-slate-100">
        {repos.data.map((repo) => {
          const selected = repo.id === selectedId
          return (
            <li key={repo.id}>
              <button
                type="button"
                onClick={() => onSelect(repo.id)}
                aria-current={selected ? 'true' : undefined}
                className={`block w-full break-words px-4 py-3 text-left text-sm focus-visible:outline-2 focus-visible:-outline-offset-4 ${
                  // A dark outline would be invisible on the dark selected background.
                  selected
                    ? 'bg-slate-900 text-white focus-visible:outline-white'
                    : 'text-slate-700 hover:bg-slate-50 focus-visible:outline-slate-900'
                }`}
              >
                <span className={selected ? 'text-slate-300' : 'text-slate-500'}>{repo.owner}/</span>
                <span className="font-medium">{repo.name}</span>
              </button>
            </li>
          )
        })}
      </ul>
    )
  }

  return (
    <section aria-labelledby="repositories-heading" className="overflow-hidden rounded-lg border border-slate-200 bg-white">
      <h2 id="repositories-heading" className="border-b border-slate-200 px-4 py-3 text-sm font-semibold text-slate-900">
        Repositories
      </h2>
      {content}
    </section>
  )
}

function PullRequestList({ repo, noRepositories }) {
  const prs = useApi(() => (repo ? getPullRequests(repo.id) : Promise.resolve(null)), [repo?.id])

  let content
  if (!repo) {
    content = (
      <p className="px-4 py-6 text-sm text-slate-500">
        {noRepositories
          ? 'Connect a repository above to see its pull requests.'
          : 'Select a repository to see its pull requests.'}
      </p>
    )
  } else if (prs.error) {
    content = (
      <div className="p-4">
        <ErrorMessage error={prs.error} onRetry={prs.refetch} />
      </div>
    )
  } else if (prs.loading) {
    // Always the skeleton while loading: any data still held belongs to the previous repository.
    content = <SkeletonRows rows={6} label="Loading pull requests" />
  } else if (!prs.data?.length) {
    content = (
      <div className="space-y-3 px-4 py-6 text-sm text-slate-500">
        <p>No pull requests found in this repository. Open one on GitHub, then check again.</p>
        <button
          type="button"
          onClick={prs.refetch}
          className="rounded-md border border-slate-300 px-3 py-1.5 font-medium text-slate-700 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900"
        >
          Check again
        </button>
      </div>
    )
  } else {
    content = (
      <ul className="divide-y divide-slate-100">
        {prs.data.map((pr) => (
          <PullRequestRow key={pr.id} pr={pr} repo={repo} />
        ))}
      </ul>
    )
  }

  return (
    <section aria-labelledby="pull-requests-heading" className="overflow-hidden rounded-lg border border-slate-200 bg-white">
      <div className="flex items-baseline justify-between gap-4 border-b border-slate-200 px-4 py-3">
        <h2 id="pull-requests-heading" className="text-sm font-semibold text-slate-900">
          {repo ? `Pull requests in ${repo.owner}/${repo.name}` : 'Pull requests'}
        </h2>
        {repo && !prs.loading && prs.data?.length > 0 && (
          <span className="text-xs tabular-nums text-slate-500">{prs.data.length} shown</span>
        )}
      </div>
      {content}
    </section>
  )
}

function PullRequestRow({ pr, repo }) {
  const githubUrl = `https://github.com/${repo.owner}/${repo.name}/pull/${pr.github_pr_number}`
  return (
    <li className="flex items-center gap-2 hover:bg-slate-50">
      <Link
        to={`/pr/${pr.id}`}
        className="flex min-w-0 flex-1 items-center gap-3 px-4 py-3 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-slate-900"
      >
        <span className="w-14 shrink-0 font-mono text-sm tabular-nums text-slate-500">#{pr.github_pr_number}</span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium text-slate-900" title={pr.title || undefined}>
            {pr.title || 'Untitled pull request'}
          </span>
          <span className="block truncate text-xs text-slate-500">{pr.author ? `by ${pr.author}` : 'Author unknown'}</span>
        </span>
        <StateBadge state={pr.state} />
      </Link>
      <a
        href={githubUrl}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={`Open pull request #${pr.github_pr_number} on GitHub (opens in a new tab)`}
        className="mr-3 shrink-0 rounded px-2 py-1 text-xs font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-slate-900"
      >
        GitHub ↗
      </a>
    </li>
  )
}

function parseRepository(raw) {
  const cleaned = raw
    .trim()
    .replace(/^https?:\/\/(www\.)?github\.com\//i, '')
    .replace(/\.git$/i, '')
    .replace(/\/+$/, '')
  const parts = cleaned.split('/')
  if (!cleaned || parts.length !== 2 || parts.some((part) => !part)) {
    return { error: 'Enter the repository as owner/name, for example pallets/flask.' }
  }
  const [owner, name] = parts
  if (!REPO_SEGMENT.test(owner) || !REPO_SEGMENT.test(name)) {
    return { error: 'Owner and name can only contain letters, numbers, ".", "_" and "-".' }
  }
  if (owner.length > MAX_SEGMENT_LENGTH || name.length > MAX_SEGMENT_LENGTH) {
    return { error: 'Owner and name must each be 100 characters or fewer.' }
  }
  return { owner, name }
}
