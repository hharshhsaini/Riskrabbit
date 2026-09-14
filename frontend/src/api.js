// The only module that talks to the backend. Components call these functions,
// never fetch() directly, so an API change is a change to this one file.

// Without the trailing slash, so a pasted "https://api.example.com/" does not produce "//health".
const BASE_URL = import.meta.env.VITE_API_URL?.replace(/\/+$/, '')

async function request(path, { method = 'GET', body } = {}) {
  if (!BASE_URL) {
    throw new Error('VITE_API_URL is not set. Add it to frontend/.env (or the Vercel project settings) and rebuild.')
  }

  const options = { method, headers: {} }
  if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json'
    options.body = JSON.stringify(body)
  }

  let response
  try {
    response = await fetch(`${BASE_URL}${path}`, options)
  } catch {
    throw new Error(`Could not reach the API at ${BASE_URL}. Is the backend running?`)
  }

  const text = await response.text()
  let data = null
  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = null
    }
  }

  if (!response.ok) {
    throw new Error(detailMessage(data?.detail) || `The server returned an error (HTTP ${response.status}).`)
  }
  return data
}

// FastAPI sends a string detail for our own errors and a list of field errors for
// validation failures (422).
function detailMessage(detail) {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((item) => item?.msg).filter(Boolean).join('; ')
  }
  return null
}

const id = (value) => encodeURIComponent(value)

export const getRepos = () => request('/repositories')

export const connectRepo = (owner, name) =>
  request('/repositories', { method: 'POST', body: { owner, name } })

export const getPullRequests = (repoId) => request(`/repositories/${id(repoId)}/pull-requests`)

export const getPullRequest = (pullRequestId) => request(`/pull-requests/${id(pullRequestId)}`)

export const predict = (pullRequestId) =>
  request('/predictions', { method: 'POST', body: { pull_request_id: pullRequestId } })

export const getPRHistory = (pullRequestId) => request(`/predictions/pull-request/${id(pullRequestId)}`)

export const getRepoHistory = (repoId) => request(`/repositories/${id(repoId)}/predictions`)
