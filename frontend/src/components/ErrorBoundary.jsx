import { Component } from 'react'

// Catches an error thrown while rendering, so a bug shows a message instead of a blank
// page. React has no hook for this, which is why this one component is a class.
export default class ErrorBoundary extends Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-5 text-red-800">
        <h1 className="text-base font-semibold">Something went wrong on this page.</h1>
        <p className="mt-1 text-sm">Reload the page to try again. If it keeps happening, go back to the dashboard.</p>
        <div className="mt-4 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="rounded-md bg-red-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-700"
          >
            Reload page
          </button>
          <a
            href="/"
            className="rounded-md px-3 py-1.5 text-sm font-medium text-red-800 underline underline-offset-2 hover:text-red-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-700"
          >
            Back to the dashboard
          </a>
        </div>
      </div>
    )
  }
}
