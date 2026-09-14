import { Link, Route, Routes, useLocation } from 'react-router'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import Dashboard from './pages/Dashboard.jsx'
import PRDetail from './pages/PRDetail.jsx'

export default function App() {
  const location = useLocation()
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-baseline gap-x-3 gap-y-1 px-4 py-4 sm:px-6">
          <Link
            to="/"
            className="rounded text-lg font-semibold text-slate-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900"
          >
            Deployment Risk
          </Link>
          <span className="text-sm text-slate-500">Risk scores for pull requests</span>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        {/* Keyed by path, so moving to another page clears a previous crash. */}
        <ErrorBoundary key={location.pathname}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/pr/:id" element={<PRDetail />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </ErrorBoundary>
      </main>
    </div>
  )
}

function NotFound() {
  return (
    <div className="space-y-3">
      <h1 className="text-xl font-semibold text-slate-900">Page not found</h1>
      <Link
        to="/"
        className="rounded text-sm font-medium text-slate-700 underline underline-offset-2 hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900"
      >
        Back to the dashboard
      </Link>
    </div>
  )
}
