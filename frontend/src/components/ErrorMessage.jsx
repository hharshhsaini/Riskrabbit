export default function ErrorMessage({ error, onRetry }) {
  return (
    <div
      role="alert"
      className="flex items-start justify-between gap-4 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
    >
      <p className="min-w-0 break-words">{error?.message || 'Something went wrong.'}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="shrink-0 rounded font-medium underline underline-offset-2 hover:text-red-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-700"
        >
          Retry
        </button>
      )}
    </div>
  )
}
