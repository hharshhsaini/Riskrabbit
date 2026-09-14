export default function SkeletonRows({ rows = 4, label = 'Loading' }) {
  return (
    <div>
      <p role="status" className="sr-only">
        {label}…
      </p>
      <ul aria-hidden="true" className="divide-y divide-slate-100">
        {Array.from({ length: rows }, (_, index) => (
          <li key={index} className="flex items-center gap-3 px-4 py-3.5">
            <div className="h-4 w-12 rounded bg-slate-200 motion-safe:animate-pulse" />
            <div className="h-4 flex-1 rounded bg-slate-200 motion-safe:animate-pulse" />
            <div className="h-5 w-16 rounded-full bg-slate-200 motion-safe:animate-pulse" />
          </li>
        ))}
      </ul>
    </div>
  )
}
