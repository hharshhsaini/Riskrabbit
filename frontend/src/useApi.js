import { useCallback, useEffect, useRef, useState } from 'react'

// Runs fn() when the component mounts and whenever deps change.
// Returns {data, loading, error, refetch}. Only the latest call may update state,
// so a slow earlier response can never overwrite a newer one.
export function useApi(fn, deps = []) {
  const [state, setState] = useState({ data: null, loading: true, error: null })
  const latestCall = useRef(0)

  const run = useCallback(async () => {
    const call = ++latestCall.current
    setState((previous) => ({ ...previous, loading: true, error: null }))
    try {
      const data = await fn()
      if (call === latestCall.current) setState({ data, loading: false, error: null })
    } catch (error) {
      if (call === latestCall.current) setState((previous) => ({ ...previous, loading: false, error }))
    }
    // deps come from the caller, so they cannot be an array literal here.
    // oxlint-disable-next-line react/use-memo, react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    run()
    return () => {
      latestCall.current += 1
    }
  }, [run])

  return { ...state, refetch: run }
}
