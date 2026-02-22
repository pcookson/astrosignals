export interface TransitSearchRequest {
  target: string
  mission: 'TESS' | 'Kepler'
  author: string
  sector: number | null
  min_period_days: number
  max_period_days: number | null
  detrend_window_days: number
  fold_bins: number
}

export interface TransitCandidate {
  period_days: number
  t0_btjd: number
  duration_hours: number
  depth_pct: number
  depth_snr?: number | null
  sde?: number | null
  n_transits_observed: number
  time_system: {
    format: string
    zero_point: number | null
    scale: string
  }
  fit?: {
    depth_pct_fit: number
    duration_hours_fit: number
    ingress_minutes_fit: number
    baseline_fit: number
    phase0_fit: number
    rms_residual: number
    vshape_metric: number
  } | null
}

export interface FoldedCurve {
  phase: number[]
  flux: number[]
  bins: number
  model?: {
    phase: number[]
    flux: number[]
  }
}

export interface TransitSearchResponse {
  found: boolean
  candidate?: TransitCandidate
  folded?: FoldedCurve
  diagnostics?: any
  reason?: string
}

const configuredApiBaseUrl =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const normalizedApiBaseUrl = configuredApiBaseUrl.replace(/\/$/, '')

function apiEndpoint(path: string): string {
  return normalizedApiBaseUrl.includes('://api:')
    ? path
    : `${normalizedApiBaseUrl}${path}`
}

export async function runTransitSearch(
  req: TransitSearchRequest
): Promise<TransitSearchResponse> {
  const response = await fetch(apiEndpoint('/api/tess/transit-search'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(req)
  })

  if (!response.ok) {
    const errorBody = (await response.json().catch(() => null)) as
      | { detail?: string; error?: string }
      | null
    throw new Error(
      errorBody?.detail || errorBody?.error || 'Failed to run transit search'
    )
  }

  return (await response.json()) as TransitSearchResponse
}
