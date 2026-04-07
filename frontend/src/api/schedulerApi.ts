import type {
  BatchSimulationRequest,
  BatchSimulationResponse,
  ExplanationResponse,
  SchedulerContext,
  SimulationRequest,
  SimulationResponse,
} from '../lib/types'

const rawBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()
const API_BASE_URL = rawBaseUrl ? rawBaseUrl.replace(/\/$/, '') : '/api'

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })

  if (!response.ok) {
    const body = await response.text()
    throw new Error(body || `Request failed with status ${response.status}`)
  }

  return response.json() as Promise<T>
}

export function loadContext(): Promise<SchedulerContext> {
  return fetchJson<SchedulerContext>('/context')
}

export function runSimulation(payload: SimulationRequest): Promise<SimulationResponse> {
  return fetchJson<SimulationResponse>('/simulate', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function runBatchSimulation(
  payload: BatchSimulationRequest,
): Promise<BatchSimulationResponse> {
  return fetchJson<BatchSimulationResponse>('/simulate-batch', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function fetchExplanation(
  city: string,
  target: 'co2' | 'wue',
  time: string,
): Promise<ExplanationResponse> {
  const params = new URLSearchParams({ city, target, time })
  return fetchJson<ExplanationResponse>(`/explain?${params.toString()}`)
}
