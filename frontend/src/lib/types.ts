export type Priority = 'low' | 'medium' | 'high'
export type CompareMode = 'candidates' | 'baseline'
export type PathType = 'balanced' | 'low_carbon' | 'low_water' | 'low_latency' | 'baseline'

export interface SchedulerContext {
  origin_city: string
  time_min: string
  time_max: string
  default_time: string
  default_workload_size: number
  priority_counts: Record<string, number>
}

export interface SimulationRequest {
  priority: Priority
  alpha: number
  beta: number
  gamma: number
  time: string
  latency_sensitivity: number
  workload_size: number
  origin_city: string
  top_k: number
}

export interface PathCandidate {
  route: string
  path: string[]
  co2: number
  wue: number
  latency: number
  score: number
  type: PathType
  scheduled_start: string
  scheduled_end: string
  origin_city: string
  assigned_city: string
  origin_latitude: number
  origin_longitude: number
  destination_latitude: number
  destination_longitude: number
  water_liters: number
}

export interface SimulationNode {
  city: string
  state?: string
  zip?: string
  avg_co2_intensity: number
  avg_wue: number
  avg_scarcity: number
  records: number
  latitude: number
  longitude: number
  has_data_center: number
}

export interface SimulationMetrics {
  co2_reduction_pct: number
  water_reduction_pct: number
  latency_delta_ms: number
  latency_delta_pct: number
}

export interface SimulationInsight {
  title: string
  summary: string
}

export interface SimulationResponse {
  time: string
  origin_city: string
  paths: PathCandidate[]
  selected_path: PathCandidate | null
  baseline_path: PathCandidate | null
  metrics: SimulationMetrics
  insight: SimulationInsight
  nodes: SimulationNode[]
}

export interface SimulationControls {
  priority: Priority
  alpha: number
  beta: number
  gamma: number
  time: string
}
