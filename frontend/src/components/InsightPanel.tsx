import { useEffect, useState } from 'react'
import type {
  BatchFlow,
  BatchSimulationResponse,
  ExplanationResponse,
  PathCandidate,
  PathType,
  SimulationMode,
  SimulationResponse,
} from '../lib/types'
import { formatLatency, formatPercent, formatDatetimeLabel } from '../lib/format'
import { fetchExplanation } from '../api/schedulerApi'

const PATH_LABELS: Record<PathType, string> = {
  balanced: 'Balanced',
  low_carbon: 'Low carbon',
  low_water: 'Low water',
  low_latency: 'Low latency',
  baseline: 'Baseline',
}

interface InsightPanelProps {
  mode: SimulationMode
  simulation: SimulationResponse | null
  batchSimulation: BatchSimulationResponse | null
  highlightedPathType: PathType | null
  highlightedFlowId: string | null
  onHighlightPath: (pathType: PathType) => void
  onHighlightFlow: (flowId: string) => void
}

const TARGET_UNIT: Record<'co2' | 'wue', string> = {
  co2: 'kg CO₂/kWh',
  wue: 'L/kWh (WUE)',
}

function ShapChart({ city, time }: { city: string; time: string }) {
  const [co2Data, setCo2Data] = useState<ExplanationResponse | null>(null)
  const [wueData, setWueData] = useState<ExplanationResponse | null>(null)
  const [activeTarget, setActiveTarget] = useState<'co2' | 'wue'>('co2')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setCo2Data(null)
    setWueData(null)

    Promise.all([
      fetchExplanation(city, 'co2', time),
      fetchExplanation(city, 'wue', time),
    ])
      .then(([co2, wue]) => {
        if (cancelled) return
        setCo2Data(co2)
        setWueData(wue)
      })
      .catch(() => { /* silently ignore — SHAP is optional enrichment */ })
      .finally(() => { if (!cancelled) setLoading(false) })

    return () => { cancelled = true }
  }, [city, time])

  const data = activeTarget === 'co2' ? co2Data : wueData
  if (loading) {
    return <p className="shap-loading">Computing feature importance…</p>
  }
  if (!data) return null

  const maxAbs = Math.max(...data.features.map(f => Math.abs(f.shap_value)), 0.0001)

  return (
    <div className="panel__subsection">
      <div className="panel__subheader">
        <h3>Why this prediction?</h3>
        <div className="shap-toggle">
          <button
            type="button"
            className={activeTarget === 'co2' ? 'is-active' : ''}
            onClick={() => setActiveTarget('co2')}
          >
            CO₂
          </button>
          <button
            type="button"
            className={activeTarget === 'wue' ? 'is-active' : ''}
            onClick={() => setActiveTarget('wue')}
          >
            WUE
          </button>
        </div>
      </div>
      <p className="shap-legend">
        Predicted {TARGET_UNIT[activeTarget]}: <strong>{data.prediction.toFixed(4)}</strong>
        {' '}(baseline: {data.base_value.toFixed(4)})
      </p>
      <div className="shap-bars">
        {data.features.map(f => {
          const pct = (Math.abs(f.shap_value) / maxAbs) * 100
          const positive = f.shap_value > 0
          return (
            <div key={f.raw_name} className="shap-row">
              <span className="shap-row__label" title={f.raw_name}>{f.name}</span>
              <div className="shap-row__track">
                <div
                  className={`shap-row__bar ${positive ? 'shap-row__bar--up' : 'shap-row__bar--down'}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className={`shap-row__value ${positive ? 'shap-value--up' : 'shap-value--down'}`}>
                {positive ? '+' : ''}{f.shap_value.toFixed(4)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function pathLabel(path: PathCandidate | null): string {
  if (!path) {
    return 'No route available'
  }
  return path.route
}

function renderSingleView(
  simulation: SimulationResponse,
  highlightedPathType: PathType | null,
  onHighlightPath: (pathType: PathType) => void,
) {
  const { selected_path: selectedPath, baseline_path: baselinePath, metrics, insight, paths } = simulation
  const comparisonPaths = paths.slice(0, 3)

  return (
    <>
      <div className="route-summary">
        <div className="route-summary__card">
          <span className="route-summary__label">Best path</span>
          <strong>{pathLabel(selectedPath)}</strong>
        </div>
        <div className="route-summary__card route-summary__card--muted">
          <span className="route-summary__label">Baseline</span>
          <strong>{pathLabel(baselinePath)}</strong>
        </div>
      </div>

      <div className="metric-grid">
        <article className="metric-card">
          <span className="metric-card__label">CO2 reduction</span>
          <strong>{formatPercent(metrics.co2_reduction_pct)}</strong>
          <p>Compared with the naive route.</p>
        </article>
        <article className="metric-card">
          <span className="metric-card__label">Water reduction</span>
          <strong>{formatPercent(metrics.water_reduction_pct)}</strong>
          <p>Lower water stress and WUE impact.</p>
        </article>
        <article className="metric-card">
          <span className="metric-card__label">Latency impact</span>
          <strong>{formatLatency(metrics.latency_delta_ms)}</strong>
          <p>Positive means slower than baseline.</p>
        </article>
      </div>

      <div className="insight-callout">
        <span className="insight-callout__eyebrow">{insight.title}</span>
        <p>{insight.summary}</p>
      </div>

      <div className="panel__subsection">
        <div className="panel__subheader">
          <h3>Top route options</h3>
          <span>{formatDatetimeLabel(simulation.time)}</span>
        </div>
        <div className="path-table">
          <div className="path-table__head">
            <span>Path</span>
            <span>Route</span>
            <span>CO2</span>
            <span>WUE</span>
            <span>Latency</span>
          </div>
          {comparisonPaths.map((path) => (
            <button
              type="button"
              key={path.type}
              className={`path-row ${highlightedPathType === path.type ? 'path-row--active' : ''}`}
              onClick={() => onHighlightPath(path.type)}
            >
              <span className={`path-tag path-tag--${path.type}`}>{PATH_LABELS[path.type]}</span>
              <span className="path-row__route">{path.route}</span>
              <span>{path.co2.toFixed(1)} kg</span>
              <span>{path.wue.toFixed(3)}</span>
              <span>{path.latency.toFixed(1)} ms</span>
            </button>
          ))}
        </div>
      </div>

      {selectedPath && (
        <ShapChart city={selectedPath.assigned_city} time={simulation.time} />
      )}
    </>
  )
}

function renderBatchRow(
  flow: BatchFlow,
  highlightedFlowId: string | null,
  onHighlightFlow: (flowId: string) => void,
) {
  return (
    <button
      type="button"
      key={flow.flow_id}
      className={`path-row ${highlightedFlowId === flow.flow_id ? 'path-row--active' : ''}`}
      onClick={() => onHighlightFlow(flow.flow_id)}
    >
      <span className={`path-tag ${flow.scenario === 'optimized' ? 'path-tag--optimized' : 'path-tag--baseline'}`}>
        {flow.jobs} jobs
      </span>
      <span className="path-row__route">{flow.route}</span>
      <span>{flow.co2.toFixed(1)} kg</span>
      <span>{flow.water_liters.toFixed(1)} L</span>
      <span>{flow.latency.toFixed(1)} ms</span>
    </button>
  )
}

function renderBatchView(
  simulation: BatchSimulationResponse,
  highlightedFlowId: string | null,
  onHighlightFlow: (flowId: string) => void,
) {
  const { selected_flow: selectedFlow, optimized_summary, comparison, insight, optimized_flows } = simulation
  const topFlows = optimized_flows.slice(0, 6)
  const baselineCoverage = simulation.baseline_summary.coverage * 100
  const optimizedCoverage = optimized_summary.coverage * 100

  return (
    <>
      <div className="route-summary">
        <div className="route-summary__card">
          <span className="route-summary__label">Busiest optimized flow</span>
          <strong>{selectedFlow?.route ?? 'No active route'}</strong>
        </div>
        <div className="route-summary__card route-summary__card--muted">
          <span className="route-summary__label">Jobs scheduled</span>
          <strong>
            {Math.round(optimized_summary.scheduled_jobs)} / {Math.round(optimized_summary.total_jobs)}
          </strong>
        </div>
      </div>

      <div className="metric-grid metric-grid--batch">
        <article className="metric-card">
          <span className="metric-card__label">CO2 reduction</span>
          <strong>{formatPercent(comparison.co2_reduction_pct)}</strong>
          <p>Optimized batch versus baseline placement.</p>
        </article>
        <article className="metric-card">
          <span className="metric-card__label">Water reduction</span>
          <strong>{formatPercent(comparison.water_reduction_pct)}</strong>
          <p>Lower total water impact across the batch.</p>
        </article>
        <article className="metric-card">
          <span className="metric-card__label">Latency impact</span>
          <strong>{formatLatency(comparison.latency_delta_ms)}</strong>
          <p>Weighted average route latency versus baseline.</p>
        </article>
        <article className="metric-card">
          <span className="metric-card__label">Coverage</span>
          <strong>{optimizedCoverage.toFixed(0)}%</strong>
          <p>Baseline was {baselineCoverage.toFixed(0)}% for the same batch.</p>
        </article>
      </div>

      <div className="insight-callout">
        <span className="insight-callout__eyebrow">{insight.title}</span>
        <p>{insight.summary}</p>
      </div>

      <div className="panel__subsection">
        <div className="panel__subheader">
          <h3>Top batch flows</h3>
          <span>
            {formatDatetimeLabel(simulation.time)} to {formatDatetimeLabel(simulation.window_end)}
          </span>
        </div>
        <div className="path-table">
          <div className="path-table__head">
            <span>Flow</span>
            <span>Route</span>
            <span>CO2</span>
            <span>Water</span>
            <span>Latency</span>
          </div>
          {topFlows.map((flow) => renderBatchRow(flow, highlightedFlowId, onHighlightFlow))}
        </div>
      </div>
    </>
  )
}

export function InsightPanel({
  mode,
  simulation,
  batchSimulation,
  highlightedPathType,
  highlightedFlowId,
  onHighlightPath,
  onHighlightFlow,
}: InsightPanelProps) {
  if (mode === 'batch') {
    if (!batchSimulation) {
      return (
        <section className="panel panel--insight">
          <div className="panel__header">
            <div>
              <p className="panel__eyebrow">Batch result</p>
              <h2 className="panel__title">Awaiting batch simulation</h2>
            </div>
          </div>
          <p className="panel__copy">
            Run the batch simulation to see how many jobs move, which routes carry the heaviest load,
            and how the optimized schedule compares with the baseline.
          </p>
        </section>
      )
    }

    return (
      <section className="panel panel--insight">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Batch result and insight</p>
            <h2 className="panel__title">How the system coordinates many jobs at once</h2>
          </div>
          <div className="panel__status">Optimized batch</div>
        </div>
        {renderBatchView(batchSimulation, highlightedFlowId, onHighlightFlow)}
      </section>
    )
  }

  if (!simulation || !simulation.selected_path) {
    return (
      <section className="panel panel--insight">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Result</p>
            <h2 className="panel__title">Awaiting simulation</h2>
          </div>
        </div>
        <p className="panel__copy">
          Run the simulation to see the recommended route, environmental trade-offs, and the top candidate paths.
        </p>
      </section>
    )
  }

  return (
    <section className="panel panel--insight">
      <div className="panel__header">
        <div>
          <p className="panel__eyebrow">Result and insight</p>
          <h2 className="panel__title">Why the system sends the job there</h2>
        </div>
        <div className="panel__status">Recommended</div>
      </div>
      {renderSingleView(simulation, highlightedPathType, onHighlightPath)}
    </section>
  )
}
