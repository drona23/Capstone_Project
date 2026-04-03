import type {
  BatchFlow,
  BatchSimulationResponse,
  PathCandidate,
  PathType,
  SimulationMode,
  SimulationResponse,
} from '../lib/types'
import { formatLatency, formatPercent, formatDatetimeLabel } from '../lib/format'

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
