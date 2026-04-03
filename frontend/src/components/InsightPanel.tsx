import type { PathCandidate, PathType, SimulationResponse } from '../lib/types'
import { formatLatency, formatPercent, formatDatetimeLabel } from '../lib/format'

const PATH_LABELS: Record<PathType, string> = {
  balanced: 'Balanced',
  low_carbon: 'Low carbon',
  low_water: 'Low water',
  low_latency: 'Low latency',
  baseline: 'Baseline',
}

interface InsightPanelProps {
  simulation: SimulationResponse | null
  highlightedPathType: PathType | null
  onHighlightPath: (pathType: PathType) => void
}

function pathLabel(path: PathCandidate | null): string {
  if (!path) {
    return 'No route available'
  }
  return path.route
}

export function InsightPanel({
  simulation,
  highlightedPathType,
  onHighlightPath,
}: InsightPanelProps) {
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

  const { selected_path: selectedPath, baseline_path: baselinePath, metrics, insight, paths } = simulation
  const comparisonPaths = paths.slice(0, 3)

  return (
    <section className="panel panel--insight">
      <div className="panel__header">
        <div>
          <p className="panel__eyebrow">Result and insight</p>
          <h2 className="panel__title">Why the system sends the job there</h2>
        </div>
        <div className="panel__status">Recommended</div>
      </div>

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
    </section>
  )
}
