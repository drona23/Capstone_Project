import type { CompareMode, SchedulerContext, SimulationControls } from '../lib/types'
import { toDatetimeLocalValue } from '../lib/format'

interface ControlPanelProps {
  controls: SimulationControls
  context: SchedulerContext
  compareMode: CompareMode
  isLoading: boolean
  onControlsChange: (next: SimulationControls) => void
  onCompareModeChange: (mode: CompareMode) => void
  onRun: () => void
}

function updateControl<K extends keyof SimulationControls>(
  controls: SimulationControls,
  key: K,
  value: SimulationControls[K],
): SimulationControls {
  return { ...controls, [key]: value }
}

export function ControlPanel({
  controls,
  context,
  compareMode,
  isLoading,
  onControlsChange,
  onCompareModeChange,
  onRun,
}: ControlPanelProps) {
  return (
    <section className="panel panel--controls">
      <div className="panel__header">
        <div>
          <p className="panel__eyebrow">Control panel</p>
          <h2 className="panel__title">Shape the routing decision</h2>
        </div>
        <div className="panel__meta">
          <span>Origin</span>
          <strong>{context.origin_city}</strong>
        </div>
      </div>

      <div className="control-grid">
        <label className="field">
          <span className="field__label">Priority</span>
          <select
            value={controls.priority}
            onChange={(event) =>
              onControlsChange(updateControl(controls, 'priority', event.target.value as SimulationControls['priority']))
            }
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </label>

        <label className="field">
          <span className="field__label">Simulation time</span>
          <input
            type="datetime-local"
            value={controls.time}
            min={toDatetimeLocalValue(context.time_min)}
            max={toDatetimeLocalValue(context.time_max)}
            onChange={(event) => onControlsChange(updateControl(controls, 'time', event.target.value))}
          />
        </label>

        <div className="segmented-control">
          <button
            type="button"
            className={compareMode === 'candidates' ? 'is-active' : ''}
            onClick={() => onCompareModeChange('candidates')}
          >
            Candidates
          </button>
          <button
            type="button"
            className={compareMode === 'baseline' ? 'is-active' : ''}
            onClick={() => onCompareModeChange('baseline')}
          >
            Baseline vs recommended
          </button>
        </div>
      </div>

      <div className="slider-stack">
        <label className="slider-field">
          <div className="slider-field__header">
            <span>Carbon weight</span>
            <strong>{controls.alpha.toFixed(2)}</strong>
          </div>
          <input
            type="range"
            min="0"
            max="2"
            step="0.05"
            value={controls.alpha}
            onChange={(event) => onControlsChange(updateControl(controls, 'alpha', Number(event.target.value)))}
          />
        </label>

        <label className="slider-field">
          <div className="slider-field__header">
            <span>Water weight</span>
            <strong>{controls.beta.toFixed(2)}</strong>
          </div>
          <input
            type="range"
            min="0"
            max="2"
            step="0.05"
            value={controls.beta}
            onChange={(event) => onControlsChange(updateControl(controls, 'beta', Number(event.target.value)))}
          />
        </label>

        <label className="slider-field">
          <div className="slider-field__header">
            <span>Latency weight</span>
            <strong>{controls.gamma.toFixed(2)}</strong>
          </div>
          <input
            type="range"
            min="0"
            max="2"
            step="0.05"
            value={controls.gamma}
            onChange={(event) => onControlsChange(updateControl(controls, 'gamma', Number(event.target.value)))}
          />
        </label>
      </div>

      <div className="control-footer">
        <div className="control-footer__meta">
          <span>Workload size</span>
          <strong>{context.default_workload_size.toFixed(1)} units</strong>
        </div>
        <button type="button" className="run-button" onClick={onRun} disabled={isLoading}>
          {isLoading ? 'Running…' : 'Run simulation'}
        </button>
      </div>
    </section>
  )
}
