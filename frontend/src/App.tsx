import { useEffect, useMemo, useState } from 'react'

import { loadContext, runSimulation } from './api/schedulerApi'
import { ControlPanel } from './components/ControlPanel'
import { InsightPanel } from './components/InsightPanel'
import { SimulationMap } from './components/SimulationMap'
import { toDatetimeLocalValue } from './lib/format'
import type {
  CompareMode,
  PathType,
  SchedulerContext,
  SimulationControls,
  SimulationRequest,
  SimulationResponse,
} from './lib/types'
import './App.css'

function buildInitialControls(context: SchedulerContext): SimulationControls {
  return {
    priority: 'medium',
    alpha: 1,
    beta: 1,
    gamma: 1,
    time: toDatetimeLocalValue(context.default_time),
  }
}

function createRequestPayload(
  controls: SimulationControls,
  context: SchedulerContext,
): SimulationRequest {
  return {
    priority: controls.priority,
    alpha: controls.alpha,
    beta: controls.beta,
    gamma: controls.gamma,
    time: controls.time,
    latency_sensitivity: 0.55,
    workload_size: context.default_workload_size,
    origin_city: context.origin_city,
    top_k: 4,
  }
}

function App() {
  const [context, setContext] = useState<SchedulerContext | null>(null)
  const [controls, setControls] = useState<SimulationControls | null>(null)
  const [submittedControls, setSubmittedControls] = useState<SimulationControls | null>(null)
  const [simulation, setSimulation] = useState<SimulationResponse | null>(null)
  const [compareMode, setCompareMode] = useState<CompareMode>('candidates')
  const [highlightedPathType, setHighlightedPathType] = useState<PathType | null>(null)
  const [loadingContext, setLoadingContext] = useState(true)
  const [loadingSimulation, setLoadingSimulation] = useState(false)
  const [contextError, setContextError] = useState<string | null>(null)
  const [simulationError, setSimulationError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function initialize() {
      try {
        setLoadingContext(true)
        const nextContext = await loadContext()
        if (cancelled) {
          return
        }
        setContext(nextContext)
        const nextControls = buildInitialControls(nextContext)
        setControls(nextControls)
        setSubmittedControls(nextControls)
      } catch (error) {
        if (!cancelled) {
          setContextError(error instanceof Error ? error.message : 'Unable to load scheduler context.')
        }
      } finally {
        if (!cancelled) {
          setLoadingContext(false)
        }
      }
    }

    void initialize()

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!context || !submittedControls) {
      return
    }

    const activeControls = submittedControls
    const activeContext = context
    let cancelled = false

    async function executeSimulation() {
      try {
        setLoadingSimulation(true)
        setSimulationError(null)
        const result = await runSimulation(createRequestPayload(activeControls, activeContext))
        if (cancelled) {
          return
        }
        setSimulation(result)
        setHighlightedPathType(result.selected_path?.type ?? result.paths[0]?.type ?? null)
      } catch (error) {
        if (!cancelled) {
          setSimulationError(error instanceof Error ? error.message : 'Simulation request failed.')
        }
      } finally {
        if (!cancelled) {
          setLoadingSimulation(false)
        }
      }
    }

    void executeSimulation()

    return () => {
      cancelled = true
    }
  }, [context, submittedControls])

  const selectedPathType = useMemo<PathType | null>(() => {
    if (highlightedPathType) {
      return highlightedPathType
    }
    return simulation?.selected_path?.type ?? null
  }, [highlightedPathType, simulation?.selected_path?.type])

  if (loadingContext) {
    return (
      <main className="app-shell app-shell--centered">
        <div className="loading-state">
          <p className="loading-state__eyebrow">Connecting</p>
          <h1>Loading sustainability routing context…</h1>
        </div>
      </main>
    )
  }

  if (contextError || !context || !controls) {
    return (
      <main className="app-shell app-shell--centered">
        <div className="error-state">
          <p className="loading-state__eyebrow">Backend unavailable</p>
          <h1>React frontend could not reach the scheduling API.</h1>
          <p>{contextError ?? 'Start `python3 -m uvicorn src.api:app --reload` and try again.'}</p>
        </div>
      </main>
    )
  }

  return (
    <main className="app-shell">
      <header className="hero-header">
        <div>
          <p className="hero-header__eyebrow">Interactive sustainability-aware workload scheduler</p>
          <h1>See where the workload goes, why it moves, and what it changes.</h1>
          <p className="hero-header__copy">
            This simulation routes a workload across candidate data centers and explains the carbon,
            water, and latency trade-offs behind the recommendation.
          </p>
        </div>
        <div className="hero-header__status">
          <span className="status-dot" />
          FastAPI simulation backend connected
        </div>
      </header>

      {simulationError ? (
        <div className="inline-error">
          <strong>Simulation error:</strong> {simulationError}
        </div>
      ) : null}

      <section className="map-stage">
        <div className="map-stage__header">
          <div>
            <p className="map-stage__eyebrow">Main map</p>
            <h2>Routing decision space</h2>
          </div>
          <p className="map-stage__legend">
            Green routes optimize carbon, blue routes optimize water, gold routes balance objectives,
            and red routes favor latency.
          </p>
        </div>

        <SimulationMap
          nodes={simulation?.nodes ?? []}
          paths={simulation?.paths ?? []}
          selectedPathType={selectedPathType}
          compareMode={compareMode}
          recommendedPath={simulation?.selected_path ?? null}
          baselinePath={simulation?.baseline_path ?? null}
        />
      </section>

      <section className="bottom-layout">
        <ControlPanel
          controls={controls}
          context={context}
          compareMode={compareMode}
          isLoading={loadingSimulation}
          onControlsChange={setControls}
          onCompareModeChange={setCompareMode}
          onRun={() => setSubmittedControls(controls)}
        />

        <InsightPanel
          simulation={simulation}
          highlightedPathType={selectedPathType}
          onHighlightPath={setHighlightedPathType}
        />
      </section>
    </main>
  )
}

export default App
