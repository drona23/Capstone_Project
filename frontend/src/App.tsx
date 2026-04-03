import { useEffect, useMemo, useState } from 'react'

import { loadContext, runBatchSimulation, runSimulation } from './api/schedulerApi'
import { ControlPanel } from './components/ControlPanel'
import { InsightPanel } from './components/InsightPanel'
import { SimulationMap } from './components/SimulationMap'
import { toDatetimeLocalValue } from './lib/format'
import type {
  BatchSimulationRequest,
  BatchSimulationResponse,
  CompareMode,
  PathType,
  SchedulerContext,
  SimulationControls,
  SimulationMode,
  SimulationRequest,
  SimulationResponse,
} from './lib/types'
import './App.css'

function buildInitialControls(context: SchedulerContext): SimulationControls {
  return {
    mode: 'batch',
    priority: 'medium',
    alpha: 1,
    beta: 1,
    gamma: 1,
    time: toDatetimeLocalValue(context.default_time),
    batchSize: context.default_batch_size,
  }
}

function createSingleRequestPayload(
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

function createBatchRequestPayload(
  controls: SimulationControls,
): BatchSimulationRequest {
  return {
    priority: controls.priority,
    alpha: controls.alpha,
    beta: controls.beta,
    gamma: controls.gamma,
    time: controls.time,
    latency_sensitivity: 0.55,
    batch_size: controls.batchSize,
  }
}

function App() {
  const [context, setContext] = useState<SchedulerContext | null>(null)
  const [controls, setControls] = useState<SimulationControls | null>(null)
  const [submittedControls, setSubmittedControls] = useState<SimulationControls | null>(null)
  const [singleSimulation, setSingleSimulation] = useState<SimulationResponse | null>(null)
  const [batchSimulation, setBatchSimulation] = useState<BatchSimulationResponse | null>(null)
  const [compareMode, setCompareMode] = useState<CompareMode>('candidates')
  const [highlightedPathType, setHighlightedPathType] = useState<PathType | null>(null)
  const [highlightedFlowId, setHighlightedFlowId] = useState<string | null>(null)
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

        if (activeControls.mode === 'batch') {
          const result = await runBatchSimulation(createBatchRequestPayload(activeControls))
          if (cancelled) {
            return
          }
          setBatchSimulation(result)
          setHighlightedFlowId(result.selected_flow?.flow_id ?? result.optimized_flows[0]?.flow_id ?? null)
          return
        }

        const result = await runSimulation(createSingleRequestPayload(activeControls, activeContext))
        if (cancelled) {
          return
        }
        setSingleSimulation(result)
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

  const activeMode = useMemo<SimulationMode>(() => {
    if (submittedControls) {
      return submittedControls.mode
    }
    return controls?.mode ?? 'batch'
  }, [controls?.mode, submittedControls])

  const selectedPathType = useMemo<PathType | null>(() => {
    if (activeMode !== 'single') {
      return null
    }
    if (highlightedPathType) {
      return highlightedPathType
    }
    return singleSimulation?.selected_path?.type ?? null
  }, [activeMode, highlightedPathType, singleSimulation?.selected_path?.type])

  const selectedFlowId = useMemo<string | null>(() => {
    if (activeMode !== 'batch') {
      return null
    }
    if (highlightedFlowId) {
      return highlightedFlowId
    }
    return batchSimulation?.selected_flow?.flow_id ?? batchSimulation?.optimized_flows[0]?.flow_id ?? null
  }, [activeMode, batchSimulation?.optimized_flows, batchSimulation?.selected_flow?.flow_id, highlightedFlowId])

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
          <h1>Simulate one decision or watch 50–200 jobs move as a coordinated batch.</h1>
          <p className="hero-header__copy">
            Switch between a single routing recommendation and a multi-job stress test to see how
            carbon, water, and latency trade-offs change when the system schedules many workloads
            at once.
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
            <h2>{activeMode === 'batch' ? 'Batch routing flows' : 'Routing decision space'}</h2>
          </div>
          <p className="map-stage__legend">
            {activeMode === 'batch'
              ? 'Green bands show the optimized batch schedule. Toggle the baseline overlay to compare how traffic shifts across data centers.'
              : 'Green routes optimize carbon, blue routes optimize water, gold routes balance objectives, and red routes favor latency.'}
          </p>
        </div>

        <SimulationMap
          mode={activeMode}
          nodes={activeMode === 'batch' ? batchSimulation?.nodes ?? [] : singleSimulation?.nodes ?? []}
          paths={singleSimulation?.paths ?? []}
          flows={batchSimulation?.optimized_flows ?? []}
          baselineFlows={batchSimulation?.baseline_flows ?? []}
          selectedPathType={selectedPathType}
          selectedFlowId={selectedFlowId}
          compareMode={compareMode}
          recommendedPath={singleSimulation?.selected_path ?? null}
          baselinePath={singleSimulation?.baseline_path ?? null}
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
          onRun={() => setSubmittedControls({ ...controls })}
        />

        <InsightPanel
          mode={activeMode}
          simulation={singleSimulation}
          batchSimulation={batchSimulation}
          highlightedPathType={selectedPathType}
          highlightedFlowId={selectedFlowId}
          onHighlightPath={setHighlightedPathType}
          onHighlightFlow={setHighlightedFlowId}
        />
      </section>
    </main>
  )
}

export default App
