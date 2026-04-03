import { useEffect, useMemo, useRef } from 'react'
import type { Feature, FeatureCollection, LineString, Point } from 'geojson'
import maplibregl, { LngLatBounds, type Map as MapLibreMap, type Popup } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

import type {
  BatchFlow,
  CompareMode,
  PathCandidate,
  PathType,
  SimulationMode,
  SimulationNode,
} from '../lib/types'

const MAP_STYLE_URL = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'
const NODE_SOURCE_ID = 'simulation-nodes'
const PATH_SOURCE_ID = 'simulation-paths'
const NODE_LAYER_ID = 'simulation-node-circles'
const PATH_LAYER_ID = 'simulation-path-lines'
const UNITED_STATES_BOUNDS: [[number, number], [number, number]] = [
  [-127.5, 23.0],
  [-65.0, 50.5],
]
const UNITED_STATES_CENTER: [number, number] = [-98.5, 39.8]
const UNITED_STATES_MIN_ZOOM = 3.1

const PATH_TYPE_COLORS: Record<PathType, string> = {
  balanced: '#d39d1a',
  low_carbon: '#2f9e63',
  low_water: '#2f6fe4',
  low_latency: '#cf2f2f',
  baseline: '#677282',
}

const FLOW_SCENARIO_COLORS = {
  optimized: '#1f8f57',
  baseline: '#7b8796',
} as const

type NodeProperties = {
  city: string
  state?: string
  avg_co2_intensity: number
  avg_wue: number
  avg_scarcity: number
  records: number
  has_data_center: number
}

type LineProperties = {
  route: string
  typeLabel: string
  co2: number
  water: number
  latency: number
  score: number
  lineColor: string
  lineWidth: number
  arrow: string
  jobs?: number
  scenario?: string
}

interface SimulationMapProps {
  mode: SimulationMode
  nodes: SimulationNode[]
  paths: PathCandidate[]
  flows: BatchFlow[]
  baselineFlows: BatchFlow[]
  selectedPathType: PathType | null
  selectedFlowId: string | null
  compareMode: CompareMode
  recommendedPath: PathCandidate | null
  baselinePath: PathCandidate | null
}

function buildNodeCollection(nodes: SimulationNode[]): FeatureCollection<Point, NodeProperties> {
  return {
    type: 'FeatureCollection',
    features: nodes.map<Feature<Point, NodeProperties>>((node) => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [node.longitude, node.latitude],
      },
      properties: {
        city: node.city,
        state: node.state,
        avg_co2_intensity: node.avg_co2_intensity,
        avg_wue: node.avg_wue,
        avg_scarcity: node.avg_scarcity,
        records: node.records,
        has_data_center: node.has_data_center,
      },
    })),
  }
}

function buildDisplayedPaths(
  compareMode: CompareMode,
  paths: PathCandidate[],
  recommendedPath: PathCandidate | null,
  baselinePath: PathCandidate | null,
): PathCandidate[] {
  if (compareMode === 'baseline') {
    return [baselinePath, recommendedPath].filter((path): path is PathCandidate => Boolean(path))
  }
  return paths
}

function buildDisplayedFlows(
  compareMode: CompareMode,
  flows: BatchFlow[],
  baselineFlows: BatchFlow[],
): BatchFlow[] {
  if (compareMode === 'baseline') {
    return [...baselineFlows, ...flows]
  }
  return flows
}

function buildPathCollection(
  paths: PathCandidate[],
  selectedPathType: PathType | null,
): FeatureCollection<LineString, LineProperties> {
  return {
    type: 'FeatureCollection',
    features: paths.map<Feature<LineString, LineProperties>>((path) => {
      const isHighlighted = selectedPathType === null || selectedPathType === path.type
      return {
        type: 'Feature',
        geometry: {
          type: 'LineString',
          coordinates: [
            [path.origin_longitude, path.origin_latitude],
            [path.destination_longitude, path.destination_latitude],
          ],
        },
        properties: {
          route: path.route,
          typeLabel: path.type.replace('_', ' '),
          co2: path.co2,
          water: path.water_liters,
          latency: path.latency,
          score: path.score,
          lineColor: PATH_TYPE_COLORS[path.type],
          lineWidth: isHighlighted ? 6.5 : 3.5,
          arrow: '▶',
        },
      }
    }),
  }
}

function buildFlowCollection(
  flows: BatchFlow[],
  selectedFlowId: string | null,
): FeatureCollection<LineString, LineProperties> {
  const jobs = flows.map((flow) => flow.jobs)
  const minJobs = jobs.length > 0 ? Math.min(...jobs) : 1
  const maxJobs = jobs.length > 0 ? Math.max(...jobs) : 1
  const span = Math.max(1, maxJobs - minJobs)

  return {
    type: 'FeatureCollection',
    features: flows.map<Feature<LineString, LineProperties>>((flow) => {
      const emphasized = selectedFlowId === null || selectedFlowId === flow.flow_id
      const width = 3.5 + ((flow.jobs - minJobs) / span) * 6.5 + (emphasized ? 1.5 : 0)
      return {
        type: 'Feature',
        geometry: {
          type: 'LineString',
          coordinates: [
            [flow.origin_longitude, flow.origin_latitude],
            [flow.destination_longitude, flow.destination_latitude],
          ],
        },
        properties: {
          route: flow.route,
          typeLabel: flow.scenario === 'optimized' ? 'Optimized batch flow' : 'Baseline batch flow',
          co2: flow.co2,
          water: flow.water_liters,
          latency: flow.latency,
          score: flow.score,
          jobs: flow.jobs,
          scenario: flow.scenario,
          lineColor: FLOW_SCENARIO_COLORS[flow.scenario],
          lineWidth: width,
          arrow: '▶',
        },
      }
    }),
  }
}

function extent(values: number[]): [number, number] {
  if (values.length === 0) {
    return [0, 1]
  }
  const min = Math.min(...values)
  const max = Math.max(...values)
  return min === max ? [min, min + 1] : [min, max]
}

function nodePopupHtml(properties: NodeProperties): string {
  return `
    <div class="map-popup">
      <div class="map-popup__title">${properties.city}</div>
      <div>CO2 intensity: ${properties.avg_co2_intensity.toFixed(3)} kg/kWh</div>
      <div>WUE: ${properties.avg_wue.toFixed(3)} L/kWh</div>
      <div>Scarcity: ${properties.avg_scarcity.toFixed(3)}</div>
    </div>
  `
}

function linePopupHtml(properties: LineProperties): string {
  const batchLine = typeof properties.jobs === 'number' ? `<div>Jobs: ${properties.jobs}</div>` : ''
  return `
    <div class="map-popup">
      <div class="map-popup__title">${properties.route}</div>
      <div>${properties.typeLabel}</div>
      ${batchLine}
      <div>CO2: ${properties.co2.toFixed(2)} kg</div>
      <div>Water: ${properties.water.toFixed(2)} L</div>
      <div>Latency: ${properties.latency.toFixed(1)} ms</div>
    </div>
  `
}

export function SimulationMap({
  mode,
  nodes,
  paths,
  flows,
  baselineFlows,
  selectedPathType,
  selectedFlowId,
  compareMode,
  recommendedPath,
  baselinePath,
}: SimulationMapProps) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const popupRef = useRef<Popup | null>(null)
  const hasFittedRef = useRef(false)

  const displayedPaths = useMemo(
    () => buildDisplayedPaths(compareMode, paths, recommendedPath, baselinePath),
    [baselinePath, compareMode, paths, recommendedPath],
  )
  const displayedFlows = useMemo(
    () => buildDisplayedFlows(compareMode, flows, baselineFlows),
    [baselineFlows, compareMode, flows],
  )
  const nodeCollection = useMemo(() => buildNodeCollection(nodes), [nodes])
  const lineCollection = useMemo(
    () =>
      mode === 'batch'
        ? buildFlowCollection(displayedFlows, selectedFlowId)
        : buildPathCollection(displayedPaths, selectedPathType),
    [displayedFlows, displayedPaths, mode, selectedFlowId, selectedPathType],
  )

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) {
      return
    }

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: MAP_STYLE_URL,
      center: UNITED_STATES_CENTER,
      zoom: 3.45,
      maxBounds: UNITED_STATES_BOUNDS,
      minZoom: UNITED_STATES_MIN_ZOOM,
      attributionControl: false,
    })

    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right')
    const popup = new maplibregl.Popup({
      closeButton: false,
      closeOnClick: false,
      offset: 18,
      className: 'simulation-map-popup',
    })

    mapRef.current = map
    popupRef.current = popup

    map.on('load', () => {
      map.addSource(NODE_SOURCE_ID, {
        type: 'geojson',
        data: nodeCollection,
      })
      map.addSource(PATH_SOURCE_ID, {
        type: 'geojson',
        data: lineCollection,
      })

      map.addLayer({
        id: 'simulation-scarcity-heat',
        type: 'heatmap',
        source: NODE_SOURCE_ID,
        paint: {
          'heatmap-weight': ['coalesce', ['get', 'avg_scarcity'], 0],
          'heatmap-intensity': 0.7,
          'heatmap-radius': 38,
          'heatmap-opacity': 0.38,
          'heatmap-color': [
            'interpolate',
            ['linear'],
            ['heatmap-density'],
            0,
            'rgba(227, 242, 235, 0)',
            0.4,
            'rgba(208, 230, 201, 0.45)',
            1,
            'rgba(112, 177, 116, 0.75)',
          ],
        },
      })

      map.addLayer({
        id: PATH_LAYER_ID,
        type: 'line',
        source: PATH_SOURCE_ID,
        layout: {
          'line-cap': 'round',
          'line-join': 'round',
        },
        paint: {
          'line-color': ['to-color', ['get', 'lineColor']],
          'line-width': ['coalesce', ['get', 'lineWidth'], 4],
          'line-opacity': [
            'case',
            ['==', ['get', 'scenario'], 'baseline'],
            0.52,
            0.94,
          ],
        },
      })

      map.addLayer({
        id: 'simulation-path-arrows',
        type: 'symbol',
        source: PATH_SOURCE_ID,
        layout: {
          'symbol-placement': 'line',
          'symbol-spacing': 110,
          'text-field': ['get', 'arrow'],
          'text-size': 14,
          'text-keep-upright': false,
        },
        paint: {
          'text-color': ['to-color', ['get', 'lineColor']],
          'text-opacity': 0.88,
        },
      })

      map.addLayer({
        id: NODE_LAYER_ID,
        type: 'circle',
        source: NODE_SOURCE_ID,
        paint: {
          'circle-color': '#5ca86d',
          'circle-radius': 12,
          'circle-stroke-width': ['case', ['==', ['get', 'has_data_center'], 1], 2.5, 1.25],
          'circle-stroke-color': [
            'case',
            ['==', ['get', 'has_data_center'], 1],
            '#123c2b',
            'rgba(255,255,255,0.85)',
          ],
          'circle-opacity': 0.92,
        },
      })

      const enterLayer = () => {
        map.getCanvas().style.cursor = 'pointer'
      }
      const leaveLayer = () => {
        map.getCanvas().style.cursor = ''
        popup.remove()
      }

      map.on('mouseenter', NODE_LAYER_ID, enterLayer)
      map.on('mouseleave', NODE_LAYER_ID, leaveLayer)
      map.on('mousemove', NODE_LAYER_ID, (event) => {
        const feature = event.features?.[0]
        if (!feature || feature.geometry.type !== 'Point') {
          return
        }
        popup
          .setLngLat(feature.geometry.coordinates as [number, number])
          .setHTML(nodePopupHtml(feature.properties as unknown as NodeProperties))
          .addTo(map)
      })

      map.on('mouseenter', PATH_LAYER_ID, enterLayer)
      map.on('mouseleave', PATH_LAYER_ID, leaveLayer)
      map.on('mousemove', PATH_LAYER_ID, (event) => {
        const feature = event.features?.[0]
        if (!feature || !event.lngLat) {
          return
        }
        popup
          .setLngLat(event.lngLat)
          .setHTML(linePopupHtml(feature.properties as unknown as LineProperties))
          .addTo(map)
      })
    })

    return () => {
      popup.remove()
      map.remove()
      mapRef.current = null
      popupRef.current = null
    }
  }, [lineCollection, nodeCollection])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.isStyleLoaded()) {
      return
    }

    const nodeSource = map.getSource(NODE_SOURCE_ID) as maplibregl.GeoJSONSource | undefined
    const pathSource = map.getSource(PATH_SOURCE_ID) as maplibregl.GeoJSONSource | undefined
    nodeSource?.setData(nodeCollection)
    pathSource?.setData(lineCollection)

    const [co2Min, co2Max] = extent(nodes.map((node) => Number(node.avg_co2_intensity)))
    const [wueMin, wueMax] = extent(nodes.map((node) => Number(node.avg_wue)))

    map.setPaintProperty(NODE_LAYER_ID, 'circle-color', [
      'interpolate',
      ['linear'],
      ['coalesce', ['get', 'avg_co2_intensity'], co2Min],
      co2Min,
      '#2ca86b',
      (co2Min + co2Max) / 2,
      '#e7ad2e',
      co2Max,
      '#d44833',
    ])

    map.setPaintProperty(NODE_LAYER_ID, 'circle-radius', [
      'interpolate',
      ['linear'],
      ['coalesce', ['get', 'avg_wue'], wueMin],
      wueMin,
      8,
      wueMax,
      24,
    ])

    if (nodes.length === 0) {
      return
    }

    const bounds = new LngLatBounds()
    nodes.forEach((node) => {
      bounds.extend([node.longitude, node.latitude])
    })
    if (!bounds.isEmpty() && (!hasFittedRef.current || lineCollection.features.length > 0)) {
      map.fitBounds(bounds, {
        padding: {
          top: 70,
          bottom: 70,
          left: 70,
          right: 70,
        },
        maxZoom: 5.2,
        duration: 900,
      })
      if (map.getZoom() < UNITED_STATES_MIN_ZOOM) {
        map.setZoom(UNITED_STATES_MIN_ZOOM)
      }
      hasFittedRef.current = true
    }
  }, [lineCollection, nodeCollection, nodes])

  return <div className="simulation-map" ref={mapContainerRef} />
}
