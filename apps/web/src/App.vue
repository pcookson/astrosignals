<template>
  <main>
    <h1>AstroSignals</h1>

    <label class="source">
      Data source
      <select v-model="source">
        <option value="tess">TESS/Kepler</option>
        <option value="ztf">ZTF</option>
      </select>
    </label>

    <form v-if="source === 'tess'" @submit.prevent="ingestTessAndPlot" class="form">
      <label>
        Target
        <input v-model="target" type="text" />
      </label>

      <label>
        Mission
        <select v-model="mission">
          <option value="TESS">TESS</option>
          <option value="Kepler">Kepler</option>
        </select>
      </label>

      <label>
        Author
        <input v-model="author" type="text" />
      </label>

      <label>
        Sector (optional)
        <input v-model="sector" type="text" placeholder="blank = null" />
      </label>

      <button type="submit" :disabled="loading">Ingest &amp; Plot</button>
    </form>

    <form v-else @submit.prevent="ingestZtfAndPlot" class="form">
      <label>
        objectId (optional)
        <input v-model="ztfObjectId" type="text" placeholder="ZTF18abcdefg" />
      </label>

      <label>
        RA (deg, optional)
        <input v-model="ztfRa" type="number" step="any" />
      </label>

      <label>
        Dec (deg, optional)
        <input v-model="ztfDec" type="number" step="any" />
      </label>

      <label>
        radiusArcsec (optional)
        <input v-model="ztfRadiusArcsec" type="number" step="any" />
      </label>

      <label>
        Band (optional)
        <select v-model="ztfBand">
          <option value="g">g</option>
          <option value="r">r</option>
          <option value="i">i</option>
        </select>
      </label>

      <button type="submit" :disabled="loading">Ingest &amp; Plot (ZTF)</button>
    </form>

    <p v-if="loading">Loading light curve...</p>
    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="result?.source === 'ZTF'" class="ztf-note">
      ZTF: plotting magnitude vs time (mjd)
    </p>

    <div v-show="result" ref="plotEl" class="plot"></div>

    <p v-if="source === 'tess' && result && transitLoading" class="transit-status">
      Finding transit candidate...
    </p>
    <p v-if="source === 'tess' && transitError" class="error">{{ transitError }}</p>

    <p
      v-if="source === 'tess' && result && transitResult && !transitResult.found"
      class="subtle"
    >
      Transit candidate not found{{ transitResult.reason ? `: ${transitResult.reason}` : '' }}
    </p>

    <section
      v-if="source === 'tess' && transitResult?.found && transitResult.candidate"
      class="candidate-card"
    >
      <h2>Transit Candidate</h2>
      <p class="candidate-item">
        <span class="candidate-label">
          Period (days)
          <Tooltip
            label="Period (days)"
            text="Estimated time between transits. More observed transits and longer time coverage usually mean a more precise period."
          />
        </span>
        <span class="candidate-value">{{ transitResult.candidate.period_days.toFixed(5) }}</span>
      </p>
      <p class="candidate-item">
        <span class="candidate-label">
          Epoch (BTJD)
          <Tooltip
            label="Epoch (BTJD)"
            text="Estimated mid-transit time for one reference transit, in BTJD (BJD − 2457000). Use this with the period to predict future transits."
          />
        </span>
        <span class="candidate-value">{{ transitResult.candidate.t0_btjd.toFixed(5) }}</span>
      </p>
      <p v-if="epochMjd(transitResult.candidate) !== null" class="candidate-item">
        <span class="candidate-label">
          Epoch (MJD)
          <Tooltip
            label="Epoch (MJD)"
            text="Same epoch expressed as MJD for convenience (MJD = JD − 2400000.5)."
          />
        </span>
        <span class="candidate-value">{{ epochMjd(transitResult.candidate)?.toFixed(5) }}</span>
      </p>
      <p class="candidate-item">
        <span class="candidate-label">
          Duration (hours)
          <Tooltip
            label="Duration (hours)"
            text="Approximate time from transit start to end. Longer durations can indicate a larger star, a wider orbit, or a lower-impact transit."
          />
        </span>
        <span class="candidate-value">{{ transitResult.candidate.duration_hours.toFixed(2) }}</span>
      </p>
      <p class="candidate-item">
        <span class="candidate-label">
          Depth (%)
          <Tooltip
            label="Depth (%)"
            text="Approximate drop in brightness during transit. Depth ≈ (Rp/Rs)², so larger planets generally produce deeper transits."
          />
        </span>
        <span class="candidate-value">{{ transitResult.candidate.depth_pct.toFixed(3) }}</span>
      </p>
      <p v-if="transitResult.candidate.sde != null" class="candidate-item">
        <span class="candidate-label">
          SDE
          <Tooltip
            label="SDE"
            text="Signal Detection Efficiency: how strong the best BLS peak is compared to the background. Higher is better; values ~8+ often look promising, but false positives still happen."
          />
        </span>
        <span class="candidate-value">{{ transitResult.candidate.sde.toFixed(2) }}</span>
      </p>
      <p v-if="transitResult.candidate.depth_snr != null" class="candidate-item">
        <span class="candidate-label">
          Depth SNR
          <Tooltip
            label="Depth SNR"
            text="Depth signal-to-noise ratio: transit depth relative to the typical scatter in the detrended light curve. Higher is more confident."
          />
        </span>
        <span class="candidate-value">{{ transitResult.candidate.depth_snr.toFixed(2) }}</span>
      </p>
      <p class="candidate-item">
        <span class="candidate-label">
          Number of transits observed
          <Tooltip
            label="Number of transits observed"
            text="How many transits fall within the time range analyzed. More transits usually increases confidence."
          />
        </span>
        <span class="candidate-value">{{ transitResult.candidate.n_transits_observed }}</span>
      </p>
    </section>

    <div
      v-show="source === 'tess' && transitResult?.found"
      ref="transitPlotEl"
      class="plot transit-plot"
    ></div>

    <section v-if="result" class="meta">
      <p v-if="result.target">Target: {{ result.target }}</p>
      <p v-if="result.mission">Mission: {{ result.mission }}</p>
      <p v-if="result.author">Author: {{ result.author }}</p>
      <p v-if="typeof result.n_points === 'number'">Points: {{ result.n_points }}</p>
      <p v-if="result.cache">Cache: {{ result.cache.hit ? 'hit' : 'miss' }}</p>
      <p v-if="result.source === 'ZTF' && result.selected">
        Selected oid: {{ result.selected.oid }}
      </p>
      <p
        v-if="result.source === 'ZTF' && result.available && result.available.n_oids > 1"
        class="warning"
      >
        Multiple sources in radius; showing oid with most points
      </p>
      <p v-if="result.source === 'ZTF' && result.available">
        Available oids: {{ result.available.n_oids }}
      </p>
      <p v-if="result.source === 'ZTF' && result.selected?.filtercodes?.length">
        Filtercodes: {{ result.selected.filtercodes.join(', ') }}
      </p>
    </section>

    <small>API base URL: {{ configuredApiBaseUrl }}</small>
  </main>
</template>

<script setup lang="ts">
import Plotly from 'plotly.js-dist-min'
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import Tooltip from './components/Tooltip.vue'

import {
  runTransitSearch,
  type TransitCandidate,
  type TransitSearchRequest,
  type TransitSearchResponse
} from './services/api'

type IngestResponse = {
  source?: string
  target?: string
  mission?: string
  author?: string
  n_points?: number
  time: number[]
  flux?: number[]
  flux_err?: number[] | null
  mag?: number[]
  magerr?: number[] | null
  selected?: {
    oid: number
    filtercodes: string[]
  }
  available?: {
    oids: number[]
    n_oids: number
  }
  cache?: {
    hit: boolean
    key: string
    path: string
  }
}

type Source = 'tess' | 'ztf'

const configuredApiBaseUrl =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const normalizedApiBaseUrl = configuredApiBaseUrl.replace(/\/$/, '')

function apiEndpoint(path: string): string {
  return normalizedApiBaseUrl.includes('://api:')
    ? path
    : `${normalizedApiBaseUrl}${path}`
}

const source = ref<Source>('tess')

const target = ref('TIC 25155310')
const mission = ref<'TESS' | 'Kepler'>('TESS')
const author = ref('SPOC')
const sector = ref('')

const ztfObjectId = ref('')
const ztfRa = ref('298.0025')
const ztfDec = ref('29.87147')
const ztfRadiusArcsec = ref('5')
const ztfBand = ref<'g' | 'r' | 'i'>('r')

const loading = ref(false)
const error = ref('')
const result = ref<IngestResponse | null>(null)
const plotEl = ref<HTMLDivElement | null>(null)

const transitLoading = ref(false)
const transitError = ref('')
const transitResult = ref<TransitSearchResponse | null>(null)
const transitPlotEl = ref<HTMLDivElement | null>(null)
const lastTransitRequestKey = ref('')

let transitSearchTimer: ReturnType<typeof setTimeout> | null = null
let transitSearchSequence = 0

function parseSector(): number | null {
  const value = sector.value.trim()
  if (!value) {
    return null
  }

  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error('Sector must be a whole number or left blank')
  }

  return parsed
}

function parseOptionalNumber(
  value: string | number | null | undefined,
  fieldName: string
): number | null {
  if (value === null || value === undefined || value === '') {
    return null
  }

  const parsed =
    typeof value === 'number' ? value : Number(String(value).trim())
  if (!Number.isFinite(parsed)) {
    throw new Error(`${fieldName} must be a valid number or left blank`)
  }

  return parsed
}

function clearTransitState(resetCacheKey: boolean) {
  transitLoading.value = false
  transitError.value = ''
  transitResult.value = null
  if (resetCacheKey) {
    lastTransitRequestKey.value = ''
  }
  if (transitSearchTimer) {
    clearTimeout(transitSearchTimer)
    transitSearchTimer = null
  }
  transitSearchSequence += 1
}

async function renderPlot(data: IngestResponse) {
  await nextTick()
  if (!plotEl.value) {
    return
  }

  const isZtf = data.source === 'ZTF'
  const y = isZtf ? data.mag : data.flux
  if (!y) {
    throw new Error('No plottable values returned by API')
  }

  const trace: Record<string, unknown> = {
    x: data.time,
    y,
    type: 'scatter',
    mode: 'lines',
    name: 'Light curve'
  }
  if (isZtf && data.magerr) {
    trace.error_y = { type: 'data', array: data.magerr, visible: true }
  }

  await Plotly.react(
    plotEl.value,
    [trace],
    {
      margin: { t: 20, r: 20, b: 50, l: 60 },
      xaxis: { title: 'time' },
      yaxis:
        isZtf
          ? { title: 'mag', autorange: 'reversed' }
          : { title: 'normalized flux' }
    },
    { responsive: true }
  )
}

async function renderTransitPlot(data: TransitSearchResponse) {
  await nextTick()
  if (!transitPlotEl.value) {
    return
  }

  if (!data.found || !data.folded) {
    Plotly.purge(transitPlotEl.value)
    return
  }

  await Plotly.react(
    transitPlotEl.value,
    [
      {
        x: data.folded.phase,
        y: data.folded.flux,
        type: 'scatter',
        mode: 'markers',
        name: 'Binned folded flux',
        marker: { size: 5 }
      }
    ],
    {
      title: 'Phase-folded Transit (BLS best period)',
      margin: { t: 44, r: 20, b: 50, l: 60 },
      xaxis: { title: 'phase', range: [-0.5, 0.5] },
      yaxis: { title: 'detrended flux' },
      shapes: [
        {
          type: 'line',
          x0: -0.5,
          x1: 0.5,
          y0: 1.0,
          y1: 1.0,
          line: { color: '#777', width: 1, dash: 'dot' }
        }
      ]
    },
    { responsive: true }
  )
}

function buildTransitRequest(): TransitSearchRequest {
  return {
    target: target.value,
    mission: mission.value,
    author: author.value,
    sector: parseSector(),
    min_period_days: 0.5,
    max_period_days: null,
    detrend_window_days: 0.75,
    fold_bins: 200
  }
}

function scheduleTransitSearch() {
  if (transitSearchTimer) {
    clearTimeout(transitSearchTimer)
  }
  transitSearchTimer = setTimeout(() => {
    transitSearchTimer = null
    void runTransitSearchForCurrentIngest(false)
  }, 250)
}

async function runTransitSearchForCurrentIngest(force: boolean) {
  if (source.value !== 'tess' || !result.value) {
    return
  }

  let request: TransitSearchRequest
  try {
    request = buildTransitRequest()
  } catch (err) {
    transitResult.value = null
    transitError.value =
      err instanceof Error ? err.message : 'Invalid transit search settings'
    return
  }
  const requestKey = JSON.stringify(request)
  if (!force && requestKey === lastTransitRequestKey.value) {
    return
  }

  const sequence = ++transitSearchSequence
  transitLoading.value = true
  transitError.value = ''

  try {
    const response = await runTransitSearch(request)
    if (sequence !== transitSearchSequence) {
      return
    }

    transitResult.value = response
    lastTransitRequestKey.value = requestKey
    await renderTransitPlot(response)
  } catch (err) {
    if (sequence !== transitSearchSequence) {
      return
    }

    transitResult.value = null
    transitError.value =
      err instanceof Error ? err.message : 'Failed to run transit search'
  } finally {
    if (sequence === transitSearchSequence) {
      transitLoading.value = false
    }
  }
}

function epochMjd(candidate: TransitCandidate): number | null {
  const zeroPoint = candidate.time_system.zero_point
  if (
    candidate.time_system.format.toUpperCase() === 'BTJD' &&
    typeof zeroPoint === 'number'
  ) {
    return candidate.t0_btjd + zeroPoint - 2400000.5
  }
  return null
}

async function ingestTessAndPlot() {
  loading.value = true
  error.value = ''

  try {
    const response = await fetch(apiEndpoint('/api/ingest'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        target: target.value,
        mission: mission.value,
        author: author.value,
        sector: parseSector()
      })
    })

    if (!response.ok) {
      const errorBody = (await response.json().catch(() => null)) as
        | { detail?: string; error?: string }
        | null
      throw new Error(
        errorBody?.detail || errorBody?.error || 'Failed to ingest light curve'
      )
    }

    const data = (await response.json()) as IngestResponse
    result.value = data
    await renderPlot(data)

    clearTransitState(true)
    scheduleTransitSearch()
  } catch (err) {
    result.value = null
    clearTransitState(true)
    error.value = err instanceof Error ? err.message : 'Failed to ingest light curve'
  } finally {
    loading.value = false
  }
}

async function ingestZtfAndPlot() {
  loading.value = true
  error.value = ''

  try {
    const response = await fetch(apiEndpoint('/api/ingest/ztf'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        objectId: ztfObjectId.value.trim() || null,
        ra: parseOptionalNumber(ztfRa.value, 'RA'),
        dec: parseOptionalNumber(ztfDec.value, 'Dec'),
        radiusArcsec: parseOptionalNumber(ztfRadiusArcsec.value, 'radiusArcsec'),
        band: ztfBand.value
      })
    })

    if (!response.ok) {
      const errorBody = (await response.json().catch(() => null)) as
        | { detail?: string; error?: string; next?: string }
        | null
      throw new Error(
        [errorBody?.error || errorBody?.detail || 'Failed to ingest ZTF light curve', errorBody?.next]
          .filter(Boolean)
          .join(' ')
      )
    }

    const data = (await response.json()) as IngestResponse
    result.value = data
    await renderPlot(data)

    clearTransitState(true)
  } catch (err) {
    result.value = null
    clearTransitState(true)
    error.value =
      err instanceof Error ? err.message : 'Failed to ingest ZTF light curve'
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  const params = new URLSearchParams(window.location.search)
  const value = params.get('source')
  if (value === 'tess' || value === 'ztf') {
    source.value = value
  }
})

watch(source, (value) => {
  const params = new URLSearchParams(window.location.search)
  params.set('source', value)
  const query = params.toString()
  const nextUrl = query
    ? `${window.location.pathname}?${query}`
    : window.location.pathname
  window.history.replaceState({}, '', nextUrl)

  error.value = ''
  result.value = null
  clearTransitState(true)
})

onBeforeUnmount(() => {
  if (plotEl.value) {
    Plotly.purge(plotEl.value)
  }
  if (transitPlotEl.value) {
    Plotly.purge(transitPlotEl.value)
  }
})
</script>

<style scoped>
main {
  min-height: 100vh;
  width: min(980px, 92vw);
  margin: 0 auto;
  padding: 1.5rem 0;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  font-family: sans-serif;
}

h1 {
  margin: 0;
  font-size: 2rem;
}

h2 {
  margin: 0;
  font-size: 1.1rem;
}

.source {
  max-width: 240px;
}

.form {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.75rem;
  align-items: end;
}

label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.95rem;
}

input,
select,
button {
  padding: 0.5rem;
  font: inherit;
}

.plot {
  width: 100%;
  min-height: 420px;
}

.transit-plot {
  min-height: 360px;
}

.candidate-card {
  border: 1px solid #d8dde3;
  border-radius: 8px;
  padding: 0.75rem;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 0.6rem 0.8rem;
}

.candidate-card h2 {
  grid-column: 1 / -1;
}

.candidate-card p {
  margin: 0;
}

.candidate-item {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.2rem;
}

.candidate-label {
  display: inline-flex;
  align-items: center;
  line-height: 1.25;
  font-size: 0.95rem;
  color: #344054;
}

.candidate-value {
  line-height: 1.2;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.transit-status,
.subtle {
  margin: 0;
  font-size: 0.95rem;
}

.subtle {
  color: #475467;
}

.meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.5rem;
}

.meta p,
small {
  margin: 0;
}

.error {
  color: #b30000;
}

.ztf-note {
  margin: 0;
  font-size: 0.95rem;
}

.warning {
  color: #8a3f00;
}

@media (prefers-color-scheme: dark) {
  .candidate-label {
    color: #cbd5e1;
  }
}
</style>
