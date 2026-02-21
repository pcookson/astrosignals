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

    <section v-if="result" class="meta">
      <p v-if="result.target">Target: {{ result.target }}</p>
      <p v-if="result.mission">Mission: {{ result.mission }}</p>
      <p v-if="result.author">Author: {{ result.author }}</p>
      <p v-if="typeof result.n_points === 'number'">Points: {{ result.n_points }}</p>
      <p v-if="result.cache">Cache: {{ result.cache.hit ? 'hit' : 'miss' }}</p>
      <p v-if="result.source === 'ZTF' && result.selected">
        Selected oid: {{ result.selected.oid }}
      </p>
      <p v-if="result.source === 'ZTF' && result.available && result.available.n_oids > 1" class="warning">
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
  } catch (err) {
    result.value = null
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
  } catch (err) {
    result.value = null
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
})

onBeforeUnmount(() => {
  if (plotEl.value) {
    Plotly.purge(plotEl.value)
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
</style>
