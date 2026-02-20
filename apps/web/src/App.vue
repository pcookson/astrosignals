<template>
  <main>
    <h1>AstroSignals</h1>

    <form @submit.prevent="ingestAndPlot" class="form">
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

    <p v-if="loading">Loading light curve...</p>
    <p v-if="error" class="error">{{ error }}</p>

    <div v-show="result" ref="plotEl" class="plot"></div>

    <section v-if="result" class="meta">
      <p>Target: {{ result.target }}</p>
      <p>Mission: {{ result.mission }}</p>
      <p>Author: {{ result.author }}</p>
      <p>Points: {{ result.n_points }}</p>
    </section>

    <small>API base URL: {{ configuredApiBaseUrl }}</small>
  </main>
</template>

<script setup lang="ts">
import Plotly from 'plotly.js-dist-min'
import { nextTick, onBeforeUnmount, ref } from 'vue'

type IngestResponse = {
  target: string
  mission: string
  author: string
  n_points: number
  time: number[]
  flux: number[]
  flux_err: number[] | null
}

const configuredApiBaseUrl =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const normalizedApiBaseUrl = configuredApiBaseUrl.replace(/\/$/, '')
const ingestUrl = normalizedApiBaseUrl.includes('://api:')
  ? '/api/ingest'
  : `${normalizedApiBaseUrl}/api/ingest`

const target = ref('TIC 25155310')
const mission = ref<'TESS' | 'Kepler'>('TESS')
const author = ref('SPOC')
const sector = ref('')

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

async function renderPlot(data: IngestResponse) {
  await nextTick()
  if (!plotEl.value) {
    return
  }

  await Plotly.react(
    plotEl.value,
    [
      {
        x: data.time,
        y: data.flux,
        type: 'scatter',
        mode: 'lines',
        name: 'Flux'
      }
    ],
    {
      margin: { t: 20, r: 20, b: 50, l: 60 },
      xaxis: { title: 'Time (days)' },
      yaxis: { title: 'Normalized Flux' }
    },
    { responsive: true }
  )
}

async function ingestAndPlot() {
  loading.value = true
  error.value = ''

  try {
    const requestBody = {
      target: target.value,
      mission: mission.value,
      author: author.value,
      sector: parseSector()
    }

    const response = await fetch(ingestUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    })

    if (!response.ok) {
      const errorBody = (await response.json().catch(() => null)) as
        | { detail?: string }
        | null
      throw new Error(errorBody?.detail || 'Failed to ingest light curve')
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
</style>
