<template>
  <main>
    <h1>AstroSignals</h1>
    <p v-if="loading">Checking API...</p>
    <p v-else-if="connected">API connected ✅</p>
    <p v-else>API not reachable ❌</p>
    <button type="button" @click="checkApi" :disabled="loading">
      Retry
    </button>
    <small>API base URL: {{ configuredApiBaseUrl }}</small>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'

const configuredApiBaseUrl =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const loading = ref(false)
const connected = ref(false)

async function checkApi() {
  loading.value = true

  try {
    const response = await fetch('/api/health')
    const data = await response.json()
    connected.value = response.ok && data.ok === true
  } catch {
    connected.value = false
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  checkApi()
})
</script>

<style scoped>
main {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 0.75rem;
  font-family: sans-serif;
}

h1 {
  margin: 0;
  font-size: 2rem;
}

p {
  margin: 0;
}

button {
  padding: 0.5rem 1rem;
}

small {
  color: #444;
}
</style>
