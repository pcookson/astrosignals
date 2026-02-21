<template>
  <span class="tooltip-wrap" ref="wrapEl">
    <button
      ref="triggerEl"
      type="button"
      tabindex="0"
      class="tooltip-trigger"
      :aria-label="`More info: ${label}`"
      :aria-expanded="open ? 'true' : 'false'"
      :aria-describedby="open ? tooltipId : undefined"
      @mouseenter="onOpen"
      @mouseleave="onLeave"
      @focus="onOpen"
      @blur="onBlur"
      @click="onClick"
      @keydown.esc.prevent="onEscape"
    >
      <span aria-hidden="true">ⓘ</span>
    </button>

    <div
      v-if="open"
      ref="bubbleEl"
      :id="tooltipId"
      role="tooltip"
      class="tooltip-bubble"
      :style="bubbleStyle"
    >
      {{ text }}
    </div>
  </span>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref } from 'vue'

const props = defineProps<{
  label: string
  text: string
}>()

const open = ref(false)
const triggerEl = ref<HTMLButtonElement | null>(null)
const bubbleEl = ref<HTMLDivElement | null>(null)
const wrapEl = ref<HTMLSpanElement | null>(null)
const bubbleTop = ref(0)
const bubbleLeft = ref(0)

const tooltipIdSeed = Math.random().toString(36).slice(2, 8)
const tooltipId = computed(
  () => `tooltip-${props.label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${tooltipIdSeed}`
)

const bubbleStyle = computed(() => ({
  top: `${bubbleTop.value}px`,
  left: `${bubbleLeft.value}px`
}))

function onOpen() {
  if (open.value) {
    return
  }
  open.value = true
  void nextTick(updatePosition)
}

function onClick() {
  open.value = !open.value
  if (open.value) {
    void nextTick(updatePosition)
  }
}

function onLeave() {
  if (document.activeElement !== triggerEl.value) {
    open.value = false
  }
}

function onBlur() {
  open.value = false
}

function onEscape() {
  open.value = false
  triggerEl.value?.blur()
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function updatePosition() {
  if (!triggerEl.value || !bubbleEl.value) {
    return
  }

  const triggerRect = triggerEl.value.getBoundingClientRect()
  const bubbleRect = bubbleEl.value.getBoundingClientRect()
  const gap = 8
  const viewportPadding = 8

  const preferredTop = triggerRect.top - bubbleRect.height - gap
  if (preferredTop >= viewportPadding) {
    bubbleTop.value = preferredTop
    bubbleLeft.value = clamp(
      triggerRect.left + triggerRect.width / 2 - bubbleRect.width / 2,
      viewportPadding,
      window.innerWidth - bubbleRect.width - viewportPadding
    )
    return
  }

  const rightLeft = triggerRect.right + gap
  const canPlaceRight = rightLeft + bubbleRect.width <= window.innerWidth - viewportPadding
  if (canPlaceRight) {
    bubbleTop.value = clamp(
      triggerRect.top + triggerRect.height / 2 - bubbleRect.height / 2,
      viewportPadding,
      window.innerHeight - bubbleRect.height - viewportPadding
    )
    bubbleLeft.value = rightLeft
    return
  }

  bubbleTop.value = clamp(
    triggerRect.bottom + gap,
    viewportPadding,
    window.innerHeight - bubbleRect.height - viewportPadding
  )
  bubbleLeft.value = clamp(
    triggerRect.left + triggerRect.width / 2 - bubbleRect.width / 2,
    viewportPadding,
    window.innerWidth - bubbleRect.width - viewportPadding
  )
}

function onDocumentPointer(event: MouseEvent | TouchEvent) {
  const target = event.target as Node | null
  if (!target || !wrapEl.value) {
    return
  }
  if (!wrapEl.value.contains(target)) {
    open.value = false
  }
}

function onWindowResize() {
  if (open.value) {
    updatePosition()
  }
}

document.addEventListener('mousedown', onDocumentPointer)
document.addEventListener('touchstart', onDocumentPointer, { passive: true })
window.addEventListener('resize', onWindowResize)
window.addEventListener('scroll', onWindowResize, true)

onBeforeUnmount(() => {
  document.removeEventListener('mousedown', onDocumentPointer)
  document.removeEventListener('touchstart', onDocumentPointer)
  window.removeEventListener('resize', onWindowResize)
  window.removeEventListener('scroll', onWindowResize, true)
})
</script>

<style scoped>
.tooltip-wrap {
  display: inline-flex;
  align-items: center;
  position: relative;
}

.tooltip-trigger {
  border: 0;
  background: transparent;
  color: #556070;
  cursor: pointer;
  padding: 0;
  margin-left: 0.25rem;
  line-height: 1;
  font-size: 0.95rem;
}

.tooltip-trigger:focus-visible {
  outline: 2px solid #4f46e5;
  outline-offset: 2px;
  border-radius: 4px;
}

.tooltip-bubble {
  position: fixed;
  z-index: 2000;
  max-width: 260px;
  border-radius: 8px;
  padding: 0.5rem 0.6rem;
  font-size: 0.82rem;
  line-height: 1.35;
  color: #111827;
  background: #ffffff;
  border: 1px solid #d6dbe1;
  box-shadow: 0 8px 20px rgba(16, 24, 40, 0.14);
}

@media (prefers-color-scheme: dark) {
  .tooltip-trigger {
    color: #c7d2e0;
  }

  .tooltip-bubble {
    color: #e5e7eb;
    background: #111827;
    border-color: #334155;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45);
  }
}
</style>
