import { computed, nextTick, onBeforeUnmount, ref } from 'vue';
const props = defineProps();
const open = ref(false);
const triggerEl = ref(null);
const bubbleEl = ref(null);
const wrapEl = ref(null);
const bubbleTop = ref(0);
const bubbleLeft = ref(0);
const tooltipIdSeed = Math.random().toString(36).slice(2, 8);
const tooltipId = computed(() => `tooltip-${props.label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${tooltipIdSeed}`);
const bubbleStyle = computed(() => ({
    top: `${bubbleTop.value}px`,
    left: `${bubbleLeft.value}px`
}));
function onOpen() {
    if (open.value) {
        return;
    }
    open.value = true;
    void nextTick(updatePosition);
}
function onClick() {
    open.value = !open.value;
    if (open.value) {
        void nextTick(updatePosition);
    }
}
function onLeave() {
    if (document.activeElement !== triggerEl.value) {
        open.value = false;
    }
}
function onBlur() {
    open.value = false;
}
function onEscape() {
    open.value = false;
    triggerEl.value?.blur();
}
function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}
function updatePosition() {
    if (!triggerEl.value || !bubbleEl.value) {
        return;
    }
    const triggerRect = triggerEl.value.getBoundingClientRect();
    const bubbleRect = bubbleEl.value.getBoundingClientRect();
    const gap = 8;
    const viewportPadding = 8;
    const preferredTop = triggerRect.top - bubbleRect.height - gap;
    if (preferredTop >= viewportPadding) {
        bubbleTop.value = preferredTop;
        bubbleLeft.value = clamp(triggerRect.left + triggerRect.width / 2 - bubbleRect.width / 2, viewportPadding, window.innerWidth - bubbleRect.width - viewportPadding);
        return;
    }
    const rightLeft = triggerRect.right + gap;
    const canPlaceRight = rightLeft + bubbleRect.width <= window.innerWidth - viewportPadding;
    if (canPlaceRight) {
        bubbleTop.value = clamp(triggerRect.top + triggerRect.height / 2 - bubbleRect.height / 2, viewportPadding, window.innerHeight - bubbleRect.height - viewportPadding);
        bubbleLeft.value = rightLeft;
        return;
    }
    bubbleTop.value = clamp(triggerRect.bottom + gap, viewportPadding, window.innerHeight - bubbleRect.height - viewportPadding);
    bubbleLeft.value = clamp(triggerRect.left + triggerRect.width / 2 - bubbleRect.width / 2, viewportPadding, window.innerWidth - bubbleRect.width - viewportPadding);
}
function onDocumentPointer(event) {
    const target = event.target;
    if (!target || !wrapEl.value) {
        return;
    }
    if (!wrapEl.value.contains(target)) {
        open.value = false;
    }
}
function onWindowResize() {
    if (open.value) {
        updatePosition();
    }
}
document.addEventListener('mousedown', onDocumentPointer);
document.addEventListener('touchstart', onDocumentPointer, { passive: true });
window.addEventListener('resize', onWindowResize);
window.addEventListener('scroll', onWindowResize, true);
onBeforeUnmount(() => {
    document.removeEventListener('mousedown', onDocumentPointer);
    document.removeEventListener('touchstart', onDocumentPointer);
    window.removeEventListener('resize', onWindowResize);
    window.removeEventListener('scroll', onWindowResize, true);
});
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_components;
let __VLS_directives;
/** @type {__VLS_StyleScopedClasses['tooltip-trigger']} */ ;
/** @type {__VLS_StyleScopedClasses['tooltip-trigger']} */ ;
/** @type {__VLS_StyleScopedClasses['tooltip-bubble']} */ ;
// CSS variable injection 
// CSS variable injection end 
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
    ...{ class: "tooltip-wrap" },
    ref: "wrapEl",
});
/** @type {typeof __VLS_ctx.wrapEl} */ ;
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onMouseenter: (__VLS_ctx.onOpen) },
    ...{ onMouseleave: (__VLS_ctx.onLeave) },
    ...{ onFocus: (__VLS_ctx.onOpen) },
    ...{ onBlur: (__VLS_ctx.onBlur) },
    ...{ onClick: (__VLS_ctx.onClick) },
    ...{ onKeydown: (__VLS_ctx.onEscape) },
    ref: "triggerEl",
    type: "button",
    tabindex: "0",
    ...{ class: "tooltip-trigger" },
    'aria-label': (`More info: ${__VLS_ctx.label}`),
    'aria-expanded': (__VLS_ctx.open ? 'true' : 'false'),
    'aria-describedby': (__VLS_ctx.open ? __VLS_ctx.tooltipId : undefined),
});
/** @type {typeof __VLS_ctx.triggerEl} */ ;
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
    'aria-hidden': "true",
});
if (__VLS_ctx.open) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ref: "bubbleEl",
        id: (__VLS_ctx.tooltipId),
        role: "tooltip",
        ...{ class: "tooltip-bubble" },
        ...{ style: (__VLS_ctx.bubbleStyle) },
    });
    /** @type {typeof __VLS_ctx.bubbleEl} */ ;
    (__VLS_ctx.text);
}
/** @type {__VLS_StyleScopedClasses['tooltip-wrap']} */ ;
/** @type {__VLS_StyleScopedClasses['tooltip-trigger']} */ ;
/** @type {__VLS_StyleScopedClasses['tooltip-bubble']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            open: open,
            triggerEl: triggerEl,
            bubbleEl: bubbleEl,
            wrapEl: wrapEl,
            tooltipId: tooltipId,
            bubbleStyle: bubbleStyle,
            onOpen: onOpen,
            onClick: onClick,
            onLeave: onLeave,
            onBlur: onBlur,
            onEscape: onEscape,
        };
    },
    __typeProps: {},
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
    __typeProps: {},
});
; /* PartiallyEnd: #4569/main.vue */
