const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const normalizedApiBaseUrl = configuredApiBaseUrl.replace(/\/$/, '');
function apiEndpoint(path) {
    return normalizedApiBaseUrl.includes('://api:')
        ? path
        : `${normalizedApiBaseUrl}${path}`;
}
export async function runTransitSearch(req) {
    const response = await fetch(apiEndpoint('/api/tess/transit-search'), {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(req)
    });
    if (!response.ok) {
        const errorBody = (await response.json().catch(() => null));
        throw new Error(errorBody?.detail || errorBody?.error || 'Failed to run transit search');
    }
    return (await response.json());
}
