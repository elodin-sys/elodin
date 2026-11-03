// The disk backend is optional. Requests and response headers never carry credentials into storage.
const prefix = 'elodin-session-cache-v1-';
const indexEntries = 4096, indexBytes = 2 * 1024 * 1024;
let active = '', generation = 0, diskBudget = 0, entryBudget = 0;
let reads = 0, writes = 0, serial = Promise.resolve();
const stats = globalThis.__elodinCacheStats = {hits: 0, misses: 0, writes: 0, errors: 0, diskBytes: 0, pendingBytes: 0};

async function identity(scope) {
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(scope));
    return prefix + Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}

function requestKey(name, key) {
    return new Request(new URL('/__elodin_cache__/' + name + '/' + encodeURIComponent(key), location.origin), {credentials: 'omit'});
}

async function limited(response, budget) {
    if (!response.body) return new Uint8Array();
    const reader = response.body.getReader(), chunks = [];
    let size = 0;
    try {
        for (;;) {
            const {done, value} = await reader.read();
            if (done) break;
            size += value.byteLength;
            if (size > budget) throw Error('Recording chunk exceeds the configured byte budget');
            chunks.push(value);
        }
    } catch (error) {
        await reader.cancel().catch(() => {});
        throw error;
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    return bytes;
}

export function resident(kind, bytes) { stats[kind] = bytes; }

export function activate(scope, diskBytes, entryBytes) {
    if (scope === active && diskBytes === diskBudget && entryBytes === entryBudget) return;
    const previous = active;
    active = scope;
    generation++;
    diskBudget = Math.max(0, diskBytes);
    entryBudget = Math.max(0, entryBytes);
    stats.diskBytes = 0;
    const token = generation;
    globalThis.navigator?.storage?.estimate?.().then(value => {
        if (token === generation) { stats.storageUsage = value.usage; stats.storageQuota = value.quota; }
    }).catch(() => {});
    if (previous && globalThis.caches) identity(previous).then(name => caches.delete(name)).catch(() => {});
}

async function open(scope) {
    if (scope !== active || !scope || !diskBudget || !globalThis.caches || globalThis.__elodinDisableDiskCache) return null;
    return caches.open(await identity(scope));
}

export async function get(scope, key) {
    if (reads >= 4) { stats.misses++; return undefined; }
    reads++;
    const token = generation;
    try {
        const cache = await open(scope);
        if (!cache) { stats.misses++; return undefined; }
        const name = await identity(scope), response = await cache.match(requestKey(name, key));
        if (!response || token !== generation) { stats.misses++; return undefined; }
        const bytes = await limited(response, entryBudget);
        if (scope !== active || token !== generation) return undefined;
        stats.hits++;
        return bytes;
    } catch (_) { stats.errors++; stats.misses++; return undefined; }
    finally { reads--; }
}

export function request(scope, key, done) {
    Promise.race([get(scope, key), new Promise(resolve => setTimeout(resolve, 2000))])
        .then(value => done(value), () => done(undefined));
}

export function put(scope, key, data, priority = false) {
    if (!scope || scope !== active || !diskBudget || data.byteLength > entryBudget || writes >= (priority ? 2 : 1) || globalThis.__elodinDisableDiskCache) return;
    const bytes = data.slice(), token = generation;
    writes++;
    stats.pendingBytes += bytes.byteLength;
    serial = serial.catch(() => {}).then(async () => {
        if (token !== generation) return false;
        const cache = await open(scope);
        if (!cache) return false;
        const name = await identity(scope), indexKey = requestKey(name, '__index__');
        const metadata = await cache.match(indexKey);
        const index = metadata ? JSON.parse(new TextDecoder().decode(await limited(metadata, indexBytes))) : {updated: Date.now(), entries: []};
        if (!Array.isArray(index.entries) || index.entries.length > indexEntries) throw Error('Invalid recording cache index');
        const previous = index.entries.findIndex(entry => entry.key === key);
        if (previous >= 0) index.entries.splice(previous, 1);
        let total = index.entries.reduce((sum, entry) => sum + entry.bytes, 0);
        while (index.entries.length && (total + bytes.byteLength > diskBudget || index.entries.length >= indexEntries)) {
            const oldest = index.entries.shift();
            await cache.delete(requestKey(name, oldest.key));
            total -= oldest.bytes;
        }
        if (bytes.byteLength > diskBudget || token !== generation) return false;
        await cache.put(requestKey(name, key), new Response(bytes, {headers: {'Content-Type': 'application/octet-stream'}}));
        if (token !== generation) { await caches.delete(name); return false; }
        index.updated = Date.now();
        index.entries.push({key, bytes: bytes.byteLength});
        try {
            await cache.put(indexKey, new Response(JSON.stringify(index), {headers: {'Content-Type': 'application/json'}}));
        } catch (error) {
            await cache.delete(requestKey(name, key)).catch(() => {});
            throw error;
        }
        stats.diskBytes = total + bytes.byteLength;
        stats.writes++;
        return true;
    }).catch(() => { stats.errors++; return false; }).finally(() => { writes--; stats.pendingBytes -= bytes.byteLength; });
    return serial;
}

export function write(scope, key, data, done) {
    const pending = put(scope, key, data, true);
    if (!pending) { done(undefined); return; }
    Promise.race([pending, new Promise(resolve => setTimeout(() => resolve(false), 2000))])
        .then(ok => done(ok ? new Uint8Array() : undefined), () => done(undefined));
}

addEventListener('pagehide', () => activate('', diskBudget, entryBudget));

// Old session namespaces are disposable, including after an interrupted browser shutdown.
if (globalThis.caches) caches.keys().then(async names => {
    for (const name of names.filter(name => name.startsWith(prefix)).slice(0, 8)) {
        const cache = await caches.open(name), metadata = await cache.match(requestKey(name, '__index__'));
        const updated = metadata ? JSON.parse(new TextDecoder().decode(await limited(metadata, indexBytes))).updated : 0;
        if (Date.now() - updated > 24 * 60 * 60 * 1000) await caches.delete(name);
    }
}).catch(() => {});
