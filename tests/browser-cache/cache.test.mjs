import {test, beforeEach} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {webcrypto} from 'node:crypto';

if (!globalThis.crypto) Object.defineProperty(globalThis, 'crypto', {value: webcrypto});
globalThis.location = {origin: 'https://example.test'};
globalThis.addEventListener = () => {};
const stores = new Map();
let fail = false, failIndex = false;
class FakeCache {
    entries = new Map();
    async match(request) { return this.entries.get(request.url)?.clone(); }
    async put(request, response) {
        if (fail || (failIndex && request.url.endsWith('/__index__'))) throw new DOMException('No space', 'QuotaExceededError');
        this.entries.set(request.url, response.clone());
    }
    async delete(request) { return this.entries.delete(request.url); }
}
globalThis.caches = {
    async keys() { return [...stores.keys()]; },
    async open(name) {
        if (!stores.has(name)) stores.set(name, new FakeCache());
        return stores.get(name);
    },
    async delete(name) { return stores.delete(name); },
};
const source = await readFile(new URL('../../libs/impeller/bevy/src/browser_cache.js', import.meta.url), 'utf8');
const cache = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
let session = 0;
beforeEach(() => {
    fail = false; failIndex = false;
    globalThis.__elodinDisableDiskCache = false;
    cache.activate('session-' + ++session, 12, 8);
});
const scope = () => 'session-' + session;

test('round trip copies bytes before WASM memory changes', async () => {
    const data = new Uint8Array([1, 2, 3]);
    const write = cache.put(scope(), 'video/1', data);
    data.fill(9);
    assert.equal(await write, true);
    assert.deepEqual(await cache.get(scope(), 'video/1'), new Uint8Array([1, 2, 3]));
    for (const store of stores.values()) {
        for (const [url, response] of store.entries) {
            assert.equal(url.includes(scope()), false);
            assert.equal(response.headers.has('authorization'), false);
            assert.equal(response.headers.has('set-cookie'), false);
        }
    }
});

test('disk eviction causes a miss and preserves the byte limit', async () => {
    assert.equal(await cache.put(scope(), 'old', new Uint8Array(8)), true);
    assert.equal(await cache.put(scope(), 'new', new Uint8Array(8)), true);
    assert.equal(await cache.get(scope(), 'old'), undefined);
    assert.equal((await cache.get(scope(), 'new')).length, 8);
    assert.equal(globalThis.__elodinCacheStats.diskBytes, 8);
});

test('oversized entries and concurrent copies remain bounded', async () => {
    assert.equal(cache.put(scope(), 'large', new Uint8Array(9)), undefined);
    const writes = Array.from({length: 10}, (_, i) => cache.put(scope(), String(i), new Uint8Array(8)));
    assert.ok(globalThis.__elodinCacheStats.pendingBytes <= 16);
    await Promise.all(writes);
    assert.equal(globalThis.__elodinCacheStats.pendingBytes, 0);
    assert.ok(globalThis.__elodinCacheStats.diskBytes <= 12);
});

test('quota failure and disabled storage fall back to a miss', async () => {
    fail = true;
    assert.equal(await cache.put(scope(), 'failed', new Uint8Array(4)), false);
    assert.equal(await cache.get(scope(), 'failed'), undefined);
    globalThis.__elodinDisableDiskCache = true;
    assert.equal(await cache.get(scope(), 'absent'), undefined);
    assert.equal(cache.put(scope(), 'disabled', new Uint8Array(4)), undefined);
});

test('session changes reject in-flight writes and old reads', async () => {
    const previous = scope();
    const write = cache.put(previous, 'old', new Uint8Array(4));
    cache.activate('other-recording', 12, 8);
    assert.equal(await write, false);
    assert.equal(await cache.get(previous, 'old'), undefined);
    assert.equal(await cache.get('other-recording', 'old'), undefined);
});

test('unavailable Cache API returns a miss', async () => {
    const saved = globalThis.caches;
    delete globalThis.caches;
    assert.equal(await cache.get(scope(), 'missing'), undefined);
    assert.equal(await cache.put(scope(), 'missing', new Uint8Array(4)), false);
    globalThis.caches = saved;
});

test('an interrupted index write does not leave an untracked payload', async () => {
    failIndex = true;
    assert.equal(await cache.put(scope(), 'interrupted', new Uint8Array(4)), false);
    assert.equal(await cache.get(scope(), 'interrupted'), undefined);
});

test('an in-flight read cannot cross a recording change', async () => {
    const previous = scope();
    assert.equal(await cache.put(previous, 'frame', new Uint8Array([4])), true);
    const pending = cache.get(previous, 'frame');
    cache.activate('next-session', 12, 8);
    assert.equal(await pending, undefined);
});

test('asset commits retain a write slot while playback chunks arrive', async () => {
    cache.activate(scope(), 64, 8);
    for (let index = 0; index < 3; index++) {
        const playback = cache.put(scope(), 'video/' + index, new Uint8Array(4));
        const committed = new Promise(resolve => cache.write(scope(), 'asset/' + index, new Uint8Array(8).fill(index), value => resolve(value !== undefined)));
        assert.equal(await committed, true);
        await playback;
    }
    for (let index = 0; index < 3; index++) {
        assert.equal((await cache.get(scope(), 'asset/' + index))[0], index);
    }
});

test('tiny chunks cannot grow the disk index without a bound', async () => {
    const current = scope() + "-index";
    cache.activate(current, 1024 * 1024, 8);
    assert.equal(await cache.put(current, 'initial', new Uint8Array(1)), true);
    const store = [...stores.values()].at(-1);
    const indexURL = [...store.entries.keys()].find(url => url.endsWith('/__index__'));
    const entries = Array.from({length: 4096}, (_, index) => ({key: 'tiny/' + index, bytes: 1}));
    await store.put(new Request(indexURL), new Response(JSON.stringify({updated: Date.now(), entries})));
    assert.equal(await cache.put(current, 'next', new Uint8Array(1)), true);
    const index = await (await store.match(new Request(indexURL))).json();
    assert.equal(index.entries.length, 4096);
    assert.equal(index.entries[0].key, 'tiny/1');
    assert.equal(index.entries.at(-1).key, 'next');
});
