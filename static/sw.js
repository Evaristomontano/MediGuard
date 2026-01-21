const CACHE_NAME = 'mediguard-v1';
// No guardamos archivos en caché por ahora para no complicar, 
// pero el evento 'fetch' debe estar presente.
self.addEventListener('install', (event) => {
    self.skipWaiting();
});

self.addEventListener('fetch', (event) => {
    event.respondWith(fetch(event.request));
});