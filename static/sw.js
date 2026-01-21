self.addEventListener('install', (e) => {
  console.log('[Service Worker] Install');
});

self.addEventListener('fetch', (e) => {
  // Estrategia básica: solo deja pasar las peticiones
  e.respondWith(fetch(e.request));
});