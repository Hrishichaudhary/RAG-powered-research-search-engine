export default defineNuxtConfig({
  compatibilityDate: '2025-07-10',
  modules: [
    '@nuxtjs/tailwindcss',
    '@nuxt/content'
  ],
  css: ['~/assets/css/tailwind.css'],
  pages: true,
  runtimeConfig: {
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || process.env.VITE_API_BASE || 'http://127.0.0.1:8000'
    }
  },
  content: {
    experimental: {
      search: true
    }
  }
})