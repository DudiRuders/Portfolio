import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

// https://astro.build/config
export default defineConfig({
  // Podstawowy adres Twojej strony na GitHub Pages
  site: 'https://DudiRuders.github.io',
  
  // Nazwa Twojego repozytorium (zwróć uwagę na wielkość liter!)
  base: '/Portfolio',

  integrations: [tailwind()],
});