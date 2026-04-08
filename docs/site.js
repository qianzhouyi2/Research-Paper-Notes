(() => {
  const input = document.querySelector('[data-search-input]');
  const results = document.querySelector('[data-search-results]');
  const toggle = document.querySelector('[data-menu-toggle]');
  const body = document.body;
  const rootPrefix = body.dataset.rootPrefix || '';
  const currentPath = body.dataset.pagePath || '';
  let entries = [];

  function withRoot(path) {
    return `${rootPrefix}${path}`;
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function renderResults(query) {
    if (!results) return;
    const q = query.trim().toLowerCase();
    if (!q) {
      results.classList.remove('visible');
      results.innerHTML = '';
      return;
    }

    const filtered = entries
      .filter((entry) =>
        entry.title.toLowerCase().includes(q) ||
        entry.path.toLowerCase().includes(q) ||
        entry.summary.toLowerCase().includes(q)
      )
      .slice(0, 12);

    if (!filtered.length) {
      results.classList.add('visible');
      results.innerHTML = '<div class="panel" style="padding:14px 16px;">No matching pages.</div>';
      return;
    }

    const cards = filtered
      .map((entry) => {
        const summary = entry.summary ? escapeHtml(entry.summary) : 'Open page';
        const section = escapeHtml(entry.section || 'page');
        return `
          <a class="panel" href="${encodeURI(withRoot(entry.url))}">
            <strong>${escapeHtml(entry.title)}</strong>
            <span>${section}</span>
            <span>${summary}</span>
          </a>`;
      })
      .join('');

    results.classList.add('visible');
    results.innerHTML = cards;
  }

  if (input && results) {
    fetch(withRoot('search-index.json'))
      .then((res) => res.json())
      .then((data) => {
        entries = Array.isArray(data) ? data.filter((entry) => entry.url !== currentPath) : [];
      })
      .catch(() => {
        entries = [];
      });

    input.addEventListener('input', (event) => renderResults(event.target.value));
    document.addEventListener('click', (event) => {
      if (!results.contains(event.target) && event.target !== input) {
        results.classList.remove('visible');
      }
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        results.classList.remove('visible');
        input.blur();
      }
    });
  }

  if (toggle) {
    toggle.addEventListener('click', () => {
      body.classList.toggle('menu-open');
    });
  }
})();
