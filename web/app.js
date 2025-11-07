const MAX_EVENT_ROWS = 50;

function setupTabs() {
  const tabButtons = Array.from(document.querySelectorAll('[data-tab-target]'));
  const panels = new Map(
    Array.from(document.querySelectorAll('[data-tab-panel]')).map((panel) => [
      panel.dataset.tabPanel,
      panel,
    ]),
  );

  if (!tabButtons.length || !panels.size) {
    return;
  }

  const activate = (id) => {
    tabButtons.forEach((btn) => {
      const isActive = btn.dataset.tabTarget === id;
      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-selected', String(isActive));
    });
    panels.forEach((panel, key) => {
      panel.hidden = key !== id;
    });
  };

  tabButtons.forEach((button) => {
    button.addEventListener('click', () => activate(button.dataset.tabTarget));
  });

  activate(tabButtons[0].dataset.tabTarget);
}

function initSseBridge() {
  const root = document.querySelector('[data-stream-root]');
  const statusEl = document.getElementById('job-sse-status');
  if (!root || !statusEl) {
    return;
  }

  const fieldMap = new Map();
  root.querySelectorAll('[data-sse-field]').forEach((el) => {
    fieldMap.set(el.dataset.sseField, el);
  });
  const logEl = root.querySelector('[data-sse-log]');

  const setStatus = (value) => {
    statusEl.textContent = value;
  };

  const appendLog = (html) => {
    if (!logEl) {
      return;
    }
    const entry = document.createElement('li');
    entry.innerHTML = html;
    logEl.prepend(entry);
    while (logEl.children.length > MAX_EVENT_ROWS) {
      logEl.removeChild(logEl.lastChild);
    }
  };

  const updateField = (field, payload) => {
    const el = fieldMap.get(field);
    if (!el) {
      return;
    }
    switch (field) {
      case 'manifest':
        el.textContent = formatManifest(payload);
        break;
      case 'raw':
        el.textContent = payload;
        break;
      case 'links':
        renderLinks(el, payload);
        break;
      case 'artifacts':
        renderArtifacts(el, payload);
        break;
      default:
        el.innerHTML = payload;
    }
  };

  let source = null;

  const connect = (jobId) => {
    if (source) {
      source.close();
    }
    const template = root.dataset.streamTemplate || '/jobs/{job_id}/stream';
    const url = template.replace('{job_id}', jobId || 'demo');
    root.dataset.jobId = jobId;
    source = new EventSource(url);
    setStatus('Connecting…');
    source.addEventListener('open', () => setStatus(`Connected (${jobId})`));
    source.addEventListener('error', () => setStatus('Retrying…'));
    source.addEventListener('state', (event) => updateField('state', event.data));
    source.addEventListener('progress', (event) => updateField('progress', event.data));
    source.addEventListener('runtime', (event) => updateField('runtime', event.data));
    source.addEventListener('manifest', (event) => updateField('manifest', event.data));
    source.addEventListener('rendered', (event) => updateField('rendered', event.data));
    source.addEventListener('raw', (event) => updateField('raw', event.data));
    source.addEventListener('links', (event) => updateField('links', event.data));
    source.addEventListener('artifacts', (event) => updateField('artifacts', event.data));
    source.addEventListener('log', (event) => appendLog(event.data));
  };

  const defaultJob = root.dataset.jobId || 'demo';
  connect(defaultJob);

  window.addEventListener('beforeunload', () => {
    if (source) {
      source.close();
    }
  });

  return { connect };
}

function formatManifest(raw) {
  try {
    const parsed = JSON.parse(raw);
    return JSON.stringify(parsed, null, 2);
  } catch {
    return raw;
  }
}

function renderLinks(container, raw) {
  let rows;
  try {
    rows = JSON.parse(raw);
  } catch {
    container.innerHTML = `<p class="placeholder">Invalid links payload</p>`;
    return;
  }

  if (!Array.isArray(rows) || !rows.length) {
    container.innerHTML = `<p class="placeholder">No links yet.</p>`;
    return;
  }

  const header = ['text', 'href', 'source', 'delta'];
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  header.forEach((label) => {
    const th = document.createElement('th');
    th.textContent = label.toUpperCase();
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    header.forEach((key) => {
      const td = document.createElement('td');
      td.textContent = row[key] ?? '—';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.innerHTML = '';
  container.appendChild(table);
}

function renderArtifacts(container, raw) {
  let rows;
  try {
    rows = JSON.parse(raw);
  } catch {
    container.innerHTML = `<li class="placeholder">Invalid artifact payload</li>`;
    return;
  }

  if (!Array.isArray(rows) || !rows.length) {
    container.innerHTML = `<li class="placeholder">No artifacts yet.</li>`;
    return;
  }

  container.innerHTML = '';
  rows.forEach((artifact) => {
    const li = document.createElement('li');
    const left = document.createElement('div');
    const idEl = document.createElement('strong');
    idEl.textContent = artifact.id ?? 'tile';
    const offsetEl = document.createElement('small');
    offsetEl.textContent = artifact.offset ?? '';
    left.append(idEl);
    left.append(document.createElement('br'));
    left.append(offsetEl);
    const right = document.createElement('div');
    right.textContent = artifact.sha ?? '';
    li.append(left, right);
    container.appendChild(li);
  });
}

function initStreamControls(sse) {
  const runButton = document.getElementById('run-job');
  const jobInput = document.getElementById('job-id');
  const root = document.querySelector('[data-stream-root]');
  if (!runButton || !jobInput || !root || !sse) {
    return;
  }

  runButton.addEventListener('click', () => {
    const jobId = jobInput.value.trim() || 'demo';
    sse.connect(jobId);
    alert('Capture submission not implemented yet; reattaching stream to ' + jobId);
  });

  const refreshButton = document.querySelector('[data-links-refresh]'); 
  if (refreshButton) {
    refreshButton.addEventListener('click', async () => {
      const template = root.dataset.linksTemplate || '/jobs/{job_id}/links.json';
      const jobId = root.dataset.jobId || jobInput.value.trim() || 'demo';
      const url = template.replace('{job_id}', jobId);
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderLinks(root.querySelector('[data-sse-field=\"links\"]'), JSON.stringify(data));
      } catch (error) {
        console.error('Failed to refresh links', error);
      }
    });
  }
}

function init() {
  setupTabs();
  const sse = initSseBridge();
  initStreamControls(sse);
}

init();
