function setupTabs() {
  const tabButtons = Array.from(document.querySelectorAll('[data-tab-target]'));
  const panels = new Map(Array.from(document.querySelectorAll('[data-tab-panel]')).map((panel) => [panel.dataset.tabPanel, panel]));

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

function setupSseStatus() {
  const statusEl = document.querySelector('#job-sse-status');
  if (!statusEl) {
    return;
  }

  const setStatus = (text) => {
    statusEl.textContent = text;
  };

  document.body.addEventListener('htmx:sseOpen', () => setStatus('Connected'));
  document.body.addEventListener('htmx:sseClose', () => setStatus('Closed'));
  document.body.addEventListener('htmx:sseError', () => setStatus('Error'));
}

function preventUnimplementedFormSubmit() {
  const runButton = document.getElementById('run-job');
  if (!runButton) {
    return;
  }

  runButton.addEventListener('click', () => {
    alert('Job submission UI is stubbed until backend /jobs endpoint lands.');
  });
}

function init() {
  setupTabs();
  setupSseStatus();
  preventUnimplementedFormSubmit();
}

init();
