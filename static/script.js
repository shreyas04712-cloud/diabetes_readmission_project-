/* script.js – Async form handler and results renderer for ClinIQ */

(function () {
  'use strict';

  const form          = document.getElementById('predict-form');
  const submitBtn     = document.getElementById('submit-btn');
  const btnText       = submitBtn.querySelector('.btn-text');
  const btnLoader     = submitBtn.querySelector('.btn-loader');
  const resultsSection= document.getElementById('results-section');
  const errorBanner   = document.getElementById('error-banner');
  const errorText     = document.getElementById('error-text');

  // ── Collect every form field into a plain object ────────────────────
  function collectFormData() {
    const data = {};
    const elements = form.elements;
    for (let el of elements) {
      if (!el.name) continue;
      data[el.name] = el.value.trim();
    }
    return data;
  }

  // ── Serialize numbers that should be numbers ────────────────────────
  const NUMERIC_FIELDS = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_diagnoses', 'number_outpatient',
    'number_emergency', 'number_inpatient'
  ];
  function coerceTypes(data) {
    const out = { ...data };
    NUMERIC_FIELDS.forEach(f => {
      if (f in out) out[f] = parseFloat(out[f]) || 0;
    });
    return out;
  }

  // ── Toggle loading state ────────────────────────────────────────────
  function setLoading(on) {
    submitBtn.disabled = on;
    btnText.hidden     = on;
    btnLoader.hidden   = !on;
  }

  // ── Show / hide error banner ────────────────────────────────────────
  function showError(msg) {
    errorText.textContent = '⚠ ' + msg;
    errorBanner.hidden = false;
  }
  function clearError() {
    errorBanner.hidden = true;
    errorText.textContent = '';
  }

  // ── Animate bar after a short tick (lets browser paint first) ───────
  function animateBar(barEl, pct, isHigh) {
    barEl.classList.remove('bar-high', 'bar-low');
    barEl.style.width = '0%';
    barEl.classList.add(isHigh ? 'bar-high' : 'bar-low');
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        barEl.style.width = pct + '%';
      });
    });
  }

  // ── Render one model card ────────────────────────────────────────────
  function renderCard(suffix, result) {
    const card     = document.getElementById('card-' + suffix);
    const iconEl   = document.getElementById('icon-' + suffix);
    const labelEl  = document.getElementById('label-' + suffix);
    const barEl    = document.getElementById('bar-' + suffix);
    const probEl   = document.getElementById('prob-' + suffix);

    const isReadmit = result.label === 1;
    const prob      = result.probability;           // already 0–100

    // icon
    iconEl.textContent = isReadmit ? '🚨' : '✅';

    // label
    labelEl.textContent = isReadmit ? 'Readmission Likely' : 'No Readmission';
    labelEl.className = 'verdict-label ' + (isReadmit ? 'readmit' : 'noreadmit');

    // bar
    animateBar(barEl, Math.min(prob, 100), isReadmit);

    // probability text
    probEl.textContent = prob.toFixed(1) + '%';

    // card border / glow
    card.classList.remove('state-readmit', 'state-noreadmit');
    card.classList.add(isReadmit ? 'state-readmit' : 'state-noreadmit');
  }

  // ── Main form submit handler ─────────────────────────────────────────
  form.addEventListener('submit', async function (e) {
    e.preventDefault();
    clearError();
    resultsSection.hidden = true;

    const raw     = collectFormData();
    const payload = coerceTypes(raw);

    setLoading(true);

    try {
      const response = await fetch('/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload)
      });

      const json = await response.json();

      if (!response.ok || json.status !== 'ok') {
        throw new Error(json.message || 'Prediction failed – please check your inputs.');
      }

      const preds = json.predictions;

      // render each card
      renderCard('catboost', preds.catboost);
      renderCard('rf',        preds.random_forest);
      renderCard('stacking',  preds.stacking);

      // reveal results section
      resultsSection.hidden = false;
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
      showError(err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  });

})();
