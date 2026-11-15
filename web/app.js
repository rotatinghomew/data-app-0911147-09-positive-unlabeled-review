const operations = [
  {value: 'review_queue', label: '模型排序队列（正例未标注学习）'},
  {value: 'queue', label: '遗留操作：最早优先'}
];
const assumptions = [
  {value: 'stratified', label: '分层上报假设（依批次与团队）'},
  {value: 'scar', label: 'SCAR（恒定上报率）'}
];
const defaults = {operation: 'review_queue', daily_capacity: 10, assumption: 'stratified'};
const sources = ['items', 'positive_reports', 'audits'];
let study = null;
let latestRequest = {...defaults};

async function api(route, payload) {
  const options = payload === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
  };
  const response = await fetch(route, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || String(response.status));
  return result;
}

function display(target, value) {
  document.getElementById(target).textContent = JSON.stringify(value, null, 2);
}

function renderTable(source) {
  const holder = document.getElementById('records');
  holder.replaceChildren();
  const raw = study[source];
  const rows = Array.isArray(raw) ? raw : Object.entries(raw).map(([key, value]) => ({key, value}));
  if (!rows.length) { holder.textContent = '当前集合没有记录'; return; }
  const columns = [...new Set(rows.flatMap(row => Object.keys(row)))];
  const table = document.createElement('table');
  const head = table.createTHead().insertRow();
  for (const column of columns) {
    const cell = document.createElement('th'); cell.textContent = column; head.append(cell);
  }
  const body = table.createTBody();
  for (const row of rows) {
    const rendered = body.insertRow();
    for (const column of columns) {
      const value = row[column];
      rendered.insertCell().textContent = value === null ? '缺失' : typeof value === 'object' ? JSON.stringify(value) : String(value ?? '');
    }
  }
  holder.append(table);
  document.getElementById('record-count').textContent = String(rows.length) + ' 条记录';
}

function renderControls() {
  const form = document.getElementById('action-form');
  const operationLabel = document.createElement('label');
  operationLabel.textContent = '复核操作';
  const operationSelect = document.createElement('select');
  operationSelect.name = 'operation';
  for (const {value, label} of operations) {
    const option = document.createElement('option');
    option.value = value; option.textContent = label;
    option.selected = defaults.operation === value;
    operationSelect.append(option);
  }
  operationLabel.append(operationSelect);
  const capacityLabel = document.createElement('label');
  capacityLabel.textContent = '每天可复核数量';
  const capacityInput = document.createElement('input');
  capacityInput.type = 'number'; capacityInput.min = '0';
  capacityInput.name = 'daily_capacity'; capacityInput.value = defaults.daily_capacity;
  capacityLabel.append(capacityInput);
  const assumptionLabel = document.createElement('label');
  assumptionLabel.textContent = '选择假设';
  const assumptionSelect = document.createElement('select');
  assumptionSelect.name = 'assumption';
  for (const {value, label} of assumptions) {
    const option = document.createElement('option');
    option.value = value; option.textContent = label;
    option.selected = defaults.assumption === value;
    assumptionSelect.append(option);
  }
  assumptionLabel.append(assumptionSelect);
  const button = document.createElement('button');
  button.type = 'submit'; button.textContent = '生成复核队列';
  form.append(operationLabel, capacityLabel, assumptionLabel, button);
  const syncAssumption = () => { assumptionSelect.disabled = operationSelect.value !== 'review_queue'; };
  operationSelect.addEventListener('change', syncAssumption);
  syncAssumption();
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const capacity = Number(capacityInput.value);
    latestRequest = operationSelect.value === 'review_queue'
      ? {operation: 'review_queue', daily_capacity: capacity, assumption: assumptionSelect.value}
      : {operation: 'queue', limit: capacity};
    try { display('result', await api('/api/action', latestRequest)); }
    catch (error) { display('result', {error: error.message}); }
  });
}

async function refreshNotes() {
  const result = await api('/api/notes');
  const holder = document.getElementById('notes'); holder.replaceChildren();
  for (const note of result.notes) {
    const item = document.createElement('li'); item.textContent = note.text + ' · ' + note.created_at; holder.append(item);
  }
}

async function start() {
  study = await api('/api/study');
  display('summary', await api('/api/summary'));
  const review = await api('/api/review');
  display('learning', {
    label_states: review.label_states, label_notice: review.label_notice,
    conflicts: review.conflicts, conflict_notice: review.conflict_notice,
    features: review.features, leakage_guards: review.leakage_guards,
    assumptions: review.assumptions, prevalence: review.prevalence,
    sensitivity: review.sensitivity
  });
  display('validation', {validation: review.validation, suppressed_metrics: review.suppressed_metrics});
  display('feedback', review.feedback);
  const selector = document.getElementById('source');
  for (const source of sources) {
    const option = document.createElement('option'); option.value = source; option.textContent = source; selector.append(option);
  }
  selector.addEventListener('change', () => renderTable(selector.value));
  renderTable(sources[0]); renderControls();
  document.getElementById('note-form').addEventListener('submit', async event => {
    event.preventDefault();
    const field = document.getElementById('note');
    try { await api('/api/notes', {text: field.value, selection: latestRequest}); field.value = ''; await refreshNotes(); }
    catch (error) { display('result', {error: error.message}); }
  });
  await refreshNotes();
}

start().catch(error => display('summary', {error: error.message}));
