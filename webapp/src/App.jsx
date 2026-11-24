import { useEffect, useMemo, useState } from 'react'

const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`
const api = async (url, opts) => {
  const r = await fetch(url, opts)
  const body = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(body.detail || `Request failed (${r.status})`)
  return body
}

export default function App() {
  const [tab, setTab] = useState('screen')
  return (
    <div className="shell">
      <header>
        <div>
          <h1>Nutrient Deficiency Screener</h1>
          <p className="sub">Machine-learning risk estimate from self-reported information · VTU Major Project Phase-2 · Sir MVIT</p>
        </div>
        <nav>
          <button className={tab === 'screen' ? 'on' : ''} onClick={() => setTab('screen')}>Screening</button>
          <button className={tab === 'models' ? 'on' : ''} onClick={() => setTab('models')}>Model comparison</button>
        </nav>
      </header>
      <p className="banner" role="note"><b>Who this is for:</b> the model was trained on about 6,200 people aged <b>12–85</b>, women and men, from a US national health survey (NHANES 2005–06). It has not been validated for other populations.</p>
      {tab === 'screen' ? <Screening /> : <Models />}
    </div>
  )
}

/* ------------------------------------------------------------------ screening */
function Screening() {
  const [schema, setSchema] = useState([])
  const [samples, setSamples] = useState([])
  const [answers, setAnswers] = useState({})
  const [report, setReport] = useState(null)
  const [actual, setActual] = useState(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api('/api/schema').then(setSchema).catch(e => setErr(e.message))
    api('/api/samples').then(setSamples).catch(() => {})
  }, [])

  const groups = useMemo(() => {
    const g = {}
    schema.forEach(f => (g[f.group] ||= []).push(f))
    return Object.entries(g)
  }, [schema])

  const set = (name, v) => setAnswers(a => ({ ...a, [name]: v }))
  const submit = async (inputs = answers, truth = null) => {
    setBusy(true); setErr(''); setActual(truth)
    const clean = Object.fromEntries(Object.entries(inputs).filter(([, v]) => v !== '' && v !== null && v !== undefined))
    try { setReport(await api('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ answers: clean }) })) }
    catch (e) { setReport(null); setErr(e.message) }
    setBusy(false)
    setTimeout(() => document.getElementById('result')?.scrollIntoView({ behavior: 'smooth' }), 50)
  }
  const loadSample = s => {
    const filled = Object.fromEntries(Object.entries(s.inputs).map(([k, v]) => [k, v ?? '']))
    setAnswers(filled); submit(filled, s.actual)
  }

  return (
    <main className="grid">
      <section className="card">
        <h2>Your details</h2>
        <p className="muted">Every question is optional — blanks are filled with a typical value, so more answers means a more personal estimate.</p>
        {samples.length > 0 && (
          <div className="samples">
            <span className="muted">Demo: real unseen people from the test set</span>
            {samples.map(s => <button key={s.id} className="chip" onClick={() => loadSample(s)}>Person {s.id}</button>)}
          </div>
        )}
        <form onSubmit={e => { e.preventDefault(); submit() }}>
          {groups.map(([name, fields], i) => (
            <details key={name} open={i < 3}>
              <summary>{name}</summary>
              <div className="fields">{fields.map(f => <Field key={f.name} f={f} value={answers[f.name] ?? ''} onChange={v => set(f.name, v)} />)}</div>
            </details>
          ))}
          <div className="actions">
            <button className="primary" disabled={busy}>{busy ? 'Analysing…' : 'Estimate my risk'}</button>
            <button type="button" className="ghost" onClick={() => { setAnswers({}); setReport(null); setErr('') }}>Clear</button>
          </div>
        </form>
        {err && <p className="error" role="alert">{err}</p>}
      </section>
      <section id="result">{report ? <Report r={report} actual={actual} /> : <Placeholder />}</section>
    </main>
  )
}

function Field({ f, value, onChange }) {
  const id = `f-${f.name}`
  return (
    <label className="field" htmlFor={id}>
      <span>{f.label}{f.unit ? <em> ({f.unit})</em> : null}</span>
      {f.kind === 'choice'
        ? <select id={id} value={value} onChange={e => onChange(e.target.value)}>
            <option value="">— not answered —</option>
            {f.options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        : <input id={id} type="number" step="any" min={f.min} max={f.max} value={value} placeholder={f.computed ? 'auto' : ''} onChange={e => onChange(e.target.value)} />}
      {f.help && <small>{f.help}</small>}
    </label>
  )
}

function Placeholder() {
  return <div className="card empty"><h2>Your report appears here</h2><p className="muted">Fill in what you know — or pick a demo person — and press <b>Estimate my risk</b>. The report shows the predicted pattern, the factors behind it (SHAP), and food-based suggestions.</p></div>
}

/* ------------------------------------------------------------------ report */
function Report({ r, actual }) {
  const none = r.predicted === 'No Deficiency'
  const maxScore = Math.max(...r.scores.map(s => s.score), 0.01)
  return (
    <div className="stack">
      <div className={`card hero ${none ? 'ok' : 'warn'}`}>
        <span className="tag">Predicted pattern</span>
        <h2>{r.kb.title}</h2>
        <p>Model score <b>{pct(r.confidence)}</b> · second most likely: <b>{r.runner_up}</b></p>
        {actual && <p className="truth">Demo only — this person's actual lab-based label: <b>{actual}</b> {actual === r.predicted ? '✓ matches' : '✗ does not match'}</p>}
        <p className="muted">{r.kb.summary}</p>
      </div>

      <div className="card">
        <h3>Model scores vs. how common each pattern is</h3>
        <p className="muted small">Bars: the model's score. Tick: share of the training population in that pattern. Scores come from a model trained on <i>balanced</i> classes, so they are relative signals, not real-world probabilities.</p>
        {r.scores.map(s => (
          <div className="bar-row" key={s.label}>
            <span className="bar-label">{s.label}</span>
            <div className="bar"><div className="fill" style={{ width: `${(s.score / maxScore) * 100}%` }} /><i className="tick" style={{ left: `${Math.min(s.prevalence / maxScore, 1) * 100}%` }} title={`Prevalence ${pct(s.prevalence, 1)}`} /></div>
            <span className="bar-val">{pct(s.score)}</span>
          </div>
        ))}
      </div>

      <div className="card">
        <h3>Why this result? <span className="muted small">(SHAP — top factors for “{r.predicted}”)</span></h3>
        <ul className="drivers">
          {r.drivers.map(d => (
            <li key={d.feature} className={d.direction}>
              <span className="arrow">{d.direction === 'raises' ? '▲' : '▼'}</span>
              <div><b>{d.feature}</b><div className="muted small">Your answer: {d.your_value}</div></div>
              <span className="muted small">{d.direction} the score</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="card">
        <h3>What you can do</h3>
        <h4>Food sources</h4><ul>{r.kb.foods.map(x => <li key={x}>{x}</li>)}</ul>
        <h4>Tips</h4><ul>{r.kb.tips.map(x => <li key={x}>{x}</li>)}</ul>
        <h4>When to see a clinician</h4><p>{r.kb.see_doctor}</p>
        <p className="muted small">Label definition used in training: {r.kb.cutoff}.</p>
      </div>

      <div className="card caution">
        <h3>How far to trust this</h3>
        <p>Model: <b>{r.model.name}</b>, trained on {r.model.n_train.toLocaleString()} people. On people it had never seen it reached macro-F1 <b>{r.model.holdout_macro_f1.toFixed(2)}</b> and AUC <b>{r.model.holdout_auc.toFixed(2)}</b> — better than chance, but far from a diagnosis.</p>
        <p className="small">{r.disclaimer}</p>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ model comparison */
function Models() {
  const [m, setM] = useState(null)
  const [err, setErr] = useState('')
  useEffect(() => { api('/api/model-info').then(setM).catch(e => setErr(e.message)) }, [])
  if (err) return <p className="error">{err}</p>
  if (!m) return <p className="muted">Loading…</p>
  const rows = [...m.comparison].sort((a, b) => a.rank - b.rank)
  const top = Math.max(...rows.map(r => r.f1_mean))
  return (
    <main className="stack wide">
      <div className="card">
        <h2>Nine classifiers, one fair test</h2>
        <p className="muted">All nine were trained on identical data under the same stratified 5-fold cross-validation ({m.n_folds} folds in total). Preprocessing, SMOTE and feature selection were fitted inside each training fold, so no validation data leaked into training. Ranked by macro-F1, AUC-ROC as tie-breaker.</p>
        <div className="scroll">
          <table>
            <thead><tr><th>#</th><th>Model</th><th>Macro-F1</th><th></th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>AUC-ROC</th><th>vs. winner (Holm p)</th></tr></thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.model} className={r.model === m.best ? 'best' : ''}>
                  <td>{r.rank}</td><td><b>{r.model}</b>{r.model === m.best && <span className="tag">best</span>}</td>
                  <td>{r.f1_mean.toFixed(3)} <span className="muted small">±{r.f1_std.toFixed(3)}</span></td>
                  <td className="minibar"><div style={{ width: `${(r.f1_mean / top) * 100}%` }} /></td>
                  <td>{r.accuracy_mean.toFixed(3)}</td><td>{r.precision_mean.toFixed(3)}</td><td>{r.recall_mean.toFixed(3)}</td><td>{r.auc_mean.toFixed(3)}</td>
                  <td>{r.model === m.best ? '—' : (m.holm_p[r.model] < 0.05 ? <b>{m.holm_p[r.model] < 0.001 ? '<0.001' : m.holm_p[r.model].toFixed(3)}</b> : m.holm_p[r.model].toFixed(2))}</td>
                </tr>
              ))}
              <tr className="baseline"><td></td><td><i>Always guess “No Deficiency”</i></td><td>{m.baseline_f1.toFixed(3)}</td><td></td><td>{m.baseline_accuracy.toFixed(3)}</td><td colSpan="4" className="muted small">reference, not one of the nine</td></tr>
            </tbody>
          </table>
        </div>
        <p className="muted small">Bold p-values: the winner is significantly better than that model (paired Wilcoxon over folds, Holm-corrected, p&lt;0.05). Friedman test across all nine: p = {m.friedman_p.toExponential(1)}.</p>
      </div>
      <div className="card">
        <h3>Final check on unseen people</h3>
        <p>The winner was retrained and tested <b>once</b> on a 15% hold-out set that took no part in model selection: accuracy <b>{pct(m.holdout.accuracy, 1)}</b>, macro-F1 <b>{m.holdout.f1.toFixed(3)}</b>, macro-AUC <b>{m.holdout.auc.toFixed(3)}</b>.</p>
        <p className="muted">Accuracy sits below the {pct(m.baseline_accuracy, 1)} of the “always No Deficiency” guess because the classes were balanced with SMOTE: the models deliberately trade overall accuracy for finding minority deficiencies. That is why macro-F1 and AUC, not accuracy, decide the ranking.</p>
      </div>
    </main>
  )
}
