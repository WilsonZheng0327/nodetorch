// AgentSettings — configure which LLM provider the agent uses. Talks to the
// backend config REST routes; API keys are stored server-side (the form's key
// field is write-only and shows blank even when a key is set).

import './AgentSettings.css';

import { useEffect, useState } from 'react';
import { X, Check, Loader2 } from 'lucide-react';
import { API_BASE } from '../../api/base';

const API = API_BASE;

interface ProviderField {
  key: string;
  label: string;
  placeholder?: string;
  required?: boolean;
  secret?: boolean;
}
interface Provider {
  name: string;
  label: string;
  fields: ProviderField[];
}
interface PublicConfig {
  provider: string;
  model: string;
  base_url: string;
  hasApiKey: boolean;
}

interface Props {
  open: boolean;
  onClose: () => void;
  onSaved?: () => void;
}

export function AgentSettings({ open, onClose, onSaved }: Props) {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [provider, setProvider] = useState('');
  const [values, setValues] = useState<Record<string, string>>({});
  const [hasApiKey, setHasApiKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [feedback, setFeedback] = useState<{ ok: boolean; msg: string } | null>(null);

  useEffect(() => {
    if (!open) return;
    setFeedback(null);
    Promise.all([
      fetch(`${API}/agent/providers`).then((r) => r.json()),
      fetch(`${API}/agent/config`).then((r) => r.json()),
    ])
      .then(([provData, cfg]: [{ providers: Provider[] }, PublicConfig]) => {
        setProviders(provData.providers);
        setProvider(cfg.provider);
        setValues({ base_url: cfg.base_url || '', model: cfg.model || '', api_key: '' });
        setHasApiKey(cfg.hasApiKey);
      })
      .catch(() => setFeedback({ ok: false, msg: 'Could not reach the backend.' }));
  }, [open]);

  if (!open) return null;

  const current = providers.find((p) => p.name === provider);

  async function save() {
    setSaving(true);
    setFeedback(null);
    try {
      const body: Record<string, string> = { provider };
      for (const f of current?.fields ?? []) {
        // Blank api_key = keep existing key on the server.
        if (f.key === 'api_key' && !values.api_key) continue;
        body[f.key] = values[f.key] ?? '';
      }
      const res = await fetch(`${API}/agent/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }).then((r) => r.json());
      if (res.status === 'ok') {
        setHasApiKey(res.config.hasApiKey);
        setValues((v) => ({ ...v, api_key: '' }));
        setFeedback({ ok: true, msg: 'Saved.' });
        onSaved?.();
      } else {
        setFeedback({ ok: false, msg: res.error || 'Save failed.' });
      }
    } catch (e) {
      setFeedback({ ok: false, msg: e instanceof Error ? e.message : 'Save failed.' });
    } finally {
      setSaving(false);
    }
  }

  async function test() {
    setTesting(true);
    setFeedback(null);
    try {
      const res = await fetch(`${API}/agent/test`, { method: 'POST' }).then((r) => r.json());
      setFeedback(
        res.status === 'ok'
          ? { ok: true, msg: 'Connection works.' }
          : { ok: false, msg: res.error || 'Connection failed.' },
      );
    } catch (e) {
      setFeedback({ ok: false, msg: e instanceof Error ? e.message : 'Connection failed.' });
    } finally {
      setTesting(false);
    }
  }

  return (
    // Close only when the press STARTS on the backdrop itself — not when a
    // drag (e.g. selecting text) begins inside the panel and ends out here.
    <div
      className="agent-settings-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="agent-settings">
        <div className="agent-settings-header">
          <span>AI Assistant — Provider</span>
          <button className="agent-settings-x" onClick={onClose} title="Close">
            <X size={16} />
          </button>
        </div>

        <div className="agent-settings-body">
          <label className="agent-settings-field">
            <span>Provider</span>
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              {providers.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>

          {current?.fields.map((f) => (
            <label className="agent-settings-field" key={f.key}>
              <span>
                {f.label}
                {f.required ? ' *' : ''}
              </span>
              <input
                type={f.secret ? 'password' : 'text'}
                placeholder={
                  f.key === 'api_key' && hasApiKey
                    ? '•••••••• (a key is set — leave blank to keep)'
                    : f.placeholder
                }
                value={values[f.key] ?? ''}
                onChange={(e) => setValues((v) => ({ ...v, [f.key]: e.target.value }))}
              />
            </label>
          ))}

          {feedback && (
            <div className={`agent-settings-feedback ${feedback.ok ? 'ok' : 'err'}`}>
              {feedback.ok && <Check size={13} />}
              {feedback.msg}
            </div>
          )}
        </div>

        <div className="agent-settings-actions">
          <button className="agent-settings-btn" onClick={test} disabled={testing || saving}>
            {testing ? <Loader2 size={14} className="spin" /> : null} Test connection
          </button>
          <button
            className="agent-settings-btn agent-settings-btn-primary"
            onClick={save}
            disabled={saving || testing}
          >
            {saving ? <Loader2 size={14} className="spin" /> : null} Save
          </button>
        </div>
      </div>
    </div>
  );
}
