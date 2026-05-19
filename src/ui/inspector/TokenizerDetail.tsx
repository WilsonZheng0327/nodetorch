// TokenizerDetail — vocab inspector for tokenizer nodes.
// Calls POST /tokenizer/preview to get the vocab+stats for the upstream
// corpus, then renders a searchable/sortable table.

import { useState, useEffect, useMemo } from 'react';

type VocabEntry = { id: number; token: string; freq: number; special?: boolean };

type PreviewResult = {
  vocab: VocabEntry[];
  stats: Record<string, number | string | boolean>;
  merges?: [string, string][];
  sampleEncode?: { id: number; token: string }[];
};

type SortMode = 'freq-desc' | 'freq-asc' | 'id-asc' | 'alpha' | 'length-desc';

const SORT_OPTIONS: { value: SortMode; label: string }[] = [
  { value: 'freq-desc', label: 'Frequency (high → low)' },
  { value: 'freq-asc', label: 'Frequency (low → high)' },
  { value: 'id-asc', label: 'Token ID' },
  { value: 'alpha', label: 'Alphabetical' },
  { value: 'length-desc', label: 'Token length' },
];

interface Props {
  nodeType: string;
  nodeProperties: Record<string, any>;
  datasetType: string | null;
}

export function TokenizerDetail({ nodeType, nodeProperties, datasetType }: Props) {
  const [data, setData] = useState<PreviewResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [sort, setSort] = useState<SortMode>('freq-desc');
  const [search, setSearch] = useState('');
  const [showMerges, setShowMerges] = useState(false);
  const [sampleText, setSampleText] = useState('');
  const [sampleResult, setSampleResult] = useState<{ id: number; token: string }[] | null>(null);

  const isBpe = nodeType === 'ml.preprocessing.tokenizer_bpe';

  // Fetch vocab whenever inputs change. Sample-encoding is fetched separately.
  useEffect(() => {
    if (!datasetType) return;
    setLoading(true);
    setError(null);
    fetch('http://localhost:8000/tokenizer/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        nodeType,
        properties: nodeProperties,
        datasetType,
      }),
    })
      .then((r) => r.json())
      .then((res) => {
        if (res.status === 'ok') setData(res.result);
        else setError(res.error ?? 'Failed to build vocab');
      })
      .catch(() => setError('Cannot connect to backend'))
      .finally(() => setLoading(false));
  }, [nodeType, datasetType, JSON.stringify(nodeProperties)]);

  const sortedVocab = useMemo(() => {
    if (!data) return [];
    const filtered = search
      ? data.vocab.filter((v) => v.token.toLowerCase().includes(search.toLowerCase()))
      : data.vocab;
    const arr = [...filtered];
    switch (sort) {
      case 'freq-desc':
        arr.sort((a, b) => b.freq - a.freq);
        break;
      case 'freq-asc':
        arr.sort((a, b) => a.freq - b.freq);
        break;
      case 'id-asc':
        arr.sort((a, b) => a.id - b.id);
        break;
      case 'alpha':
        arr.sort((a, b) => a.token.localeCompare(b.token));
        break;
      case 'length-desc':
        arr.sort((a, b) => b.token.length - a.token.length);
        break;
    }
    return arr;
  }, [data, sort, search]);

  const handleEncodeSample = () => {
    if (!datasetType || !sampleText.trim()) {
      setSampleResult(null);
      return;
    }
    fetch('http://localhost:8000/tokenizer/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        nodeType,
        properties: nodeProperties,
        datasetType,
        sampleText,
      }),
    })
      .then((r) => r.json())
      .then((res) => {
        if (res.status === 'ok') setSampleResult(res.result.sampleEncode ?? []);
        else setError(res.error ?? 'Failed to encode sample');
      })
      .catch(() => setError('Cannot connect to backend'));
  };

  if (!datasetType) {
    return (
      <div className="tokenizer-detail">
        <div className="heatmap-note">
          Connect a text dataset upstream of this tokenizer to see the vocabulary.
        </div>
      </div>
    );
  }

  return (
    <div className="tokenizer-detail">
      {loading && <div className="layer-detail-loading">Building vocabulary from corpus...</div>}
      {error && <div className="layer-detail-error">{error}</div>}

      {data && (
        <>
          <div className="tokenizer-stats">
            {Object.entries(data.stats).map(([k, v]) => (
              <div key={k} className="tokenizer-stat">
                <div className="tokenizer-stat-label">{formatStatLabel(k)}</div>
                <div className="tokenizer-stat-value">{formatStatValue(v)}</div>
              </div>
            ))}
          </div>

          {isBpe && data.merges && data.merges.length > 0 && (
            <div className="tokenizer-tabs">
              <button
                className={!showMerges ? 'tokenizer-tab active' : 'tokenizer-tab'}
                onClick={() => setShowMerges(false)}
              >
                Vocabulary ({data.vocab.length})
              </button>
              <button
                className={showMerges ? 'tokenizer-tab active' : 'tokenizer-tab'}
                onClick={() => setShowMerges(true)}
              >
                Merge Rules ({data.merges.length})
              </button>
            </div>
          )}

          {!showMerges && (
            <>
              <div className="tokenizer-controls">
                <input
                  type="text"
                  className="tokenizer-search"
                  placeholder="Search tokens..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
                <select
                  className="tokenizer-sort"
                  value={sort}
                  onChange={(e) => setSort(e.target.value as SortMode)}
                >
                  {SORT_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </div>

              <div className="tokenizer-vocab-list">
                <div className="tokenizer-vocab-header">
                  <span className="tokenizer-col-id">ID</span>
                  <span className="tokenizer-col-token">Token</span>
                  <span className="tokenizer-col-freq">Freq</span>
                </div>
                {sortedVocab.length === 0 && (
                  <div className="layer-detail-loading">No tokens match "{search}"</div>
                )}
                {sortedVocab.slice(0, 500).map((v) => (
                  <div key={v.id} className={v.special ? 'tokenizer-vocab-row special' : 'tokenizer-vocab-row'}>
                    <span className="tokenizer-col-id">{v.id}</span>
                    <span className="tokenizer-col-token">{renderToken(v.token)}</span>
                    <span className="tokenizer-col-freq">{v.freq ? v.freq.toLocaleString() : '—'}</span>
                  </div>
                ))}
                {sortedVocab.length > 500 && (
                  <div className="heatmap-note">Showing first 500 of {sortedVocab.length.toLocaleString()} tokens. Use search to narrow.</div>
                )}
              </div>
            </>
          )}

          {showMerges && data.merges && (
            <div className="tokenizer-vocab-list">
              <div className="tokenizer-vocab-header">
                <span className="tokenizer-col-id">Rank</span>
                <span className="tokenizer-col-token">Merge</span>
                <span className="tokenizer-col-freq">→ Result</span>
              </div>
              {data.merges.slice(0, 500).map(([a, b], i) => (
                <div key={i} className="tokenizer-vocab-row">
                  <span className="tokenizer-col-id">{i + 1}</span>
                  <span className="tokenizer-col-token">{renderToken(a)} + {renderToken(b)}</span>
                  <span className="tokenizer-col-freq">{renderToken(a + b)}</span>
                </div>
              ))}
              {data.merges.length > 500 && (
                <div className="heatmap-note">Showing first 500 of {data.merges.length.toLocaleString()} merges.</div>
              )}
            </div>
          )}

          <div className="tokenizer-sample">
            <div className="detail-section-title">Try Encoding</div>
            <div className="tokenizer-sample-row">
              <input
                type="text"
                className="tokenizer-sample-input"
                placeholder="Type text to tokenize..."
                value={sampleText}
                onChange={(e) => setSampleText(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleEncodeSample()}
              />
              <button className="layer-detail-action-btn" onClick={handleEncodeSample}>
                Encode
              </button>
            </div>
            {sampleResult && (
              <div className="tokenizer-sample-chips">
                {sampleResult.length === 0 && <span className="heatmap-note">Empty.</span>}
                {sampleResult.map((t, i) => (
                  <span key={i} className="tokenizer-chip" title={`ID ${t.id}`}>
                    <span className="tokenizer-chip-token">{renderToken(t.token)}</span>
                    <span className="tokenizer-chip-id">{t.id}</span>
                  </span>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function renderToken(token: string): string {
  if (token === ' ') return '␣';
  if (token === '\n') return '↵';
  if (token === '\t') return '⇥';
  return token;
}

function formatStatLabel(key: string): string {
  // camelCase → Title Case
  return key
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, (s) => s.toUpperCase())
    .trim();
}

function formatStatValue(v: number | string | boolean): string {
  if (typeof v === 'number') {
    if (v > 0 && v < 1) return (v * 100).toFixed(1) + '%';
    return v.toLocaleString();
  }
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  return String(v);
}
