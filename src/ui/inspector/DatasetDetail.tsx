// Dataset detail panel — shows labels, sample images/texts, and stats.
// Opened from the inspector when a data node is selected.
// Handles both image datasets (MNIST, CIFAR) and text datasets (IMDb, AG News).

import { useState, useEffect, useRef } from 'react';

interface DatasetInfo {
  name: string;
  description: string;
  labels: string[];
  coarseLabels?: string[];
  channels?: number;
  imageSize?: number[];
  trainSamples: number;
  testSamples?: number;
  diskSize: string;
  sampleImages?: Record<string, number[][][]>;
  isText?: boolean;
  isLanguageModel?: boolean;
  vocabSize?: number;
  sampleTexts?: Record<string, string[]>;
}

interface AugOptions {
  augHFlip?: boolean;
  augRandomCrop?: boolean;
  augColorJitter?: boolean;
}

interface AugPreview {
  original: { pixels: number[][] | number[][][]; channels: number };
  variants: { pixels: number[][] | number[][][]; channels: number }[];
  anyEnabled: boolean;
}

interface Props {
  datasetType: string;
  augOptions?: AugOptions;
  onClose: () => void;
}

export function DatasetDetail({ datasetType, augOptions, onClose }: Props) {
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);
  const [augPreview, setAugPreview] = useState<AugPreview | null>(null);

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch(`http://localhost:8000/dataset/${datasetType}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok') {
          setInfo(data.detail);
        } else {
          setError(data.error ?? 'Failed to load dataset info');
        }
      })
      .catch(() => setError('Cannot connect to backend'))
      .finally(() => setLoading(false));
  }, [datasetType]);

  // Reset page when search changes
  useEffect(() => { setPage(0); }, [search]);

  // Fetch augmentation preview when datasetType or augOptions change
  useEffect(() => {
    if (!augOptions) {
      setAugPreview(null);
      return;
    }
    fetch('http://localhost:8000/augmentation-preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        datasetType,
        augHFlip: augOptions.augHFlip ?? false,
        augRandomCrop: augOptions.augRandomCrop ?? false,
        augColorJitter: augOptions.augColorJitter ?? false,
      }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok' && data.result) setAugPreview(data.result);
      })
      .catch(() => {});
  }, [datasetType, augOptions?.augHFlip, augOptions?.augRandomCrop, augOptions?.augColorJitter]);

  if (loading) {
    return (
      <div className="dataset-detail">
        <div className="dataset-detail-header">
          <span>Loading...</span>
          <button className="dashboard-close" onClick={onClose}>&times;</button>
        </div>
      </div>
    );
  }

  if (error || !info) {
    return (
      <div className="dataset-detail">
        <div className="dataset-detail-header">
          <span>Error</span>
          <button className="dashboard-close" onClick={onClose}>&times;</button>
        </div>
        <div className="dataset-detail-body">
          <div className="dataset-detail-error">{error}</div>
        </div>
      </div>
    );
  }

  const PAGE_SIZE = 20;
  const filteredLabels = info.labels.filter((l) =>
    l.toLowerCase().includes(search.toLowerCase()),
  );
  const totalPages = Math.ceil(filteredLabels.length / PAGE_SIZE);
  const pagedLabels = filteredLabels.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div className="dataset-detail">
      <div className="dataset-detail-header">
        <span className="dataset-detail-title">{info.name}</span>
        <button className="dashboard-close" onClick={onClose}>&times;</button>
      </div>

      <div className="dataset-detail-body">
        {/* Stats */}
        <div className="dataset-detail-stats">
          {info.imageSize && info.channels != null && (
            <div className="dataset-detail-stat">
              <span className="dataset-detail-stat-label">Image Size</span>
              <span>{info.imageSize.join('x')}, {info.channels === 1 ? 'grayscale' : 'RGB'}</span>
            </div>
          )}
          {info.isText && (
            <div className="dataset-detail-stat">
              <span className="dataset-detail-stat-label">Type</span>
              <span>Text</span>
            </div>
          )}
          <div className="dataset-detail-stat">
            <span className="dataset-detail-stat-label">{info.testSamples != null ? 'Train / Test' : 'Train Samples'}</span>
            <span>
              {info.trainSamples.toLocaleString()}
              {info.testSamples != null && ` / ${info.testSamples.toLocaleString()}`}
            </span>
          </div>
          {info.vocabSize != null ? (
            <div className="dataset-detail-stat">
              <span className="dataset-detail-stat-label">Vocab Size</span>
              <span>{info.vocabSize} chars</span>
            </div>
          ) : (
            <div className="dataset-detail-stat">
              <span className="dataset-detail-stat-label">Classes</span>
              <span>{info.labels.length}</span>
            </div>
          )}
          <div className="dataset-detail-stat">
            <span className="dataset-detail-stat-label">Disk Size</span>
            <span>{info.diskSize}</span>
          </div>
        </div>

        {/* Augmentation preview */}
        {augPreview && augPreview.anyEnabled && (
          <div className="dataset-detail-aug">
            <div className="dataset-detail-section-label">Augmentation Preview</div>
            <div className="dataset-detail-aug-grid">
              <div className="dataset-detail-aug-item">
                <MiniImage pixels={augPreview.original.pixels} channels={augPreview.original.channels} />
                <span className="dataset-detail-aug-label">original</span>
              </div>
              {augPreview.variants.map((v, i) => (
                <div key={i} className="dataset-detail-aug-item">
                  <MiniImage pixels={v.pixels} channels={v.channels} />
                  <span className="dataset-detail-aug-label">aug {i + 1}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {augPreview && !augPreview.anyEnabled && (
          <div className="dataset-detail-aug-empty">
            No augmentations enabled — toggle on the data node to preview
          </div>
        )}

        {/* Language model: show sample text directly */}
        {info.isLanguageModel && info.sampleTexts && (
          <div className="dataset-detail-labels">
            {Object.entries(info.sampleTexts).map(([key, texts]) => (
              <div key={key} className="dataset-detail-label-row">
                <span className="dataset-detail-label-name">{key}</span>
                <div className="dataset-detail-texts">
                  {texts.map((text, i) => (
                    <div key={i} className="dataset-detail-text-sample">{text}</div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Classification datasets: labels with search + pagination */}
        {!info.isLanguageModel && (
          <>
            <input
              className="dataset-detail-search"
              type="text"
              placeholder="Search labels..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />

            <div className="dataset-detail-labels">
              {pagedLabels.map((label) => (
                <div key={label} className="dataset-detail-label-row">
                  <span className="dataset-detail-label-name">{label}</span>
                  {info.sampleImages && (
                    <div className="dataset-detail-samples">
                      {(info.sampleImages[label] ?? []).map((pixels, i) => (
                        <MiniImage key={i} pixels={pixels} channels={info.channels ?? 1} />
                      ))}
                    </div>
                  )}
                  {info.sampleTexts && (
                    <div className="dataset-detail-texts">
                      {(info.sampleTexts[label] ?? []).map((text, i) => (
                        <div key={i} className="dataset-detail-text-sample">{text}</div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {filteredLabels.length === 0 && (
                <div className="dataset-detail-empty">No matching labels</div>
              )}
            </div>

            {totalPages > 1 && (
              <div className="dataset-detail-pagination">
                <button
                  className="dataset-detail-page-btn"
                  disabled={page === 0}
                  onClick={() => setPage((p) => p - 1)}
                >
                  Prev
                </button>
                <span className="dataset-detail-page-info">
                  {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, filteredLabels.length)} of {filteredLabels.length}
                </span>
                <button
                  className="dataset-detail-page-btn"
                  disabled={page >= totalPages - 1}
                  onClick={() => setPage((p) => p + 1)}
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// Small image thumbnail
function MiniImage({ pixels, channels }: { pixels: number[][] | number[][][]; channels: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !pixels || pixels.length === 0) return;

    const h = pixels.length;
    const w = Array.isArray(pixels[0][0]) ? (pixels[0] as number[][]).length : (pixels[0] as number[]).length;
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const isRGB = channels >= 3;
    const imageData = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        if (isRGB) {
          const px = (pixels as number[][][])[y][x];
          imageData.data[idx] = px[0];
          imageData.data[idx + 1] = px[1];
          imageData.data[idx + 2] = px[2];
        } else {
          const v = (pixels as number[][])[y][x];
          imageData.data[idx] = v;
          imageData.data[idx + 1] = v;
          imageData.data[idx + 2] = v;
        }
        imageData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [pixels, channels]);

  return (
    <canvas
      ref={canvasRef}
      className="dataset-detail-thumb"
      style={{ imageRendering: 'pixelated' }}
    />
  );
}
