// Pretrained model viz: model info card + before/after

import type { PretrainedTransformation } from '../types';
import { FeatureMapsGrid, VectorBars, Histogram } from './shared';

export function PretrainedViz({ t }: { t: PretrainedTransformation }) {
  return (
    <div className="tfm-pretrained">
      {/* Model info card */}
      <div className="tfm-pretrained-card">
        <div className="tfm-pretrained-header">
          <span className="tfm-pretrained-name">{t.modelName}</span>
          <span className={`tfm-pretrained-badge ${t.frozen ? 'tfm-pretrained-frozen' : 'tfm-pretrained-tuning'}`}>
            {t.frozen ? 'Frozen' : 'Fine-tuning'}
          </span>
        </div>
        <div className="tfm-pretrained-meta">
          <div className="tfm-pretrained-row">
            <span className="tfm-pretrained-key">Pretrained on</span>
            <span className="tfm-pretrained-val">{t.pretrainedOn}</span>
          </div>
          <div className="tfm-pretrained-row">
            <span className="tfm-pretrained-key">Accuracy</span>
            <span className="tfm-pretrained-val">{t.topAcc}</span>
          </div>
          <div className="tfm-pretrained-row">
            <span className="tfm-pretrained-key">Parameters</span>
            <span className="tfm-pretrained-val">{t.totalParams} total, {t.trainableParams ?? '0'} trainable</span>
          </div>
          <div className="tfm-pretrained-row">
            <span className="tfm-pretrained-key">Mode</span>
            <span className="tfm-pretrained-val">{t.mode === 'features' ? 'Feature extractor → 512-dim' : 'Full model → 1000 ImageNet classes'}</span>
          </div>
        </div>
      </div>

      {/* Architecture */}
      {t.architecture && (
        <div className="tfm-section">
          <div className="tfm-section-title">Architecture</div>
          <div className="tfm-pretrained-arch">
            {t.architecture.map((layer, i) => (
              <div key={i} className="tfm-pretrained-arch-row">
                <span className="tfm-pretrained-arch-name">{layer.name}</span>
                <span className="tfm-pretrained-arch-detail">{layer.detail}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Before / After */}
      <div className="tfm-before-after">
        <div className="tfm-ba-pane">
          <div className="tfm-ba-label">Input {t.inputShape ? `[${t.inputShape.join(', ')}]` : ''}</div>
          {t.inputFmaps && <FeatureMapsGrid data={t.inputFmaps} />}
        </div>
        <div className="tfm-ba-divider" />
        <div className="tfm-ba-pane">
          <div className="tfm-ba-label">Output {t.outputShape ? `[${t.outputShape.join(', ')}]` : ''}</div>
          {t.outputVector && (
            <VectorBars values={t.outputVector} height={180} label={`${t.outputDim ?? t.outputVector.length}-dim feature vector`} />
          )}
          {t.outputFmaps && <FeatureMapsGrid data={t.outputFmaps} />}
          {t.outputHist && <Histogram data={t.outputHist} label="Feature distribution" />}
        </div>
      </div>
    </div>
  );
}
