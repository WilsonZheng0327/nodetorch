// VAE Loss: original vs reconstruction + loss breakdown

import type { VaeLossTransformation } from '../types';
import { FeatureMapsGrid } from './shared';

export function VaeLossViz({ t }: { t: VaeLossTransformation }) {
  const beta = t.beta ?? 1.0;
  const computed = (t.reconLoss != null && t.klLoss != null) ? t.reconLoss + beta * t.klLoss : null;

  return (
    <div className="tfm-vae-loss">
      {/* Original vs Reconstruction */}
      {(t.originalFmaps || t.reconFmaps) && (
        <div className="tfm-before-after">
          <div className="tfm-ba-pane">
            <div className="tfm-ba-label">Original</div>
            {t.originalFmaps && <FeatureMapsGrid data={t.originalFmaps} />}
          </div>
          <div className="tfm-ba-divider" />
          <div className="tfm-ba-pane">
            <div className="tfm-ba-label">Reconstruction</div>
            {t.reconFmaps && <FeatureMapsGrid data={t.reconFmaps} />}
          </div>
        </div>
      )}

      {/* Loss breakdown */}
      <div className="tfm-section">
        <div className="tfm-section-title">Loss = MSE(recon, original) + {beta !== 1 ? `${beta} \u00d7 ` : ''}KL(mean, logvar)</div>
        <div className="tfm-vae-loss-breakdown">
          {t.reconLoss != null && (
            <div className="tfm-vae-loss-item">
              <span className="tfm-vae-loss-name">Reconstruction (MSE, mean)</span>
              <span className="tfm-vae-loss-value">{t.reconLoss.toFixed(4)}</span>
            </div>
          )}
          {t.klLoss != null && (
            <div className="tfm-vae-loss-item">
              <span className="tfm-vae-loss-name">KL Divergence{beta !== 1 ? ` (\u00d7${beta})` : ''}</span>
              <span className="tfm-vae-loss-value">{(beta * t.klLoss).toFixed(4)}</span>
              {beta !== 1 && <span className="tfm-vae-loss-raw">(raw: {t.klLoss.toFixed(4)})</span>}
            </div>
          )}
          {t.totalLoss != null && (
            <div className="tfm-vae-loss-item tfm-vae-loss-total">
              <span className="tfm-vae-loss-name">Total</span>
              <span className="tfm-vae-loss-value">{t.totalLoss.toFixed(4)}</span>
            </div>
          )}
          {computed != null && t.totalLoss != null && Math.abs(computed - t.totalLoss) > 0.01 && (
            <div className="tfm-note">
              Note: computed {computed.toFixed(4)} vs actual {t.totalLoss.toFixed(4)} — small difference from numerical precision
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
