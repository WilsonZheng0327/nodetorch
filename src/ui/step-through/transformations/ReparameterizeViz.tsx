// Reparameterize (VAE): mean + logvar → sampled z
// Shows: mean vector | logvar vector | → | z vector

import type { ReparameterizeTransformation } from '../types';
import { VectorBars, Histogram } from './shared';

export function ReparameterizeViz({ t }: { t: ReparameterizeTransformation }) {
  return (
    <div className="tfm-reparam">
      {/* Before: mean and logvar side by side */}
      <div className="tfm-before-after">
        <div className="tfm-ba-pane">
          <div className="tfm-ba-label">Mean ({t.meanValues?.length ?? 0} dims)</div>
          {t.meanValues && <VectorBars values={t.meanValues} height={160} />}
          {t.meanHist && <Histogram data={t.meanHist} color="#89b4fa" />}
        </div>
        <div className="tfm-ba-divider" />
        <div className="tfm-ba-pane">
          <div className="tfm-ba-label">Log Variance ({t.logvarValues?.length ?? 0} dims)</div>
          {t.logvarValues && <VectorBars values={t.logvarValues} height={160} />}
          {t.logvarHist && <Histogram data={t.logvarHist} color="#f9e2af" />}
        </div>
        <div className="tfm-ba-divider" />
        <div className="tfm-ba-pane">
          <div className="tfm-ba-label">Sampled z ({t.latentDim ?? '?'} dims)</div>
          {t.zValues && <VectorBars values={t.zValues} height={160} />}
        </div>
      </div>
    </div>
  );
}
