// Default/fallback transformation — generic output visualization

import type { DefaultTransformation, DataTransformation, DropoutTransformation, LossTransformation } from '../types';
import { FeatureMapsGrid, VectorBars, Histogram, Arrow } from './shared';

export function DefaultViz({ t }: { t: DefaultTransformation }) {
  if (t.featureMaps) return <FeatureMapsGrid data={t.featureMaps} />;
  if (t.vector) return <VectorBars values={t.vector.values} />;
  if (t.scalar != null) return <div className="tfm-scalar">{t.scalar.toFixed(6)}</div>;
  return <div className="tfm-empty">No visualization</div>;
}

export function DataViz({ t }: { t: DataTransformation }) {
  return (
    <div className="tfm-data">
      {t.featureMaps && <FeatureMapsGrid data={t.featureMaps} label="Input Data" />}
      {t.vector && <VectorBars values={t.vector.values} />}

      {t.rawHist && t.normHist && (
        <div className="tfm-section">
          <div className="tfm-section-title">Normalization</div>
          <div className="tfm-flow">
            <div className="tfm-panel">
              <Histogram data={t.rawHist} color="#6c7086" label="Raw pixels (0 to 1)" />
            </div>
            <Arrow label="normalize" />
            <div className="tfm-panel">
              <Histogram data={t.normHist} color="#89b4fa" label="After normalization" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function DropoutViz({ t }: { t: DropoutTransformation }) {
  const hasHist = t.inputHist && t.outputHist;
  const hasCounts = t.totalElements != null;

  const beforeLabel = hasCounts
    ? `Before \u00b7 ${t.inputNonzero}/${t.totalElements} nonzero`
    : 'Before';
  const afterLabel = hasCounts
    ? `After \u00b7 ${t.outputNonzero}/${t.totalElements} nonzero`
    : 'After';

  return (
    <div className="tfm-dropout">
      {(t.inputMaps || t.outputMaps) && (
        <div className="tfm-before-after">
          <div className="tfm-ba-pane">
            <div className="tfm-ba-label">Before</div>
            {t.inputMaps ? <FeatureMapsGrid data={t.inputMaps} /> : <div className="tfm-empty">&mdash;</div>}
          </div>
          <div className="tfm-ba-divider" />
          <div className="tfm-ba-pane">
            <div className="tfm-ba-label">After</div>
            {t.outputMaps ? <FeatureMapsGrid data={t.outputMaps} /> : <div className="tfm-empty">&mdash;</div>}
          </div>
        </div>
      )}

      {hasHist && (
        <div className="tfm-section">
          <div className="tfm-section-title">Value Distribution</div>
          <div className="tfm-before-after">
            <div className="tfm-ba-pane">
              <Histogram data={t.inputHist!} color="#6c7086" label={beforeLabel} />
            </div>
            <div className="tfm-ba-divider" />
            <div className="tfm-ba-pane">
              <Histogram data={t.outputHist!} color="#89b4fa" label={afterLabel} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function LossViz({ t }: { t: LossTransformation }) {
  return <div className="tfm-scalar">{t.value.toFixed(6)}</div>;
}
