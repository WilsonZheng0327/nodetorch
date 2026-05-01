// Noise Scheduler: clean image + noise = noisy image, plus timestep channel explanation

import type { NoiseSchedulerTransformation } from '../types';
import { FeatureMapsGrid } from './shared';

export function NoiseSchedulerViz({ t }: { t: NoiseSchedulerTransformation }) {
  return (
    <div className="tfm-noise-sched">
      {/* Timestep info bar */}
      {t.timestep != null && (
        <div className="tfm-noise-sched-info">
          <div className="tfm-noise-sched-timestep">
            Timestep {t.timestep} / {t.numTimesteps ?? '?'}
          </div>
          {t.signalRatio != null && t.noiseRatio != null && (
            <div className="tfm-noise-sched-ratio">
              <div className="tfm-noise-sched-bar">
                <div className="tfm-noise-sched-signal" style={{ width: `${t.signalRatio * 100}%` }} />
                <div className="tfm-noise-sched-noise" style={{ width: `${t.noiseRatio * 100}%` }} />
              </div>
              <div className="tfm-noise-sched-labels">
                <span className="tfm-noise-sched-signal-label">{(t.signalRatio * 100).toFixed(0)}% signal</span>
                <span className="tfm-noise-sched-noise-label">{(t.noiseRatio * 100).toFixed(0)}% noise</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Clean + Noise = Noisy */}
      <div className="tfm-noise-sched-flow">
        <div className="tfm-noise-sched-col">
          <div className="tfm-noise-sched-col-label">Clean Image</div>
          {t.cleanShape && <div className="tfm-noise-sched-shape">{fmtShape(t.cleanShape)}</div>}
          {t.cleanFmaps ? <FeatureMapsGrid data={t.cleanFmaps} /> : <div className="tfm-empty">&mdash;</div>}
        </div>
        <div className="tfm-noise-sched-op">+</div>
        <div className="tfm-noise-sched-col">
          <div className="tfm-noise-sched-col-label">Random Noise</div>
          {t.noiseFmaps ? <FeatureMapsGrid data={t.noiseFmaps} /> : <div className="tfm-empty">&mdash;</div>}
        </div>
        <div className="tfm-noise-sched-op">=</div>
        <div className="tfm-noise-sched-col">
          <div className="tfm-noise-sched-col-label">Noisy Image</div>
          {t.noisyShape && <div className="tfm-noise-sched-shape">{fmtShape(t.noisyShape)}</div>}
          {t.noisyFmaps ? <FeatureMapsGrid data={t.noisyFmaps} /> : <div className="tfm-empty">&mdash;</div>}
        </div>
      </div>

      {/* 3 outputs explanation */}
      <div className="tfm-section">
        <div className="tfm-section-title">3 Outputs</div>
        <div className="tfm-noise-sched-outputs">
          <div className="tfm-noise-sched-output-row">
            <span className="tfm-noise-sched-port">out</span>
            <span className="tfm-noise-sched-out-desc">Noisy image — fed into the denoising model</span>
            {t.noisyShape && <span className="tfm-noise-sched-out-shape">{fmtShape(t.noisyShape)}</span>}
          </div>
          <div className="tfm-noise-sched-output-row">
            <span className="tfm-noise-sched-port">noise</span>
            <span className="tfm-noise-sched-out-desc">The actual noise added — this is the training target (model learns to predict this)</span>
            {t.noisyShape && <span className="tfm-noise-sched-out-shape">{fmtShape(t.noisyShape)}</span>}
          </div>
          <div className="tfm-noise-sched-output-row">
            <span className="tfm-noise-sched-port">timestep</span>
            <span className="tfm-noise-sched-out-desc">
              Every pixel = <strong>{t.tNormalized?.toFixed(2) ?? '?'}</strong> (= {t.timestep}/{t.numTimesteps ?? '?'}).
              Not an image — just the scalar timestep broadcast to a spatial channel so it can be concatenated with the image for the conv layers.
            </span>
            {t.timestepShape && <span className="tfm-noise-sched-out-shape">{fmtShape(t.timestepShape)}</span>}
          </div>
        </div>
      </div>

      {/* Concat explanation */}
      {t.concatResult && (
        <div className="tfm-section">
          <div className="tfm-section-title">How timestep channel is used</div>
          <div className="tfm-noise-sched-concat-explain">
            <div>The noisy image and timestep channel are <strong>concatenated</strong> along the channel dimension before being fed into the model:</div>
            <div className="tfm-noise-sched-concat-calc">
              {t.concatExplain}  &rarr;  <strong>{t.concatResult}</strong>
            </div>
            <div className="tfm-noise-sched-concat-why">
              This way the model knows both <em>what</em> the noisy image looks like and <em>how noisy</em> it is, so it can predict the appropriate amount of noise to remove.
            </div>
          </div>
        </div>
      )}

      {/* Formula */}
      <div className="tfm-noise-sched-formula">
        x<sub>t</sub> = &radic;<span style={{ textDecoration: 'overline' }}>&alpha;</span><sub>t</sub> &middot; x<sub>0</sub> + &radic;(1 - <span style={{ textDecoration: 'overline' }}>&alpha;</span><sub>t</sub>) &middot; &epsilon;
      </div>
    </div>
  );
}

function fmtShape(shape: number[]): string {
  return `[${shape.join(', ')}]`;
}
