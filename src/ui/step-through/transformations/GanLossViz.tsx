// GAN Loss: how well does the discriminator distinguish real from fake?

import type { GanLossTransformation } from '../types';

export function GanLossViz({ t }: { t: GanLossTransformation }) {
  const hasScores = t.realScore != null || t.fakeScore != null;

  return (
    <div className="tfm-gan-loss">
      {/* Explanation */}
      <div className="tfm-gan-explain">
        The discriminator looks at an image and outputs a score.
        High score = "I think this is real." Low score = "I think this is fake."
      </div>

      {/* Score comparison */}
      {hasScores && (
        <div className="tfm-gan-scores">
          <ScoreCard
            title="Real Image"
            subtitle="Fed a real training image"
            prob={t.realProb}
            verdict={t.realProb != null ? (t.realProb > 0.5 ? 'Correctly identified as real' : 'Fooled — thinks real is fake') : undefined}
            good={t.realProb != null ? t.realProb > 0.5 : undefined}
            color="#10b981"
          />
          <ScoreCard
            title="Fake Image"
            subtitle="Fed a generator output"
            prob={t.fakeProb}
            verdict={t.fakeProb != null ? (t.fakeProb < 0.5 ? 'Correctly identified as fake' : 'Fooled — thinks fake is real') : undefined}
            good={t.fakeProb != null ? t.fakeProb < 0.5 : undefined}
            color="#f38ba8"
          />
        </div>
      )}

      {/* Loss breakdown */}
      <div className="tfm-section">
        <div className="tfm-section-title">Loss Breakdown</div>
        <div className="tfm-vae-loss-breakdown">
          {t.dLossReal != null && (
            <div className="tfm-vae-loss-item">
              <div>
                <div className="tfm-vae-loss-name">D loss on real</div>
                <div className="tfm-gan-loss-hint">How wrong D is about the real image (lower = D is correct)</div>
              </div>
              <span className="tfm-vae-loss-value">{t.dLossReal.toFixed(4)}</span>
            </div>
          )}
          {t.dLossFake != null && (
            <div className="tfm-vae-loss-item">
              <div>
                <div className="tfm-vae-loss-name">D loss on fake</div>
                <div className="tfm-gan-loss-hint">How wrong D is about the fake image (lower = D is correct)</div>
              </div>
              <span className="tfm-vae-loss-value">{t.dLossFake.toFixed(4)}</span>
            </div>
          )}
          {t.totalLoss != null && (
            <div className="tfm-vae-loss-item tfm-vae-loss-total">
              <span className="tfm-vae-loss-name">Total D Loss</span>
              <span className="tfm-vae-loss-value">{t.totalLoss.toFixed(4)}</span>
            </div>
          )}
        </div>
      </div>

      {/* Interpretation */}
      {t.realProb != null && t.fakeProb != null && (
        <div className="tfm-gan-interpret">
          <Interpretation realProb={t.realProb} fakeProb={t.fakeProb} />
        </div>
      )}
    </div>
  );
}

function ScoreCard({ title, subtitle, prob, verdict, good, color }: {
  title: string; subtitle: string; prob?: number; verdict?: string; good?: boolean; color: string;
}) {
  return (
    <div className="tfm-gan-score-card">
      <div className="tfm-gan-score-label">{title}</div>
      <div className="tfm-gan-score-subtitle">{subtitle}</div>
      {prob != null && (
        <>
          <div className="tfm-gan-score-bar">
            <div className="tfm-gan-score-bar-fill" style={{ width: `${prob * 100}%`, background: color }} />
          </div>
          <div className="tfm-gan-score-row">
            <span className="tfm-gan-score-prob-label">D says: {(prob * 100).toFixed(0)}% real</span>
          </div>
          {verdict && (
            <div className={`tfm-gan-verdict ${good ? 'tfm-gan-verdict-good' : 'tfm-gan-verdict-bad'}`}>
              {verdict}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Interpretation({ realProb, fakeProb }: { realProb: number; fakeProb: number }) {
  const dCorrectReal = realProb > 0.5;
  const dCorrectFake = fakeProb < 0.5;

  if (dCorrectReal && dCorrectFake && realProb > 0.8 && fakeProb < 0.2) {
    return <div className="tfm-gan-interpret-msg">
      <strong>Discriminator dominates.</strong> It easily tells real from fake. The generator's images are not convincing yet — early in training, this is normal.
    </div>;
  }
  if (dCorrectReal && dCorrectFake) {
    return <div className="tfm-gan-interpret-msg">
      <strong>Discriminator is ahead.</strong> It can still distinguish real from fake, but the gap is narrowing. Training is progressing.
    </div>;
  }
  if (!dCorrectFake && fakeProb > 0.7) {
    return <div className="tfm-gan-interpret-msg tfm-gan-interpret-warn">
      <strong>Generator is fooling the discriminator.</strong> D thinks the fake image is {(fakeProb * 100).toFixed(0)}% likely to be real. If images look good, training is working. If images look bad, this could be mode collapse.
    </div>;
  }
  return <div className="tfm-gan-interpret-msg tfm-gan-interpret-good">
    <strong>Balanced competition.</strong> Neither side dominates — this is the ideal GAN training dynamic.
  </div>;
}
