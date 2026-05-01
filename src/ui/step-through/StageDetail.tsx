// StageDetail — dispatches to per-layer transformation components.

import type { Stage } from './types';
import { formatShape } from './insights';
import { Conv2dViz } from './transformations/Conv2dViz';
import { LinearViz } from './transformations/LinearViz';
import { ActivationViz } from './transformations/ActivationViz';
import { SoftmaxViz } from './transformations/SoftmaxViz';
import { NormViz } from './transformations/NormViz';
import { PoolViz } from './transformations/PoolViz';
import { FlattenViz } from './transformations/FlattenViz';
import { UpsampleViz } from './transformations/UpsampleViz';
import { CrossEntropyViz } from './transformations/CrossEntropyViz';
import { MseLossViz } from './transformations/MseLossViz';
import { NoiseSchedulerViz } from './transformations/NoiseSchedulerViz';
import { ConcatViz } from './transformations/ConcatViz';
import { AddViz } from './transformations/AddViz';
import { PretrainedViz } from './transformations/PretrainedViz';
import { ReparameterizeViz } from './transformations/ReparameterizeViz';
import { ReshapeViz } from './transformations/ReshapeViz';
import { GanLossViz } from './transformations/GanLossViz';
import { VaeLossViz } from './transformations/VaeLossViz';
import { DefaultViz, DataViz, DropoutViz, LossViz } from './transformations/DefaultViz';

export function StageDetail({ stage }: { stage: Stage }) {
  const { transformation, insight } = stage;

  return (
    <div className="stage-detail">
      <div className="stage-detail-header">
        <span className="stage-detail-name">
          {stage.blockName && <span className="stage-detail-block">{stage.blockName}</span>}
          {stage.displayName}
        </span>
        <span className="stage-detail-shape">
          {formatShape(stage.inputShape)} &rarr; {formatShape(stage.outputShape)}
        </span>
      </div>

      {insight && <div className="stage-detail-insight">{insight}</div>}

      <div className="stage-detail-body">
        {transformation ? (
          <TransformationDispatcher transformation={transformation} />
        ) : (
          <div className="tfm-empty">No visualization available</div>
        )}
      </div>
    </div>
  );
}

function TransformationDispatcher({ transformation }: { transformation: NonNullable<Stage['transformation']> }) {
  switch (transformation.type) {
    case 'conv2d': return <Conv2dViz t={transformation} />;
    case 'linear': return <LinearViz t={transformation} />;
    case 'activation': return <ActivationViz t={transformation} />;
    case 'softmax': return <SoftmaxViz t={transformation} />;
    case 'norm': return <NormViz t={transformation} />;
    case 'pool': return <PoolViz t={transformation} />;
    case 'flatten': return <FlattenViz t={transformation} />;
    case 'upsample': return <UpsampleViz t={transformation} />;
    case 'dropout': return <DropoutViz t={transformation} />;
    case 'cross_entropy': return <CrossEntropyViz t={transformation} />;
    case 'noise_scheduler': return <NoiseSchedulerViz t={transformation} />;
    case 'mse_loss': return <MseLossViz t={transformation} />;
    case 'concat': return <ConcatViz t={transformation} />;
    case 'add': return <AddViz t={transformation} />;
    case 'pretrained': return <PretrainedViz t={transformation} />;
    case 'reparameterize': return <ReparameterizeViz t={transformation} />;
    case 'reshape': return <ReshapeViz t={transformation} />;
    case 'gan_loss': return <GanLossViz t={transformation} />;
    case 'vae_loss': return <VaeLossViz t={transformation} />;
    case 'data': return <DataViz t={transformation} />;
    case 'loss': return <LossViz t={transformation} />;
    case 'default': return <DefaultViz t={transformation} />;
    default: return <div className="tfm-empty">Unknown transformation type</div>;
  }
}
