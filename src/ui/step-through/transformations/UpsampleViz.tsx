// Upsample transformation: input feature maps → larger output feature maps

import type { UpsampleTransformation } from '../types';
import { FeatureMapsGrid, Arrow } from './shared';

export function UpsampleViz({ t }: { t: UpsampleTransformation }) {
  return (
    <div className="tfm-upsample">
      <div className="tfm-flow">
        <FeatureMapsGrid data={t.input} label="Input" />
        <Arrow label="Upsample" />
        <FeatureMapsGrid data={t.output} label="Output" />
      </div>
    </div>
  );
}
