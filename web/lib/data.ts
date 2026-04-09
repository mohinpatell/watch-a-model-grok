export type Meta = {
  config: {
    p: number;
    frac_train: number;
    n_layers: number;
    n_heads: number;
    d_model: number;
    d_head: number;
    d_mlp: number;
    lr: number;
    weight_decay: number;
    betas: [number, number];
    num_steps: number;
    batch_size: number;
    seed: number;
    num_checkpoints: number;
  };
  n_frames: number;
  p: number;
  seq_len: number;
  n_heads: number;
  fourier_k: number;
  shapes: {
    embeds: [number, number, number];
    attn: [number, number, number, number, number];
    fft_share: [number, number];
  };
  steps: number[];
  train_loss: number[];
  test_loss: number[];
  train_acc: number[];
  test_acc: number[];
  share_at_kstar: number[];
};

export type Dataset = {
  meta: Meta;
  embeds: Float32Array;
  attn: Float32Array;
  fftShare: Float32Array;
};

async function fetchFloat32(url: string): Promise<Float32Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`);
  const buf = await res.arrayBuffer();
  return new Float32Array(buf);
}

export async function loadDataset(basePath = "/data"): Promise<Dataset> {
  const metaRes = await fetch(`${basePath}/meta.json`);
  if (!metaRes.ok) throw new Error(`failed to load meta.json: ${metaRes.status}`);
  const meta: Meta = await metaRes.json();

  const [embeds, attn, fftShare] = await Promise.all([
    fetchFloat32(`${basePath}/embeds.bin`),
    fetchFloat32(`${basePath}/attn.bin`),
    fetchFloat32(`${basePath}/fft_share.bin`),
  ]);

  const expectedEmbeds = meta.shapes.embeds.reduce((a, b) => a * b, 1);
  const expectedAttn = meta.shapes.attn.reduce((a, b) => a * b, 1);
  const expectedFft = meta.shapes.fft_share.reduce((a, b) => a * b, 1);
  if (embeds.length !== expectedEmbeds)
    throw new Error(`embeds length ${embeds.length} != ${expectedEmbeds}`);
  if (attn.length !== expectedAttn)
    throw new Error(`attn length ${attn.length} != ${expectedAttn}`);
  if (fftShare.length !== expectedFft)
    throw new Error(`fftShare length ${fftShare.length} != ${expectedFft}`);

  return { meta, embeds, attn, fftShare };
}

export function embedsFrame(ds: Dataset, frame: number): Float32Array {
  const p = ds.meta.p;
  const stride = p * 2;
  return ds.embeds.subarray(frame * stride, (frame + 1) * stride);
}

export function attnFrame(ds: Dataset, frame: number): Float32Array {
  const [, b, h, s] = ds.meta.shapes.attn;
  const stride = b * h * s * s;
  return ds.attn.subarray(frame * stride, (frame + 1) * stride);
}

export function fftShareFrame(ds: Dataset, frame: number): Float32Array {
  const [, k] = ds.meta.shapes.fft_share;
  return ds.fftShare.subarray(frame * k, (frame + 1) * k);
}
