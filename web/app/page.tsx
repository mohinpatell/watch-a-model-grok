import DataLoader from "@/components/DataLoader";
import Scrubber from "@/components/Scrubber";
import EmbeddingScatter from "@/components/EmbeddingScatter";
import FftSpectrum from "@/components/FftSpectrum";
import FftSpectrogram from "@/components/FftSpectrogram";
import AttentionRouting from "@/components/AttentionRouting";
import LossChart from "@/components/LossChart";
import ProgressMeasure from "@/components/ProgressMeasure";
import AblationTable from "@/components/AblationTable";
import AblationCurves from "@/components/AblationCurves";
import Tex from "@/components/Tex";
import Code from "@/components/Code";

// LaTeX literals. Every expression lives here so backslashes aren't
// silently eaten by JS string escaping (\b, \t, \r, ...).
const TEX = {
  task: String.raw`(a + b) \bmod 113`,
  pairsCount: String.raw`113^2 = 12{,}769`,
  tokens: String.raw`[a,\, b,\, =]`,
  normalN01: String.raw`\mathcal{N}(0, 1)`,
  sqrtDApprox: String.raw`\sqrt{d} \approx 11\times`,
  oneOverSqrtD: String.raw`1/\sqrt{d}`,
  tenNegThree: String.raw`10^{-3}`,
  minusLambdaW: String.raw`-\lambda W`,
  kStar: String.raw`k^* = 46`,
  kEq46: String.raw`k = 46`,
  uniformBaseline: String.raw`1/\lfloor p/2 \rfloor \approx 0.018`,
  tokenDomain: String.raw`t \in \{0,\dots,p-1\}`,
  embedCos: String.raw`(\cos(2\pi k^* t / p),\, \sin(2\pi k^* t / p))`,
  trigIdentity: String.raw`\cos(k\,a)\cos(k\,b) - \sin(k\,a)\sin(k\,b) = \cos\bigl(k(a + b)\bigr).`,
  cosKSum: String.raw`\cos(k(a+b))`,
  cVar: String.raw`c`,
  cosSumMinusC: String.raw`\cos(k(a+b-c))`,
  congruence: String.raw`c \equiv a+b \pmod p`,
  aVar: String.raw`a`,
  bVar: String.raw`b`,
  eqToken: String.raw`=`,
  fiveFreqs: String.raw`k \in \{13, 14, 28, 31, 46\}`,
  fourFreqs: String.raw`k \in \{13, 14, 31, 46\}`,
  kEq28: String.raw`k = 28`,
  fortySix: String.raw`46`,
  cosSinPair: String.raw`(\cos,\sin)`,
  kStarEq46Short: String.raw`k^*=46`,
  embedCosShort: String.raw`(\cos(2\pi k^* t/p),\, \sin(2\pi k^* t/p))`,
};

export default function Home() {
  return (
    <main className="min-h-screen bg-[var(--background)] text-[var(--foreground)]">
      <article className="mx-auto max-w-[680px] px-6 pt-20 pb-24">
        <header className="mb-10">
          <h1>Watch a model grok</h1>
          <p className="mt-3 text-[var(--muted)] font-mono text-xs tracking-tight">
            Mohin Patel · April 2026
          </p>
        </header>

        <p>
          A 1-layer transformer learning <Tex expr={TEX.task} /> does
          something strange: it memorizes the training set in about 140
          gradient steps, sits on a flat plateau for nearly 8,000 more steps,
          and then <em>generalizes</em>. Test accuracy climbs from
          near-zero to 99% inside a couple thousand steps, with the final
          lift from 50% to 99% taking under a thousand. Nanda et al. (2023)
          showed
          that the model is quietly building a Fourier-arithmetic circuit the
          whole time, and you just can&rsquo;t see it from the loss curve.
        </p>

        <p>
          Here is the same 40,000-step run, with scalars recorded at 234
          log-spaced checkpoints:
        </p>

        <DataLoader />

        <figure className="my-6">
          <LossChart mode="loss" width={660} height={260} />
          <figcaption className="mt-2">
            Train loss (red) collapses in the first ~140 steps. Test loss
            (blue) climbs through memorization, peaks around step 1,500,
            drifts down through the plateau, then falls off a cliff between
            steps 6,000 and 8,500. Both axes are log-scaled; memorization
            and grokking each take multiple orders of magnitude of steps to
            resolve.
          </figcaption>
        </figure>

        <h2>The task and the model</h2>

        <p>
          The task is to predict the sum of two numbers modulo 113, taught on
          30% of the <Tex expr={TEX.pairsCount} /> possible pairs. The split
          is fixed at initialization. Each example is three tokens{" "}
          <Tex expr={TEX.tokens} />, and we train a cross-entropy classifier
          over the full 114-way vocabulary at the final position.
        </p>

        <p>
          The architecture follows Nanda&rsquo;s paper verbatim: one
          transformer block, four attention heads, width 128, no LayerNorm,
          no biases.
        </p>

        <Code
          code={`class GrokTransformer(nn.Module):
    # No LayerNorm. No biases. Four attention heads, one block.
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.token_embed = nn.Embedding(cfg.vocab_size, d)
        self.pos_embed = nn.Embedding(cfg.seq_len, d)
        self.block = Block(cfg)
        self.unembed = nn.Linear(d, cfg.vocab_size, bias=False)
        # This init matters. See below.
        nn.init.normal_(self.token_embed.weight, std=1 / math.sqrt(d))
        nn.init.normal_(self.pos_embed.weight,   std=1 / math.sqrt(d))`}
        />

        <p>
          Training is AdamW with full-batch gradient descent (batch = training
          set), learning rate 1e-3, and the one hyperparameter that matters
          most: <code>weight_decay=1.0</code>. Nothing else is unusual: no
          warm-up, no schedule, no regularization tricks.
        </p>

        <Code
          code={`opt = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98),
)
for step in range(40_000):
    loss = F.cross_entropy(model(train_x)[:, -1, :], train_y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()`}
        />

        <aside className="my-6 border-l-2 border-[var(--accent)] pl-4 text-[0.95rem]">
          <strong>Init scale matters.</strong> PyTorch&rsquo;s default{" "}
          <code>nn.Embedding</code> initializes weights as{" "}
          <Tex expr={TEX.normalN01} />, roughly{" "}
          <Tex expr={TEX.sqrtDApprox} /> the <Tex expr={TEX.oneOverSqrtD} />{" "}
          scale the rest of the network uses. Leaving the embeddings at
          default puts them at radius ≈ 11 at step 0, which weight decay has
          to slowly drag down before its pressure can shape the geometry. I
          burned a couple of days before catching this. The{" "}
          <em>init_default</em> row in the ablation table stalls at 2% test
          accuracy after 15k steps, comfortably the worst failure mode in
          the sweep.
        </aside>

        <h2>Memorization is fast</h2>

        <p>
          By step 140, train accuracy is 1.0, and the model has compressed
          the 3,830 training examples into a lookup table in weight space. Train
          loss crosses <Tex expr={TEX.tenNegThree} /> a few hundred steps
          later and keeps falling. Test loss is <em>higher</em> than it was
          at init (because confidently-wrong predictions dominate), and test
          accuracy is glued to chance. If you stopped here and called it
          overfitting, you would look entirely reasonable.
        </p>

        <h2>The plateau is not empty</h2>

        <p>
          From step ~140 to step ~6,000, train loss sits near zero and test
          loss drifts. Nothing on the curves announces the transition to
          come. Weight decay, though, is quietly dragging every parameter
          toward zero. With the train loss already at zero, the gradient from
          the data is small; the weight-decay term{" "}
          <Tex expr={TEX.minusLambdaW} /> is the dominant force. So the model
          wanders through weight configurations that still solve training, at
          steadily decreasing norm. Most of those configurations still look
          like lookup tables. A small number of them are arithmetic circuits.
        </p>

        <p>
          To see that a circuit is forming before the loss curve moves, we
          can measure the Fourier power of the token embedding matrix. If the
          final model uses a dominant frequency <Tex expr={TEX.kStar} /> to
          encode the numbers on a circle, then at any step we can ask:{" "}
          <em>
            what fraction of the embedding&rsquo;s energy sits on{" "}
            <Tex expr={TEX.kEq46} /> specifically?
          </em>
        </p>

        <figure className="my-6">
          <ProgressMeasure />
          <figcaption className="mt-2">
            Fraction of token-embedding L2 power concentrated at frequency{" "}
            <Tex expr={TEX.kEq46} />, per checkpoint. The dashed line is the
            uniform baseline <Tex expr={TEX.uniformBaseline} />. The share
            begins climbing during the plateau, thousands of steps before
            test loss moves.
          </figcaption>
        </figure>

        <p>
          <Tex expr={TEX.kEq46} /> is not special by itself. The broader
          spectrum tells a sharper story. Showing the share at <em>every</em>{" "}
          frequency, across training, makes the circuit&rsquo;s emergence
          visible as a handful of horizontal bands lighting up against
          uniform noise:
        </p>

        <figure className="my-6">
          <FftSpectrogram />
          <figcaption className="mt-2">
            Embedding power share at each frequency (y) over training steps
            (x). The orange dashed line marks <Tex expr={TEX.kEq46} />. For
            the first few thousand steps every row is the same dim blue:
            the spectrum is flat. Around step 3,000 a few frequencies
            brighten, and by the time test accuracy snaps, four rows
            (<Tex expr={TEX.fourFreqs} />) have pulled away. The fifth
            component, <Tex expr={TEX.kEq28} />, only emerges later as the
            circuit consolidates. Everything in between the bright bands
            stays at noise.
          </figcaption>
        </figure>

        <h2>The circuit</h2>

        <p>
          The final model embeds each token <Tex expr={TEX.tokenDomain} />{" "}
          near <Tex expr={TEX.embedCos} /> in a specific two-plane of the
          128-dimensional embedding space. The MLP combines the two operand
          embeddings via the angle-sum identity:
        </p>

        <Tex display expr={TEX.trigIdentity} />

        <p>
          giving <Tex expr={TEX.cosKSum} /> internally. The unembed then
          subtracts each candidate <Tex expr={TEX.cVar} />, so per-frequency
          terms at the output take the form <Tex expr={TEX.cosSumMinusC} />;
          summed across the active frequencies, these peak sharply when{" "}
          <Tex expr={TEX.congruence} />. Attention routes{" "}
          <Tex expr={TEX.aVar} /> and <Tex expr={TEX.bVar} /> into the
          position of <Tex expr={TEX.eqToken} />; the MLP and unembed then
          build the frequency-matched sum. Five frequencies (
          <Tex expr={TEX.fiveFreqs} /> in this seed, with{" "}
          <Tex expr={TEX.fortySix} /> leading by a narrow margin) participate.
        </p>

        <figure className="my-6">
          <AttentionRouting />
          <figcaption className="mt-2">
            The <Tex expr={TEX.eqToken} /> token&rsquo;s attention weights,
            averaged over five probe pairs, broken out by head. At step 0
            every head splits attention uniformly over three positions
            (~33% each). Within ~100 gradient steps the{" "}
            <Tex expr={TEX.eqToken} /> column collapses to zero and each
            head latches onto a strong <Tex expr={TEX.aVar} />-
            or <Tex expr={TEX.bVar} />-preference; one head overshoots all
            the way to roughly 1/99 by the start of the plateau. Through the plateau those extreme
            preferences soften, and by the time the model groks, three heads
            sit within ~15% of a balanced{" "}
            <Tex expr={TEX.aVar} />/<Tex expr={TEX.bVar} /> split while the
            fourth is still drifting down from its overshoot. Generalization
            apparently needs
            the two operands treated symmetrically, but memorization does
            not. Scrub the slider below to watch the overshoot and relaxation.
          </figcaption>
        </figure>

        <h2>The snap</h2>

        <p>
          Pull the scrubber below through training. You&rsquo;re looking at
          the token embeddings projected onto the 2D{" "}
          <Tex expr={TEX.cosSinPair} /> basis for{" "}
          <Tex expr={TEX.kStarEq46Short} /> of the <em>final</em> embedding,
          with each frame aligned to the previous via Procrustes so the ring
          glides rather than flips.
        </p>

        <figure className="my-6">
          <div className="rounded-lg border border-[var(--rule)] bg-[var(--surface)] p-4 flex flex-col gap-4">
            <Scrubber />
            <EmbeddingScatter />
            <FftSpectrum />
          </div>
          <figcaption className="mt-2">
            At step 0, tokens are a random cloud. During memorization, they
            cluster but don&rsquo;t organize. Across the plateau they begin
            to string out. Between step ~6,000 and step ~8,500 they snap into
            a ring, colored by token index. Modular addition becomes literal
            rotation on the circle. The bar chart below the ring shows the
            same frame&rsquo;s embedding power spectrum: early on it looks
            like noise, but during grokking energy concentrates sharply on a
            handful of independent Fourier modes: <Tex expr={TEX.kEq46} />{" "}
            (highlighted in orange) alongside a few others that together
            define the ring.
          </figcaption>
        </figure>

        <h2>What didn&rsquo;t work</h2>

        <p>
          I ran six ablations. Each changes one hyperparameter of the
          baseline and trains for 15,000 steps (well past when the baseline
          groks). Results:
        </p>

        <AblationTable />

        <p>
          The test-loss curves tell the same story. Without the full{" "}
          <code>weight_decay=1.0</code>, the model overfits and test loss
          keeps climbing. <code>wd=0.01</code> is close enough to zero
          here to produce the same failure. With PyTorch&rsquo;s default{" "}
          <Tex expr={TEX.normalN01} /> embedding init, the model memorizes
          and test accuracy hovers below 3%, the strongest failure mode in
          this sweep, and the one that confirms the init-scale note above.
          The two alternative seeds grok on roughly the same schedule as the
          baseline: the phenomenon is not seed-fragile.
        </p>

        <p>
          The <em>layer_norm</em> row is the most surprising: adding pre-norm
          LayerNorm doesn&rsquo;t block grokking; it <em>accelerates</em>{" "}
          it, to around step 3,000 versus 8,000 in the baseline. I ran the
          same LayerNorm configuration with two additional seeds to check
          this wasn&rsquo;t a lucky draw; they grokked at step 3,481 and
          5,016, both well inside the plateau the baseline is still sitting
          on. Nanda&rsquo;s paper removes LayerNorm for analytical cleanness
          (a Fourier decomposition is cleaner without the normalization), not
          because the circuit refuses to form with it. That&rsquo;s worth
          flagging because it&rsquo;s easy to read the paper and conclude
          LayerNorm is part of the problem. It isn&rsquo;t, at least not
          on this task.
        </p>

        <figure className="my-6">
          <AblationCurves />
          <figcaption className="mt-2">
            Test loss per ablation. A grokked run shows a characteristic
            overshoot-then-collapse (see <em>seed_0</em>, <em>seed_7</em>);
            a non-grokked run rises and stays up.
          </figcaption>
        </figure>

        <h2>Notes &amp; references</h2>

        <ul className="list-disc pl-5 space-y-1.5 text-[0.98rem]">
          <li>
            Nanda, Chan, Lieberum, Smith, Steinhardt (2023).{" "}
            <a
              href="https://arxiv.org/abs/2301.05217"
              target="_blank"
              rel="noreferrer"
            >
              <em>
                Progress Measures for Grokking via Mechanistic
                Interpretability.
              </em>
            </a>{" "}
            The paper that identified the Fourier circuit and introduced
            progress measures that track it.
          </li>
          <li>
            Power, Burda, Edwards, Babuschkin, Misra (2022).{" "}
            <a
              href="https://arxiv.org/abs/2201.02177"
              target="_blank"
              rel="noreferrer"
            >
              <em>
                Grokking: Generalization Beyond Overfitting on Small
                Algorithmic Datasets.
              </em>
            </a>{" "}
            The original grokking paper.
          </li>
          <li>
            All training code, the 2D-aligned embedding projection, and the
            ablation sweep live in{" "}
            <a
              href="https://github.com/mohinpatell/watch-a-model-grok"
              target="_blank"
              rel="noreferrer"
            >
              the repo
            </a>
            . The embedding projection is onto the 2D{" "}
            <Tex expr={TEX.embedCosShort} /> basis of the final embedding,
            frame-aligned by orthogonal Procrustes.
          </li>
        </ul>
      </article>
    </main>
  );
}
