% Template for ICASSP-2026 paper; to be used with:
%          spconf.sty  - ICASSP/ICIP LaTeX style file, and
%          IEEEbib.bst - IEEE bibliography style file.
% --------------------------------------------------------------------------
\documentclass{article}
\usepackage{spconf,amsmath,amssymb,graphicx,hyperref,booktabs,multirow,array}

% Lets a full-width table* also sit at the BOTTOM of a page. In a plain twocolumn
% article a table* may only go to a page top, which is why the wide tables piled
% up and were flushed out as standalone float pages.
\usepackage{stfloats}

% Float placement. topfraction/textfraction are relaxed so a wide table may be
% placed on the page where it is declared; floatpagefraction is RAISED so that
% LaTeX prefers a page top/bottom over emitting a standalone float page.
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{1}
\setcounter{totalnumber}{3}
\setcounter{dbltopnumber}{2}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.5}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.95}
\renewcommand{\dbltopfraction}{0.9}
\renewcommand{\dblfloatpagefraction}{0.95}
\setlength{\belowcaptionskip}{4pt}

% Title.
% ------
\title{MULTI-DIMENSIONAL PROSODY JUDGMENT FOR \\
LIVESTREAM SPEECH SYNTHESIS}
%
% Two affiliations: Zifan Guan is jointly affiliated, the other two authors are
% Alibaba-only, hence the superscript markers rather than spconf's single-address
% form.
% ---------------
\name{Zifan Guan$^{1,2}$, \quad Longyu Lu$^{2}$, \quad Meiguang Jin$^{2}$
\thanks{Work done at the Alibaba TaoBao \& TianMao Group Livestream AIGC team, which
supported this research.}}
\address{$^{1}$The Chinese University of Hong Kong, Shenzhen, China \\
$^{2}$Alibaba TaoBao \& TianMao Group Livestream AIGC}
%
\begin{document}
%\ninept
%
\maketitle
%
\begin{abstract}
We build an automatic prosody evaluation model for e-commerce livestream speech
synthesis, with pairwise evaluation and Best-of-8 candidate selection as its
current applications. We compare against SpeechJudge-GRM and
Gemini-3.1-pro-preview and use Qwen3-Omni as the student backbone. We develop a
pipeline combining Gemini distillation, two-stage supervised fine-tuning, and GRPO
reinforcement learning. Across four evaluation sets and a unanimous subset of T1,
Live-SpeechJudge+GRPO with 10-sample balanced-order aggregation, which averages five
judgments in each A/B presentation order, has higher point accuracy than a single
Gemini judgment in all reported columns. On a
high-confidence subset of 136 out of 400 Best-of-8 candidate sets, its
tournament-selected candidate achieves Hit@1/2/3 rates of 72.06\%, 77.94\%, and
85.29\%, respectively. We further study verdict coupling, a pattern in which
per-dimension verdicts follow the overall verdict. To remove the shared
overall-verdict target, we query the teacher separately for each of the four core
dimensions C1--C4 and train Decouple-Live-SpeechJudge without an overall
conclusion. Its non-unanimous outputs show that the model can issue different
verdicts across dimensions. In one fixed-seed run, per-dimension GRPO changes
pooled point agreement from 73.64\% to 77.94\% with one sample and from 84.10\%
to 86.10\% with 10-sample balanced-order aggregation. At 10 samples, however,
paired-bootstrap intervals for the individual dimensions include zero, so these
changes are descriptive rather than statistically significant.
\end{abstract}
%
\begin{keywords}
prosody evaluation, expressive speech evaluation, LLM-as-a-judge, reinforcement
learning
\end{keywords}
%
\section{Introduction}
\label{sec:intro}

E-commerce livestream TTS must jointly satisfy fluency, varied intonation,
plausible emotion, high energy, emphasis on sales cues, and smooth switches between
product explanation and audience interaction. A single quality score cannot
represent these requirements. We therefore seek a pairwise evaluator that provides
fine-grained prosodic feedback for two applications: selecting the best of $N$
sampled utterances at serving time and supplying scalable rewards for TTS
preference optimization \cite{dpo,grpo}.

Reference-free MOS predictors \cite{mosnet,utmos} regress a single naturalness
score and do not represent the expressive and interaction-mode phenomena in our
rubric. SpeechJudge-GRM \cite{speechjudge} is not specialized for livestream speech
and exhibits a large B-position gap under our protocol, while the stronger
Gemini-3.1-pro-preview \cite{gemini} is too costly for our deployment setting. We
therefore distill it into the audio-capable Qwen3-Omni \cite{qwen3omni}.

A second problem is \textbf{verdict coupling}: per-dimension score differences from
the teacher, SpeechJudge-GRM, and our coupled student often follow the overall
winner. A multi-dimensional rubric can then collapse to one bit and fail to provide
varied rewards. We address this by removing the overall-verdict target and applying
dimension-specific supervision.

This paper makes three contributions.

\noindent\textbf{1. A livestream audio evaluation benchmark.} We introduce a
seven-dimension pairwise rubric and 1043 human-annotated pairs spanning 114 speakers,
cross-model, same-model, and human-vs.-TTS comparisons, three speaking styles, and
three duration bands.

\noindent\textbf{2. Live-SpeechJudge.} We combine 4-way swap-consistency filtering,
a two-stage curriculum, dimension-weighted SFT, and GRPO. Its 10-sample
balanced-order aggregation has higher point accuracy than one Gemini judgment on all
reported columns. The model also shows a small measured presentation-slot gap and obtains
72.06\%/77.94\%/85.29\% Hit@1/2/3 on the retained high-confidence Best-of-8 sets.

\noindent\textbf{3. Decouple-Live-SpeechJudge (D-LSJ).} D-LSJ queries supervision
separately for C1--C4, masks uncertain pair-dimensions, removes the shared overall
verdict, and applies each GRPO advantage only to its dimension's rationale span. It
produces non-unanimous verdicts on 24.3--60.6\% of evaluated pairs. In one fixed-seed
run, span-local GRPO changes pooled 10-sample human agreement from 84.10\% to
86.10\%; the per-dimension confidence intervals include zero.


\section{Related work}
\label{sec:related}

\subsection{Automatic speech quality assessment}
\label{ssec:rel_mos}

Reference-free MOS predictors \cite{mosnet,utmos,nisqa} regress a scalar --- or a
fixed set of transmission-quality scalars --- trained to match crowdsourced ratings
of largely read speech. They are cheap enough to run inside an RL loop, but they
emit no rationale, they score a clip in isolation rather than comparing two
renderings of the same text, and they are weakly correlated with human judgments of
expressiveness. The long-form benchmark SwanBench-Speech \cite{swanbench} quantifies
this directly: over 1101 samples spanning 17 scenarios, modern TTS already
approaches human level on computable acoustic metrics such as timbre consistency and
content accuracy, while prosodic coherence and expressive richness remain clearly
behind, and conventional MOS networks agree with human expressiveness ratings far
worse than a large audio-language judge does. That benchmark consequently has to
delegate exactly those subjective dimensions to SpeechJudge \cite{speechjudge} and
Gemini. A scalar naturalness regressor is therefore not a candidate for our task,
neither as a best-of-$N$ comparator nor as a per-dimension reward.

\subsection{Model-as-a-judge benchmarks for speech synthesis}
\label{ssec:rel_bench}

A parallel line uses a strong audio LLM as the evaluator and invests the effort in
the test material instead. EmergentTTS-Eval \cite{emergentttseval} grows 1645 test
cases from a small pool of human-written seed prompts by iteratively prompting an
LLM to make each case harder, organized into six challenge categories --- emotions,
paralinguistics, foreign words, syntactic complexity, complex pronunciation, and
questions --- and scores systems with a large audio-language model as judge,
reporting high correlation with human preference. SwanBench-Speech does the same at
minute-scale duration and 17 scenarios. These works establish that
model-as-a-judge is a viable evaluation protocol for expressive TTS, and they
motivate our rubric design, but they are consumers of an expensive closed judge
rather than builders of a cheap one, and they produce a benchmark score for a
system rather than a per-utterance reward vector that an RL loop can differentiate
against. Our test sets also differ in construction intent: rather than curating
adversarial text, each pair holds the transcript fixed, reducing lexical-content
confounding and focusing the comparison on differences in the rendered delivery.

\subsection{SpeechJudge and generative speech reward models}
\label{ssec:rel_speechjudge}

SpeechJudge \cite{speechjudge} is the closest system-level analogue to this work
and our principal baseline. It contributes three artifacts around a single
subjective axis, naturalness. SpeechJudge-Data is a human-feedback corpus of 99K
speech pairs generated by a diverse set of advanced zero-shot TTS systems across
several speech styles and languages, annotated for both intelligibility and
naturalness preference; from it the authors carve SpeechJudge-Eval, a benchmark on
which they show that existing metrics and audio LLMs are weak --- the strongest
off-the-shelf model, Gemini-2.5-Flash, reaches under 70\% agreement with human
judgment. SpeechJudge-GRM is then a generative reward model initialized from
Qwen2.5-Omni-7B and post-trained in two stages: supervised fine-tuning on
chain-of-thought rationales, followed by GRPO reinforcement learning restricted to
challenging cases. It attains 77.2\% on SpeechJudge-Eval (79.4\% with
inference-time scaling at $N=10$) against 72.7\% for a classic Bradley--Terry
reward model, and the authors demonstrate its use as a reward function when
post-training a speech generation model. Its overall pipeline --- a generative
judge trained first with rationale supervision and then with RL --- closely matches
ours; we also tested our pipeline on the SpeechJudge backbone before migrating to
Qwen3-Omni (Section \ref{ssec:setup}).

Three differences separate it from what a livestream judge needs. First,
\emph{granularity}: SpeechJudge-GRM is natively trained to emit one naturalness
preference and therefore does not directly supply the reward vector required by
per-dimension TTS optimization. Under our common multidimensional prompt it can
produce section-level assessments, which we use only to analyze whether those
assessments follow the overall preference. Second, \emph{domain and bias}: it is not specialized for e-commerce livestream
speech, has low human-label agreement on our high-expressiveness sets, and carries
a strong positional prior under our protocol,
preferring whichever clip is presented second by $+0.98$ on a 10-point scale with
a 66.8\% B win rate (Table \ref{tab:posbias}) --- a bias of the kind first
catalogued for text LLM judges \cite{mtbench}. Third, \emph{supervision economics}: its accuracy rests on 99K human preference
annotations, whereas our task-specific model uses roughly two thousand
teacher-labeled pairs selected by 4-way swap consistency and ordered as a
curriculum. Because the models, domains, and annotation sources differ, this is not
a scale-controlled data-efficiency comparison with SpeechJudge; it instead shows
that a comparatively small, confidence-filtered corpus is sufficient to obtain the
reported point accuracy in our narrow deployment domain.

Two concurrent generative reward models extend the same idea. GSRM \cite{gsrm}
decomposes naturalness into interpretable acoustic attributes followed by
structured reasoning, is trained on 31K expert annotations plus a test set of real
user interactions, and improves a speech LM when used as an RLHF reward.
UniSRM \cite{unisrm} is the closest prior work to our \emph{method}: it unifies
four assessment tasks --- single-utterance A/B preference, absolute multi-dimension
scoring, scene-conditioned style consistency, and multi-turn dialogue judgment ---
in one Qwen2.5-Omni thinker trained by SFT and then GRPO, with a reward summing a
format term, a final-answer accuracy term, and a reasoning-consistency term that
compares the sign of each predicted per-dimension score difference against the
teacher's. It also removes position artifacts at the data level by shuffling A/B
order and discarding cyclically inconsistent preferences. The difference from our
method is where the per-dimension signal lands. UniSRM averages per-dimension
agreement into a single scalar next to the final-answer term, so one group-level
advantage is broadcast across the entire rationale and disagreeing dimensions can
partially cancel. We instead remove the overall verdict from the training target
and give each dimension its own group-normalized advantage on its rationale span,
preventing direct cross-dimension cancellation in the token-level objective. Where
a single preference is still required --- for best-of-$N$ selection, or to compare
against a coupled judge --- we recover one by weighting the per-dimension score
differences, but that aggregation is an explicit rule applied outside the model
rather than a verdict the rationale is written toward. This design produces non-unanimous core-dimension outputs on 24.3--60.6\% of
evaluated pairs, providing direct evidence of output diversity across dimensions.

\subsection{Context- and dimension-aware judgment}
\label{ssec:rel_ceaeval}

CEAEval \cite{ceaeval} argues that expressiveness is only meaningful relative to
context and reframes the task as expressive \emph{appropriateness}: whether a clip
matches the communicative intent implied by its discourse-level narrative. It
contributes CEAEval-D, 16.1 hours of Mandarin conversational speech drawn from a
3505-hour audiobook corpus and annotated along fifteen expressive dimensions, and
CEAEval-M, which pairs a frozen text-only planner predicting an ideal expressive
profile with a Speech-LLM judge trained in three stages --- caption-based knowledge
distillation, CoT scoring SFT, and GRPO on a rebalanced score distribution. We
adopt their adaptive audio attention bias in Section \ref{ssec:attnbias}; it trains
stably but brings no clear accuracy gain in our setting. This null result does not
establish whether the mechanism becomes beneficial at longer rationale lengths.
Our task differs in
two ways that matter for reward modeling: we compare two renderings of the
\emph{same} transcript rather than absolute-scoring a single clip against a
text-derived ideal, and we require per-dimension verdicts that survive as an
independent reward vector rather than a single appropriateness score.


\section{Task formulation and benchmark}
\label{sec:task}

\subsection{Pairwise multi-dimensional judgment}
\label{ssec:formulation}

Let $(a_A, a_B, x)$ be an instance, where $a_A, a_B$ are two livestream audio
clips rendering the same target transcript $x$. A judge $\pi$ produces, for each
evaluated rubric dimension $d$, a rationale $c_d$ and an integer score pair
$(s_d^A, s_d^B) \in \{1,\ldots,10\}^2$. The dimension verdict is
%
\begin{equation}
v_d = \mathrm{sgn}(s_d^A - s_d^B) \in \{A, \text{Tie}, B\},
\end{equation}
%
with $v_d = \text{Tie}$ when the two scores are equal. Live-SpeechJudge evaluates
all seven dimensions and then generates an overall section containing a 1--10
score for each clip, an overall verdict $V \in \{A, \text{Tie}, B\}$, and its own
justification. Decouple-Live-SpeechJudge instead evaluates and emits only the four
core dimensions C1--C4, with no L1--L3 sections and no overall verdict. When an
aggregate comparison is required, we compute it externally from the four core
score differences. Overall accuracy is the agreement of the generated or
externally aggregated verdict with the human A/B label, and ties are counted as
errors.

\begin{table*}[t]
\centering
\footnotesize
\caption{Seven-dimension livestream prosody rubric (high-expressiveness profile).
The gentle profile keeps C1--C3 with softened anchors and replaces C4 with
``Gentle Style Fit'' (energy control, steadiness, warmth, real-human calm-host
feel).}
\label{tab:rubric}
\begin{tabular}{llll}
\toprule
ID & Dimension & Gate & Measured attributes \\
\midrule
C1 & Fluency \& Naturalness    & always & rate stability, phrasing, pause naturalness, coherence \\
C2 & Intonation Variation      & always & pitch range, sentence-final contour, stress pitch, inter-sentence contrast \\
C3 & Emotional Expression      & always & emotion-content match, intensity, warmth / appeal \\
C4 & Livestream Expressiveness & always & energy, rate drive, rhythmic momentum, real-host feel \\
L1 & Key Information Emphasis  & text   & acoustic prominence on price / discount / scarcity / call-to-action \\
L2 & Emotional State Switching & text   & capture of emotional turning points, transition smoothness \\
L3 & Interaction Mode Switching& text   & tone change at explanation-to-audience markers \\
\bottomrule
\end{tabular}
\end{table*}

\subsection{Rubric design}
\label{ssec:rubric}

Table \ref{tab:rubric} lists the rubric. C1--C4 are always evaluated. L1--L3 are
\emph{activation-gated}: the judge first decides from $x$ alone whether the
dimension applies and, if not, emits a short ``not activated'' section with both
scores 0, since forcing a verdict on an inapplicable dimension only injects noise
into aggregation.

\subsection{Test sets}
\label{ssec:testsets}

We deliberately cover three difficulty regimes, since a judge that only separates
human from TTS is useless for best-of-$N$.

\begin{itemize}
\itemsep0pt \parskip0pt \topsep2pt
\item \textbf{T1} qwen3-TTS base vs.\ SFT, 417 pairs. \textbf{T1$\star$} is the
119-pair subset on which all three annotators agree.
\item \textbf{T2} qwen3-TTS-GDPO vs.\ itself (two independent sampling runs), 111
pairs, three-annotator unanimous. Hardest regime: both clips come from the same
model, so the judge must distinguish quality variation among independent samples.
\item \textbf{T3} qwen3-TTS-GDPO vs.\ BERT-CosyVoice \cite{cosyvoice}, 315 pairs,
reported as two annotation batches T3a (166) and T3b (149). Cross-model regime.
\item \textbf{T4} human recording vs.\ qwen3-TTS, 200 pairs.
\end{itemize}

Construction is stratified rather than opportunistic. From a pool of 135 speakers
we sample 114: 70 warm-recommendation, 27 passionate-promotion and 17
professional-explanation speakers. Transcripts are matched to each speaker's
category (passionate: snacks, womenswear, personal care, large and kitchen
appliances; professional: phones, 3C, jewelry, gold; warm: baby, travel, beauty,
cosmetics, tableware) and drawn evenly from ASR transcripts of real streams and
from generated copy of three functional types (product entry, product
introduction, order-pressing). Durations are balanced over 5--10 s, 10--20 s and
20--30 s. Before annotation, all evaluation partitions are frozen. No evaluation
audio, audio pair, normalized transcript, speaker, or source livestream appears in
the SFT/GRPO training or validation data.

Three annotators label every pair independently under a written protocol, blinded
to system identity and audio source. They are Alibaba-contracted annotation staff
who completed task-specific professional training. Human annotation uses the same
rubric definitions, pairwise decision criteria and instructions as the Gemini
prompt. Averaged across the main T1--T4 test sets, 45\% of pairs receive a 3-0
decision and 55\% receive a 2A1E or 2B1E decision, where E denotes an abstention;
the final A/B label is determined by the non-abstaining majority. T1--T4 therefore
evaluate the overall pairwise verdict under the multidimensional rubric; separate
human tests for individual core dimensions are introduced in Section
\ref{ssec:perdimtest}.
Quality-control exclusions are limited to objective data problems such as corrupted
audio, transcript mismatch, or protocol violations, and are made without access to
model predictions.


\begin{figure*}[t]
\centering
\includegraphics[width=0.92\linewidth]{结构图.png}
\caption{Overview. \textbf{Left:} the chain-of-thought prompt shared by teacher and
student, which fixes the rubric and states the independence, priority and
no-exaggeration instructions. \textbf{Middle top:} the coupled judge
Live-SpeechJudge emits one section per rubric dimension --- four always-on plus
three text-activated --- and closes with an overall naturalness section and a
single conclusion. \textbf{Middle bottom:} Decouple-Live-SpeechJudge emits only
the four independently supervised core-dimension sections C1--C4 and no overall
verdict; the reported preference is their weighted aggregate. \textbf{Right:} the
training pipeline. For coupled SFT, Gemini-3.1-pro-preview labels each pair four
times and only 4-0 swap-consistent pairs are retained; for decoupled SFT, each core
dimension is queried separately and retained at 4-0, 3-1, or three agreeing
non-tie votes plus one abstention. Stage 1
trains on qwen3-TTS vs.\ human pairs and stage 2 on cross-model and same-model
pairs. Live-SpeechJudge GRPO uses one overall $\pm1$ agreement reward;
Decouple-Live-SpeechJudge GRPO uses separate $\pm1$ rewards on the C1--C4
rationale spans.}
\label{fig:overview}
\end{figure*}

\section{Method}
\label{sec:method}

Figure \ref{fig:overview} summarizes the system: one prompt, two rationale formats
--- coupled and decoupled --- and a three-phase training pipeline built on
swap-consistency-filtered teacher labels.

\subsection{Teacher distillation with swap-consistency filtering}
\label{ssec:distill}

Human annotators provide pairwise preferences, but collecting detailed rationales
at training scale is costly in our setting. We therefore distill
Gemini-3.1-pro-preview. Naive distillation can propagate the teacher's order
sensitivity and
low-confidence predictions. We therefore query the teacher four times per
pair: twice in the presented order and twice with $a_A$ and $a_B$ swapped. Let $k$
be the number of the four verdicts favoring the same clip after un-swapping. We
keep $k=4$ (``4-0'') as confident supervision and discard $k=3$ and $k=2$
(``2-2'') entirely. Disagreement across repeated and swapped queries indicates
lower teacher confidence and may reflect order sensitivity, sampling variability,
or intrinsic ambiguity. We un-swap the votes to identify the physical winner, then
assign retained training pairs to balanced presentation orders so that the
winner appears equally often in position A and position B. The supervision
therefore does not systematically associate the winning label with either slot.
Table \ref{tab:main} directly ablates this threshold using v0, which expands the
training set to 2300 pairs by admitting both 4-0 and 3-1 outcomes while keeping the
model, prompt, curriculum, optimization hyperparameters, and evaluation protocol
identical to v1.

\subsection{Two-stage curriculum SFT}
\label{ssec:sft}

The student is Qwen3-Omni. We attach LoRA \cite{lora} to attention projections
only ($q,k,v,o$), rank 32. Stage 1 trains on 820 qwen3-TTS-GDPO vs.\ human
pairs --- large-gap instances that teach the model what each dimension is. Stage 2
trains on 710 cross-model (vs.\ BERT-CosyVoice) and 600 same-model self-pairs,
which teach fine discrimination. Table \ref{tab:main} compares this sequential
schedule with stage-2-only training and with single-stage training on the merged
data. After SFT, the rank-32 adapter is merged into the backbone weights.

\subsection{Dimension-weighted token loss}
\label{ssec:weights}

We use per-section weights to control the contribution of each rubric dimension to
supervised fine-tuning. Let $T_d$ be the token indices of the section for dimension
$d$ in the target rationale, and $\alpha_d$ its SFT loss weight. We optimize
%
\begin{equation}
\mathcal{L}_{\text{SFT}}(\theta) = -\frac{\sum_{d} \alpha_d \sum_{t\in T_d} \log
\pi_\theta(y_t \mid y_{<t}, a_A, a_B, x)}{\sum_{d} \alpha_d\,|T_d|},
\end{equation}
%
which is the usual next-token loss with per-section reweighting and a matching
normalizer so that the effective learning rate does not drift with $\alpha_d$. For
the coupled model we use $\alpha = (1.5, 0.7, 1.5, 1.5, 1.5, 0.7, 0.7)$ over
(C1..C4, L1..L3), plus 1.0 on the free-text summary and 2.0 on the conclusion
line.

\subsection{GRPO post-training}
\label{ssec:grpo}

Error analysis on the SFT model shows that the remaining errors are mostly
incorrect verdicts on difficult cases, suggesting that the preference judgment has
not fully converged. This is exactly the regime where policy optimization against
a verifiable reward can help, so we post-train with GRPO \cite{grpo}.

For each prompt $q$ we sample a group of $G=8$ completions $\{o_i\}$ from
$\pi_{\theta_{\text{old}}}$. For a group-level scalar $z_i$, define
%
\begin{equation}
\mathcal{N}_G(z_i)=\frac{z_i-\mathrm{mean}(\{z_j\}_{j=1}^{G})}
{\mathrm{std}(\{z_j\}_{j=1}^{G})+\epsilon}.
\end{equation}
%
Live-SpeechJudge uses one overall reward for each completion,
%
\begin{equation}
r_i^{\mathrm{overall}}=
\begin{cases}
+1 & V_i=\tilde V,\\
-1 & \mathrm{otherwise},
\end{cases}
\end{equation}
%
where $\tilde V$ is the teacher's overall verdict. Its advantage
$\mathcal{N}_G(r_i^{\mathrm{overall}})$ is applied to every generated token.

For Decouple-Live-SpeechJudge GRPO, each retained core dimension receives its own
reward,
%
\begin{equation}
r_{i,d}=
\begin{cases}
+1 & v_{i,d}=\tilde v_d,\\
-1 & \mathrm{otherwise},
\end{cases}
\qquad d\in\{\mathrm{C1},\ldots,\mathrm{C4}\},
\end{equation}
%
and is normalized independently within the rollout group. A dimension without a
confident teacher A/B label is masked and contributes no advantage. Let $T_{i,d}$
be the token span of dimension $d$ in completion $i$, and let $\lambda_d$ denote
its GRPO span weight. The token-level advantage is
%
\begin{equation}
\hat A_{i,t}=\begin{cases}
\mathcal{N}_G(r_i^{\mathrm{overall}}) & \text{coupled judge},\\
\lambda_d\mathcal{N}_G(r_{i,d}) & \text{decoupled judge},\ t\in T_{i,d},\\
0 & \text{decoupled judge},\ t\notin\bigcup_d T_{i,d}.
\end{cases}
\end{equation}
%
We set $\lambda_d=1$ for all four core dimensions. Thus, the C1 reward is applied
only to the C1 rationale span and cannot directly cancel the C4 reward in the
token-level objective, although all dimensions still share model parameters. We
maximize the clipped surrogate with a reverse-KL anchor to the SFT reference,
%
\begin{multline}
\mathcal{J}(\theta) = \mathbb{E}\Big[\tfrac{1}{G}\sum_{i}\tfrac{1}{|o_i|}\sum_{t}
\min\big(\rho_{i,t}\hat{A}_{i,t}, \\
\mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\big)\Big]
- \beta\,\mathbb{D}_{\text{KL}}[\pi_\theta \Vert \pi_{\text{ref}}],
\end{multline}
%
with $\rho_{i,t} = \pi_\theta(o_{i,t}\mid q, o_{i,<t}) /
\pi_{\theta_{\text{old}}}(o_{i,t}\mid q, o_{i,<t})$ and the unbiased low-variance
$k_3$ estimator for the KL term. Starting from the SFT-merged checkpoint, both
GRPO runs initialize a new rank-64 LoRA adapter and use $G=8$, learning rate
$1\text{e-}5$, $\beta=0.04$, and three epochs. The rank-64 adapter is merged into
the backbone after GRPO. A dimension whose $G$ rollouts all receive the same reward
has zero normalized advantage and contributes no policy gradient. We add no format
reward.

\subsection{Dimension-wise supervision}
\label{ssec:decouple}

The teacher, SpeechJudge, and our coupled student show a tendency for
per-dimension verdicts to follow the overall winner. This motivates testing whether
separately generated labels and removal of the shared overall-verdict target can
produce more varied dimension-wise outputs. We split the rubric into
single-dimension prompts and query the teacher \emph{four times per dimension}, each
call producing a CoT and score pair for that dimension only. Because repeating this
procedure for every dimension would multiply annotation cost, we construct the
Decouple-Live-SpeechJudge data using only the four core dimensions C1--C4; L1--L3
are omitted from both the training targets and model outputs. Per dimension we keep
4-0, 3-1, and three-agreeing-votes-plus-one-abstention outcomes and mask the rest.
Consequently, an uncertain dimension contributes zero token loss in SFT and zero
reward advantage in GRPO, while confident dimensions from the same pair remain
trainable --- the retention decision is made per (pair, dimension), not per pair.
The surviving per-dimension responses are concatenated into one target rationale.

The student is trained and evaluated with one joint C1--C4 rubric prompt and emits
one verdict per core dimension with no aggregate conclusion. In the name
Decouple-Live-SpeechJudge, \emph{Decouple} refers to removing the shared
overall-verdict target and applying dimension-specific supervision and span-local
rewards.

\subsection{Adaptive audio attention bias}
\label{ssec:attnbias}

We also adopt the adaptive audio attention bias of \cite{ceaeval}, a mechanism
targeting a plausible failure mode of long-CoT audio
judgment: as the rationale grows, attention mass migrates to the already-generated
text and the $\sim$12k-character rubric, and the audio itself is progressively
ignored. We partition every token $j$ into four regions by a deterministic rule
--- AUDIO (positions inside \texttt{<|audio\_start|>}\allowbreak\,\ldots\,%
\allowbreak\texttt{<|audio\_end|>}), COT
(positions with label $\neq -100$, i.e.\ the assistant rationale), PROMPT
(everything before the \texttt{\#\# The text content} marker that is not AUDIO,
i.e.\ system prompt plus rubric), and BASE (the remainder: the target transcript
and turn markers) --- and apply a learnable per-head multiplicative gate on the
attention logits, $s_{ij} \leftarrow s_{ij} + \log g(j)$, with
%
\begin{equation}
g(j) = \begin{cases}
1 + \sigma(f_{\text{audio}}) & j \in \text{AUDIO}\\
2\sigma(f_{\text{prompt}})   & j \in \text{PROMPT}\\
\sigma(f_{\text{cot}})       & j \in \text{COT}\\
1                            & j \in \text{BASE.}
\end{cases}
\end{equation}
%
BASE is ungated and fixed at 1 to anchor the scale; initializing $f=0$ gives
$g = (1.5, 1, 0.5, 1)$, i.e.\ a mild prior that up-weights audio and down-weights
the rationale. A representative training instance has 2618 PROMPT, 426 AUDIO, 634
COT and 80 BASE tokens.

In our exploratory run, the mechanism trains stably but does not increase test-set
accuracy, so it is not included in the final model or the main result tables. The
run uses COT spans of $\sim$600--700 tokens and audio spans of $\sim$200--430
tokens; whether the mechanism becomes useful for substantially longer rationales
remains an open question.


\begin{table*}[t]
\centering
\footnotesize
\caption{Overall accuracy (\%) on four independent high-expressiveness test sets
and the T1$\star$ unanimous subset. Gemini uses one call. For our models, ``1
sample'' uses the first sampled judgment, whereas ``10-sample mean'' averages scores
from five judgments in each A/B presentation order. v1 is Qwen3-Omni + two-stage
SFT + dimension weights; v1+GRPO adds GRPO post-training. v0 is a
10-sample confidence-threshold ablation trained on 2300 pairs retained at either
4-0 or 3-1.}
\label{tab:main}
\begin{tabular}{lccccc}
\toprule
Model / test set & T1 (417) & T1$\star$ (119) & T2 (111) & T3 (315) & T4 (200) \\
\midrule
SpeechJudge-GRM \cite{speechjudge} & 233/417 = 55.88 & 86/119 = 72.27 & 54/111 = 48.65 & 170/315 = 53.97 & 84/200 = 42.00 \\
Gemini-3.1-pro-preview (1 sample) \cite{gemini} & 272/417 = 65.23 & 86/119 = 72.27 & 73/111 = 65.8 & 231/315 = 73.3 & 116/200 = 58.0 \\
\midrule
w/o stage-1 SFT (10-sample mean) & 221/417 = 53.0 & 86/119 = 72.3 & 67/111 = 60.4 & 220/315 = 69.9 & 91/200 = 45.5 \\
single-stage SFT, all data pairs (10-sample mean) & 303/417 = 72.7 & 96/119 = 80.67 & 75/111 = 67.6 & 252/315 = 80.0 & 149/200 = 74.5 \\
Live-SpeechJudge v0, 4-0 + 3-1 (10-sample mean) & 297/417 = 71.22 & 94/119 = 78.99 & 73/111 = 65.77 & 254/315 = 80.63 & 146/200 = 73.0 \\
\midrule
Live-SpeechJudge v1 (1 sample) & 271/417 = 64.99 & 89/119 = 74.79 & 70/111 = 63.06 & 233/315 = 73.97 & 133/200 = 66.50 \\
Live-SpeechJudge v1 (10-sample mean) & 295/417 = 70.74 & 96/119 = 80.67 & 78/111 = 70.27 & 259/315 = 82.22 & 160/200 = 80.00 \\
Live-SpeechJudge v1 + GRPO (1 sample) & 283/417 = 67.87 & 99/119 = \textbf{83.19} & 73/111 = 65.77 & 255/315 = 80.95 & 149/200 = 74.50 \\
Live-SpeechJudge v1 + GRPO (10-sample mean) & 297/417 = \textbf{71.22} & 98/119 = 82.35 & 78/111 = \textbf{70.27} & 261/315 = \textbf{82.86} & 167/200 = \textbf{83.5} \\
\bottomrule
\end{tabular}
\end{table*}

\section{Experiments}
\label{sec:exp}

\subsection{Setup}
\label{ssec:setup}

Live-SpeechJudge is initialized from Qwen3-Omni and trained with attention-only
LoRA in bf16 using the SWIFT framework \cite{swift}. All training runs use a fixed
random seed and are run on 4 NVIDIA 5000-series GPUs with 72GB memory each. Coupled
GRPO uses batched vLLM inference, whereas Decouple-Live-SpeechJudge GRPO uses native
Hugging Face generation. The Gemini annotation used for distillation costs
approximately RMB 10,000 in total. We compare against SpeechJudge-GRM
\cite{speechjudge} and Gemini-3.1-pro-preview under the same seven-dimension
prompt.
Gemini produces one judgment per test pair in the presented order without order
swapping, reflecting its intended single-call use and the cost of repeated
proprietary-model inference. SpeechJudge-GRM and our students each produce ten
sampled judgments per pair, five in each A/B presentation order. Swapped score
pairs are mapped back to the physical clips, the scores for each clip are averaged
across the ten judgments, and the final verdict is the sign of the resulting mean
score difference. To expose the contribution of inference-time sampling, we also
report a 1-sample result for our final checkpoints: this uses the score pair from
the first sampled judgment, without skipping to a later sample or resampling. An
equal-score tie remains a tie and is counted as an error against the A/B human
label. Gemini and each student 1-sample row use exactly one generated judgment per
test pair. The 10-sample rows additionally benefit from repeated sampling and
balanced order. We report final-verdict agreement with the human label as the
evaluation metric. For paired comparisons on the same test pairs, we estimate
sampling uncertainty using 10,000 paired bootstrap resamples. Each resample draws
test-pair indices with replacement and applies the same indices to both systems;
the 2.5th and 97.5th percentiles of the resulting accuracy-difference distribution
form the 95\% confidence interval.
Unless otherwise stated, training results are from one fixed-seed run and accuracy
differences are descriptive point estimates rather than multi-seed estimates.

\subsection{Main results}
\label{ssec:main}

Five observations follow from Table \ref{tab:main}.

\noindent\textbf{1.} Under 10-sample balanced-order inference, the final GRPO
model has higher point accuracy than a single Gemini judgment in every reported
column. This is a stronger inference operating point that uses repeated sampling and
order balancing, rather than a compute-matched comparison with Gemini.

\noindent\textbf{2.} Ten-sample aggregation usually improves agreement. Relative
to one sample, the final GRPO model gains 3.35, 4.50, 1.91, and 9.00 points on T1,
T2, T3, and T4, respectively; T1$\star$ instead decreases by 0.84 points. This
comparison reflects the combined effect of repeated sampling and A/B order
balancing rather than either factor in isolation.

\noindent\textbf{3.} At the 10-sample operating point, the descriptive change from
v1 SFT to v1+GRPO is $+0.48$, $+1.68$, $0.00$, $+0.63$, and $+3.50$ points on T1,
T1$\star$, T2, T3, and T4, respectively. The measured contribution therefore
varies substantially across test sets.

\noindent\textbf{4.} The stricter filtering configuration performs better in four
of five columns despite using fewer pairs. Relative to v1's 4-0-only supervision,
v0 admits 3-1 outcomes and grows to 2300 pairs, but its 10-sample accuracy is lower
by 1.68 points on T1$\star$, 4.50 on T2, 1.59 on T3, and 7.00 on T4; T1 is higher
by 0.48 points. This ablation favors 4-0-only filtering in the present training
configuration, but does not separate label confidence from the accompanying change
in dataset composition.

\noindent\textbf{5.} Relative to the 10-sample v1 SFT row, removing the
qwen3-TTS-GDPO-vs.-human stage is associated with a 34.5-point lower T4 estimate
(45.5\%) in the human-vs.-TTS regime. This is consistent with, but does not prove,
large quality-gap examples helping establish basic prosody-judgment behavior before
fine-grained training. With the same data merged into one stage, accuracy is lower
than sequential training on T2, T3, and T4, higher on T1, and equal on
T1$\star$. These single-run results motivate the sequential schedule but do not
isolate curriculum order from training dynamics.

\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Paired-bootstrap comparison of 10-sample Live-SpeechJudge v1+GRPO
against v1 SFT. Differences are in percentage points; confidence intervals use
10,000 paired resamples.}
\label{tab:mainbootstrap}
\begin{tabular}{lcc}
\toprule
Test set & GRPO $-$ SFT & 95\% CI \\
\midrule
T1 & $+0.48$ & [$-2.64$, $+3.60$] \\
T1$\star$ & $+1.68$ & [$-2.52$, $+6.72$] \\
T2 & $0.00$ & [$-7.21$, $+7.21$] \\
T3 pooled & $+0.63$ & [$-1.90$, $+3.18$] \\
T4 & $+3.50$ & [$-1.50$, $+8.50$] \\
\bottomrule
\end{tabular}
\end{table}

All intervals in Table \ref{tab:mainbootstrap} include zero. We therefore treat
the 10-sample SFT-to-GRPO changes as descriptive point estimates rather than
statistically significant improvements under test-pair sampling uncertainty.
T1$\star$ is a subset of T1 and is not treated as independent evidence.

\begin{table}[t]
\centering
\footnotesize
\caption{Position statistics on the 417-pair T1 test set
(qwen3-TTS base vs.\ SFT). Each physical pair is scored 10 times with balanced A/B
presentation order, yielding 4170 judgments per model. Mean A/B are the reported
1--10 overall-section scores for the presentation slots after pooling both physical
clip assignments; $t$ is computed from pair-level mean slot differences.}
\label{tab:posbias}
\begin{tabular}{lccccc}
\toprule
Model & Mean A & Mean B & B$-$A & $t$ & B win \\
\midrule
Live-SpeechJudge & 7.842 & 7.854 & $+0.012$ & $+0.4$  & 50.7\% \\
SpeechJudge-GRM  & 7.061 & 8.039 & $+0.978$ & $+11.7$ & 66.8\% \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Position bias}
\label{ssec:posbias}

SpeechJudge-GRM assigns the second-presented clip nearly one additional point on
average and selects B in 66.8\% of judgments under this protocol. By contrast,
Live-SpeechJudge has a measured B$-$A gap of only $+0.012$ and a 50.7\% B win rate
(Table \ref{tab:posbias}). This smaller measured gap is consistent with our
position-balanced supervision: after votes are mapped back to the physical clips,
retained pairs are assigned to presentation orders so that the winner appears
equally often in position A and position B.

\subsection{Dimension-wise supervision and output diversity}
\label{ssec:decoupanalysis}

D-LSJ removes the shared overall-verdict target and applies dimension-specific
supervision and span-local rewards. Across the evaluation sets, 24.3--60.6\% of
its two-stage-SFT outputs contain at least two different non-tie core-dimension
verdicts, demonstrating the capacity to issue different dimension verdicts.
As a teacher-defined cross-dimension stress test, we further select 140 audio pairs
for which all four independently queried teacher labels are confident (4-0, 3-1,
or three agreeing non-tie votes plus one abstention) but disagree across
dimensions.
On this same set, the coupled LSJ produces no non-unanimous core-verdict vector
(0/140) and obtains 48.0\% per-dimension agreement with the teacher. D-LSJ after
SFT produces 90/140 non-unanimous outputs and 380/560 (67.9\%) per-dimension
agreement; span-local GRPO further changes these results to 100/140 (71.4\%) and
400/560 (71.4\%), respectively. Because this stress set is selected by disagreement
among independently queried teacher labels, it diagnoses output diversity and
teacher alignment rather than human-grounded per-dimension correctness. Within
that scope, the same-set comparison associates the decoupled training design with
fewer unanimous verdict vectors and higher agreement with the teacher labels.

D-LSJ also allows weighted aggregation of its dimension verdicts. We reserve a
held-out validation split from the training pool exclusively for selecting the
aggregation weights, choose $w=(3,0.5,0.5,3)$ on that split, and freeze it before
evaluating on any test set. Table \ref{tab:weights} shows that this
validation-selected scheme, which emphasizes C1 and C4, has a 4.3-point higher T1
estimate than uniform C1--C4 weights. In the descriptive cross-configuration
comparison against the coupled LSJ+GRPO row, D-LSJ after two-stage SFT has higher
point accuracy on T1, T4, and T2, lower accuracy on T1$\star$, and slightly lower
accuracy when T3a and T3b are pooled. Because the training objectives and stages
differ, this comparison does not isolate the effect of decoupling. The remaining
SFT rows are fixed-weight ablations rather
than candidates selected from the test results. We evaluate the effect of
per-dimension GRPO only against human per-dimension labels in Section
\ref{ssec:perdimtest}.

\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Dimension-wise student after two-stage SFT, aggregated under different
inference weight vectors $w_d$ over the four core dimensions in the order
C1/C2/C3/C4 (\%). The $3/.5/.5/3$ scheme is selected on a held-out validation
split and frozen before test evaluation; the other SFT schemes are fixed-weight
ablations. The aggregate verdict is the sign of $\sum_d w_d (s_d^A - s_d^B)$.
Columns are T1 (417 pairs), T1$\star$ (119), T4 (200), T3a (166), T3b (149) and T2
(111). The first three rows use coupled-judge outputs, for which T3 is reported as
a single 315-pair number.}
\label{tab:weights}
\begin{tabular}{lcccccc}
\toprule
Weight scheme & T1 & T1$\star$ & T4 & T3a & T3b & T2 \\
\midrule
LSJ v1 + GRPO (coupled) & 71.2 & 82.4 & 83.5 & \multicolumn{2}{c}{82.9} & 70.3 \\
Gemini (coupled prompt) & 65.2 & 72.3 & 58.0 & \multicolumn{2}{c}{73.3} & 65.8 \\
Gemini, $3/2/1/1$ & 66.7 & 81.5 & 75.5 & \multicolumn{2}{c}{75.2} & 60.4 \\
\midrule
\multicolumn{7}{l}{\emph{Decouple-Live-SpeechJudge (two-stage SFT)}}\\
\midrule
Uniform (C1--C4) & 69.3 & 78.2 & 78.5 & 73.5 & 83.9 & 71.2 \\
$3/2/1/1$ & 69.1 & 76.5 & 77.5 & 74.7 & 80.5 & 71.2 \\
$3/.5/.5/3$ & \textbf{73.6} & \textbf{80.7} & \textbf{85.5} & \textbf{77.7} & 86.6 & 71.2 \\
C2C3C4 only & 68.4 & 76.5 & 79.5 & 74.1 & 82.6 & 70.3 \\
C4 only & 69.3 & 79.0 & 79.5 & 75.3 & 85.2 & 73.0 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Human-annotated per-dimension evaluation}
\label{ssec:perdimtest}

The four evaluation sets isolate one core rubric dimension each and test whether a
judge recovers a high-confidence human label for that target dimension. We use
separate dimension-specific sets because, for a given audio pair, a confident human
preference in one dimension does not imply confident preferences in all four;
requiring a complete C1--C4 verdict vector would either discard reliable
single-dimension labels or force uncertain dimensions into the evaluation. These
sets therefore test human agreement for individual dimensions rather than full
C1--C4 verdict vectors on the same audio pairs. The same three trained annotation
staff independently label each pair using the corresponding dimension section of
the Gemini prompt. Every retained pair has a unanimous 3-0, non-tie label for its
target dimension; majority-with-abstention cases are excluded.

\begin{table}[t]
\centering
\scriptsize
\renewcommand{\arraystretch}{0.82}
\setlength{\tabcolsep}{2pt}
\caption{Point agreement with high-confidence human labels (\%). Gemini uses one
call; D-LSJ uses the first sampled judgment (1-sample) or the balanced-order mean
(10-sample). Parentheses give set sizes.}
\label{tab:perdimtest}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Model / inference & C1 (224) & C2 (172) & C3 (142) & C4 (160) & Total (698) \\
\midrule
Gemini (1-sample) & 74.55 & 77.33 & 80.28 & 84.38 & 78.65 \\
\midrule
D-LSJ (1-sample) & 68.75 & 65.70 & 76.06 & 86.88 & 73.64 \\
D-LSJ (10-sample) & 79.02 & 80.81 & 85.92 & 93.13 & 84.10 \\
D-LSJ + Agg.-GRPO (1-sample) & 70.54 & 71.51 & 79.58 & 90.62 & 77.22 \\
D-LSJ + Agg.-GRPO (10-sample) & 79.91 & \textbf{83.14} & \textbf{87.32} & 92.50 & 85.10 \\
D-LSJ + Span-GRPO (1-sample) & 71.88 & 76.74 & 76.76 & 88.75 & 77.94 \\
D-LSJ + Span-GRPO (10-sample) & \textbf{83.04} & \textbf{83.14} & 85.21 & \textbf{94.38} & \textbf{86.10} \\
\bottomrule
\end{tabular}%
}
\end{table}

In this fixed-seed comparison, aggregate-reward GRPO changes pooled agreement over
SFT from 73.64\% to 77.22\% with one sample and from 84.10\% to 85.10\% with 10
samples. Span-local GRPO has pooled point estimates of 77.94\% and 86.10\%,
respectively, 0.72 and 1.00 points above the shared aggregate-advantage run.

\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Paired-bootstrap comparisons for 10-sample per-dimension agreement.
Entries report the point-accuracy difference in percentage points and its 95\%
confidence interval from 10,000 paired resamples.}
\label{tab:perdimbootstrap}
\begin{tabular}{lcc}
\toprule
Dimension & Span-GRPO $-$ SFT & Span-GRPO $-$ Agg.-GRPO \\
\midrule
C1 & $+4.02$ [$-0.89$, $+9.38$] & $+3.12$ [$-1.79$, $+7.59$] \\
C2 & $+2.33$ [$-2.33$, $+6.98$] & $0.00$ [$-5.23$, $+5.23$] \\
C3 & $-0.70$ [$-4.23$, $+2.82$] & $-2.11$ [$-6.34$, $+2.11$] \\
C4 & $+1.25$ [$-1.88$, $+4.38$] & $+1.88$ [$-1.88$, $+5.62$] \\
\bottomrule
\end{tabular}
\end{table}

At 10 samples, Span-GRPO has higher point estimates than SFT on C1, C2, and C4
and a lower estimate on C3. Relative to aggregate-reward GRPO, it is higher on C1
and C4, equal on C2, and lower on C3. All per-dimension bootstrap intervals in
Table \ref{tab:perdimbootstrap} include zero, so we treat these differences as
descriptive rather than statistically significant. The bootstrap intervals
quantify test-pair sampling uncertainty for this fixed-seed run and do not estimate
variation across training seeds.

\subsection{Best-of-N selection}
\label{ssec:bestofn}

We additionally deploy Live-SpeechJudge as the decision component of a
Best-of-8 inference pipeline. For each transcript, the TTS model samples eight
candidate utterances. Live-SpeechJudge evaluates all
$\binom{8}{2}=28$ distinct candidate pairings, and the selected utterance is the
candidate with the largest number of pairwise wins,
%
\begin{equation}
\hat a=\arg\max_{a_i}\sum_{j\ne i}\mathbf{1}[V(a_i,a_j)=a_i].
\end{equation}
%
This tournament-style aggregation requires no absolute quality calibration and
uses the same pairwise judgment task on which the model is trained. Professionally
trained contractor annotators evaluate 400 eight-candidate sets. For a
high-confidence evaluation, we retain the 136 sets in which two annotators produce
identical ordered top-3 rankings, following the use of unanimous evaluation labels
in SpeechJudge-Eval \cite{speechjudge}. The remaining 264 sets are excluded because
the perceptual differences among their candidates are not sufficiently distinct to
support a confident ordered top-3 annotation; this confidence-based inclusion
decision is made without reference to model predictions. Hit@$k$ indicates that
the tournament winner lies within this agreed human top-$k$ set. On these 136
retained sets, the selector obtains Hit@1, Hit@2, and Hit@3 rates of 98/136 =
72.06\%, 106/136 = 77.94\%, and 116/136 = 85.29\%, respectively, compared with
random-selection baselines of 12.5\%, 25.0\%, and 37.5\%. These rates characterize
the high-confidence test set and are not estimates over all 400 collected sets.
Qualitative examples on our demo page additionally illustrate the serving workflow
and compare the selected utterance with the other seven candidates.

\subsection{Scope of the evidence}

Our results have four boundaries. First, training comparisons use one fixed seed,
so accuracy differences are descriptive point estimates rather than estimates of
run-to-run variation. The paired-bootstrap intervals quantify uncertainty from the
sampled test pairs, not variation across training seeds. All intervals for the
coupled judge's 10-sample SFT-to-GRPO comparison, as well as all reported
per-dimension intervals for Span-GRPO versus SFT and aggregate-reward GRPO, include
zero. Second, the 10-sample rows use more inference and balanced presentation order
than the single-call Gemini baseline; they characterize a stronger operating point
rather than a compute-matched comparison. Third, the 140-pair cross-dimension stress
set is defined by independently queried teacher labels, whereas the human
per-dimension sets evaluate one target dimension per audio pair. Finally, the
Best-of-8 rates apply to the 136 high-confidence sets retained from 400 collected
sets. D-LSJ is limited to C1--C4, and using its vector outputs to optimize the TTS
backbone remains future work.

\section{Conclusion}
\label{sec:conclusion}

We presented Live-SpeechJudge, a pairwise multi-dimensional prosody judge for
e-commerce livestream TTS. Its 10-sample balanced-order aggregation has higher
point accuracy than one Gemini judgment on all reported columns and shows a small
presentation-slot gap. This comparison uses a stronger inference operating point
rather than matched inference compute. The paired-bootstrap intervals for the
coupled judge's 10-sample SFT-to-GRPO changes all include zero, so those changes
remain descriptive. Decouple-Live-SpeechJudge removes the overall-verdict target
and applies dimension-specific supervision and span-local rewards to C1--C4. Its
non-unanimous outputs demonstrate varied dimension verdicts. In one fixed-seed run,
Span-GRPO changes pooled human agreement from 73.64\% to 77.94\% with one sample
and from 84.10\% to 86.10\% with 10 samples, although all per-dimension confidence
intervals include zero. On the retained 136-set Best-of-8 evaluation, the selected
candidate achieves Hit@1/2/3 rates of 72.06\%/77.94\%/85.29\%. Future work will
use D-LSJ's dimension-wise judgments to optimize the TTS backbone.


% Note: no \clearpage here on purpose. ICASSP counts references inside the 4-page
% limit, so the bibliography must start in the column where the conclusion ends and
% only spill onto the references-only 5th page.
\begin{thebibliography}{99}
\setlength{\itemsep}{0pt}\setlength{\parsep}{0pt}\setlength{\parskip}{0pt}
\bibitem{dpo}
R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn,
``Direct preference optimization: Your language model is secretly a reward
model,'' in \emph{Advances in Neural Information Processing Systems}, 2023.
\bibitem{grpo}
Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li,
Y. Wu, and D. Guo, ``DeepSeekMath: Pushing the limits of mathematical reasoning
in open language models,'' \emph{arXiv preprint arXiv:2402.03300}, 2024.
\bibitem{mosnet}
C.-C. Lo, S.-W. Fu, W.-C. Huang, X. Wang, J. Yamagishi, Y. Tsao, and H.-M. Wang,
``MOSNet: Deep learning-based objective assessment for voice conversion,'' in
\emph{Proc. Interspeech}, 2019, pp. 1541--1545.
\bibitem{utmos}
T. Saeki, D. Xin, W. Nakata, T. Koriyama, S. Takamichi, and H. Saruwatari,
``UTMOS: UTokyo-SaruLab system for VoiceMOS Challenge 2022,'' in
\emph{Proc. Interspeech}, 2022, pp. 4521--4525.
\bibitem{nisqa}
G. Mittag, B. Naderi, A. Chehadi, and S. M\"oller, ``NISQA: A deep CNN-self-attention
model for multidimensional speech quality prediction with crowdsourced datasets,''
in \emph{Proc. Interspeech}, 2021, pp. 2127--2131.
\bibitem{speechjudge}
X. Zhang, C. Wang, H. Liao, Z. Li, Y. Wang, L. Wang, D. Jia, Y. Chen, X. Li,
Z. Chen, and Z. Wu, ``SpeechJudge: Towards human-level judgment for speech
naturalness,'' \emph{arXiv preprint arXiv:2511.07931}, 2025.
\bibitem{gsrm}
M. Shen, T. Jayashankar, O. Hanna, N. Kanda, Y. Wang, K. Zmolikova, R. Xie,
N. Moritz, A. Xu, Y. Gaur, G. Wornell, Q. He, and J. Wu, ``GSRM: Generative speech
reward model for speech RLHF,'' \emph{arXiv preprint arXiv:2602.13891}, 2026.
\bibitem{unisrm}
Y. Wang, D. Yang, Y. Deng, Z. Wu, Y. Guo, H. Meng, and X. Wu, ``UniSRM: A unified
speech reward model for reasoning-based fine-grained assessment,'' \emph{arXiv
preprint arXiv:2605.23261}, 2026.
\bibitem{emergentttseval}
R. R. Manku, Y. Tang, X. Shi, M. Li, and A. Smola, ``EmergentTTS-Eval: Evaluating
TTS models on complex prosodic, expressiveness, and linguistic challenges using
model-as-a-judge,'' in \emph{Advances in Neural Information Processing Systems},
2025.
\bibitem{swanbench}
C. Pan, R. Yang, H. Wang, Z. Zhou, and X. He, ``Comprehensive benchmarking of
long-form speech generation in diverse scenarios,'' \emph{arXiv preprint
arXiv:2605.28618}, 2026.
\bibitem{ceaeval}
T. Wang, Z. Ma, Y. Peng, H. Wang, Z. Niu, Z. Huang, Y. Wu, Y.-W. Chao, Y. Jiang,
et al., ``Evaluating the expressive appropriateness of speech in rich contexts,''
\emph{arXiv preprint arXiv:2605.09413}, 2026.
\bibitem{gemini}
Gemini Team, Google, ``Gemini: A family of highly capable multimodal models,''
\emph{arXiv preprint arXiv:2312.11805}, 2023.
\bibitem{mtbench}
L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li,
D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, ``Judging
LLM-as-a-judge with MT-Bench and Chatbot Arena,'' in \emph{Advances in Neural
Information Processing Systems}, 2023.
\bibitem{qwen3omni}
J. Xu et al., ``Qwen3-Omni technical report,'' \emph{arXiv preprint
arXiv:2509.17765}, 2025.
\bibitem{dapo}
Q. Yu et al., ``DAPO: An open-source LLM reinforcement learning system at
scale,'' \emph{arXiv preprint arXiv:2503.14476}, 2025.
\bibitem{cosyvoice}
Z. Du, Q. Chen, S. Zhang, K. Hu, H. Lu, Y. Yang, H. Hu, S. Zheng, Y. Gu, Z. Ma,
Z. Gao, and Z. Yan, ``CosyVoice: A scalable multilingual zero-shot text-to-speech
synthesizer based on supervised semantic tokens,'' \emph{arXiv preprint
arXiv:2407.05407}, 2024.
\bibitem{lora}
E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen,
``LoRA: Low-rank adaptation of large language models,'' in \emph{Proc.
International Conference on Learning Representations}, 2022.
\bibitem{swift}
Y. Zhao, J. Huang, J. Hu, X. Wang, Y. Mao, D. Zhang, Z. Jiang, Z. Wu, B. Ai,
A. Wang, W. Zhou, and Y. Chen, ``SWIFT: A scalable lightweight infrastructure for
fine-tuning,'' in \emph{Proc. AAAI Conference on Artificial Intelligence}, 2025.
\end{thebibliography}

\end{document}
