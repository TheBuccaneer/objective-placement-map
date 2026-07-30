# Adversariale Tiefenanalyse der CPU/GPU-Energie- und Laufzeitkampagne

**Stand:** 30. Juli 2026  
**Datenbasis:** aktueller `new`-Unterbaum des öffentlichen Repositories, aktuelle workloadbezogene Audit-Ausgaben und die darin enthaltenen Session-Median-Zusammenfassungen.  
**Analyseeinheit:** `(Workload, Problemgröße/Conv2D-Shape)` mit vier Plattformen; CPU-Plattformwerte sind jeweils die punktoptimalen Threadkonfigurationen für das betrachtete Ziel.  
**Statistische Primäreinheit:** fünf Sessionmediane; die zehn Wiederholungen pro Session werden nicht als unabhängiges `n=50` behandelt.

## Executive Verdict

Die bisherige Hauptgeschichte trägt, aber in einer präziseren und stärkeren Form:

> **Die Kampagne zeigt nicht bloß, dass das schnellste Gerät gelegentlich nicht das energieärmste ist. Sie zeigt eine stabile, größen- und workloadabhängige Entscheidungslandkarte, in der 24 von 51 Zellen einen harten Zielkonflikt besitzen und statische GPU-Policies hohe, asymmetrische Regrets erzeugen.**

Alle elf vorgegebenen Ausgangsaussagen konnten aus den aktuellen Dateien reproduziert werden. Die zwei wichtigsten Korrekturen betreffen nicht die Zahlen, sondern die Provenienz:

1. Der ältere strategische Plan und das ältere 42-Spalten-Schema sind nicht der aktuelle Messvertrag.
2. REDUCTION ist im aktuellen Stand eine echte FP32-Summe `sum(x[0:N])`, nicht mehr `dot(x, ones)`.

## 1. Reproduktionsgate

|   claim_id | claim                                             | reported                                                   | reproduced   | own_value                                                       | deviation                                                                 | cause                                                                                                                         |
|-----------:|:--------------------------------------------------|:-----------------------------------------------------------|:-------------|:----------------------------------------------------------------|:--------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------|
|          1 | 51 workload-size/shape cells                      | 51                                                         | yes          | 51                                                              | 0                                                                         | Five 9-size workloads plus six Conv2D shapes.                                                                                 |
|          2 | Robust runtime/energy conflicts                   | 24/51                                                      | yes          | 24/51                                                           | 0                                                                         | Current tie-aware placement outputs.                                                                                          |
|          3 | Conflicts by workload                             | GEMM 2; STRIDED 2; AXPY 5; STREAM 6; REDUCTION 6; CONV2D 3 | yes          | GEMM 2; STRIDED_GEMM 2; AXPY 5; STREAM 6; REDUCTION 6; CONV2D 3 | 0                                                                         | Current placement tables; older dossiers contained stale 3/9 and 6/9 counts.                                                  |
|          4 | Pure RTX 3090/RTX 5060 Ti conflicts               | 23/24                                                      | yes          | 23/24                                                           | 0                                                                         | Only exception: STREAM 4M, runtime AMD and energy RTX 5060 Ti.                                                                |
|          5 | Large AXPY/STREAM/REDUCTION regime                | 15/15: runtime 3090, energy 5060 Ti                        | yes          | 15/15                                                           | 0                                                                         | All five sizes 16M–256M in each of the three workloads.                                                                       |
|          6 | Median large-regime effect                        | 2.12x faster; 42.5% energy saving                          | yes          | 2.121771x; 42.456627%                                           | rounding only                                                             | Median across 15 structurally related cells.                                                                                  |
|          7 | EDP point winner equals runtime point winner      | 48/51                                                      | yes          | 48/51                                                           | 0                                                                         | Exceptions: GEMM 128, STRIDED_GEMM 128, REDUCTION 1M.                                                                         |
|          8 | Always-RTX-3090 GPU-only median energy regret     | 72.4%                                                      | yes          | 72.382157%                                                      | rounding only                                                             | Per-cell GPU-only energy oracle, 51 equally weighted cells.                                                                   |
|          9 | Always-RTX-5060-Ti GPU-only median runtime regret | 110.6%                                                     | yes          | 110.563908%                                                     | rounding only                                                             | Per-cell GPU-only runtime oracle, 51 equally weighted cells.                                                                  |
|         10 | Intel REDUCTION 256M expensive last runtime gain  | ~0.24% runtime; 39.45% energy                              | yes          | 0.244180% runtime; 39.449294% energy                            | runtime differs by 0.0006 percentage points from rounded source statement | 8T runtime optimum versus 4T energy optimum; point medians. Pairwise audit reports 0.243585% due its exact ratio aggregation. |
|         11 | All canonical conflicts have 5/5 session support  | 24/24                                                      | yes          | 24/24 cells; 48/48 objective directions at 5/5                  | 0                                                                         | Fixed selected configurations evaluated session-wise; AXPY table reports support directly.                                    |

### Konfliktverteilung

| workload     |   cells |   conflicts |
|:-------------|--------:|------------:|
| AXPY         |       9 |           5 |
| CONV2D       |       6 |           3 |
| GEMM         |       9 |           2 |
| REDUCTION    |       9 |           6 |
| STREAM       |       9 |           6 |
| STRIDED_GEMM |       9 |           2 |

Die einzige der 24 Konfliktzellen, die **kein reiner GPU-vs-GPU-Konflikt** ist, ist `STREAM, 4M`: AMD gewinnt Laufzeit, die RTX 5060 Ti Energie. Die übrigen 23 Konflikte sind RTX 3090 versus RTX 5060 Ti.

## 2. Der stärkste breite Befund: das 15-von-15-Regime

Für AXPY, STREAM und REDUCTION gilt bei 16M, 32M, 64M, 128M und 256M Elementen in jeder der 15 Zellen:

- RTX 3090: Laufzeitoptimum,
- RTX 5060 Ti: Energieoptimum,
- beide Richtungen jeweils mit 5/5 Session-Support.

Die 15 Zellen sind strukturell verwandt und werden nicht als 15 unabhängige Experimente interpretiert.

| Workload   |   3090 speedup median | speedup range   |   5060Ti energy saving median % | saving range %   |   3090/5060 power median |   3090 EDP advantage median % |
|:-----------|----------------------:|:----------------|--------------------------------:|:-----------------|-------------------------:|------------------------------:|
| AXPY       |                 2.126 | 2.106–2.141     |                          42.457 | 42.0–43.2        |                    3.696 |                        18.159 |
| REDUCTION  |                 2.08  | 2.053–2.088     |                          50.483 | 49.6–50.6        |                    4.206 |                         3.042 |
| STREAM     |                 2.135 | 2.111–2.145     |                          38.876 | 38.5–39.3        |                    3.491 |                        23.356 |

Über alle 15 Zellen beträgt der Median:

- **2,121771× Laufzeitvorteil** der RTX 3090,
- **42,4566 % Boardenergieersparnis** der RTX 5060 Ti,
- **3,69648× Boardleistungsratio** RTX 3090/RTX 5060 Ti,
- **1,73782× Energieverbrauch** der RTX 3090,
- **18,1593 % EDP-Vorteil** der RTX 3090.

### Neue Differenzierung innerhalb desselben Gewinnerregimes

Die Gewinnerreihenfolge ist identisch, die Stärke des Kompromisses aber nicht:

- **REDUCTION:** 5060-Ti-Energieersparnis median 50,48 %, aber 3090-EDP-Vorteil nur 3,04 %.
- **STREAM:** Energieersparnis 38,88 %, 3090-EDP-Vorteil 23,36 %.
- **AXPY:** Energieersparnis 42,46 %, 3090-EDP-Vorteil 18,16 %.

Damit ist der Satz „EDP verhält sich wie Laufzeit“ als Gesamtbefund korrekt, aber mechanistisch zu grob. Bei REDUCTION liegt EDP nahe an der Trennlinie, bei STREAM deutlich auf der Laufzeitseite.

## 3. Energiezerlegung: Warum Race-to-idle nicht reicht

Im großen Regime ist die RTX 3090 median 2,12× schneller, benötigt aber 3,70× die Boardleistung. Auf der Log-Skala kompensiert der Zeitvorteil median nur **57,65 %** des Leistungsnachteils. Übrig bleibt ein Energieverhältnis von 1,74 zugunsten der RTX 5060 Ti.

Workloadspezifisch kompensiert die kürzere Laufzeit:

- STREAM: etwa 60,6 % des Leistungsnachteils,
- AXPY: etwa 57,6 %,
- REDUCTION: nur etwa 51,1 %.

Dies ist eine deskriptive Zerlegung, keine mikroarchitektonische Kausalerklärung. Die geringe Restabweichung von der exakten Identität `E=P*t` entsteht, weil Energie, Laufzeit und Leistung getrennt über Sessionmediane aggregiert werden.

## 4. EDP ist in dieser Kampagne überwiegend geschwindigkeitsorientiert

Der EDP-Punktgewinner entspricht in **48 von 51 Zellen** dem Laufzeit-Punktgewinner. Die drei Ausnahmen sind:

- GEMM, N=128,
- STRIDED_GEMM, N=128,
- REDUCTION, N=1M.

Über die vollständigen Rangfolgen der vier Plattformen ergibt sich:

- Median Spearman(EDP, Laufzeit): **1,0**,
- Median Kendall(EDP, Laufzeit): **1,0**,
- Median Spearman(EDP, Energie): **0,8**,
- Median Kendall(EDP, Energie): **0,667**.

Zulässiger Claim:

> *In this campaign, EDP behaves predominantly as a speed-oriented objective and does not resolve the observed energy–runtime placement conflict.*

Nicht zulässig wäre die allgemeine Aussage, EDP sei grundsätzlich kein Kompromissmaß.

## 5. Policy-Regret

### GPU-only, exakt zur Ausgangshypothese

- Immer RTX 3090: median **72,382 % Energie-Regret**.
- Immer RTX 5060 Ti: median **110,564 % Laufzeit-Regret**.

### All-platform statische Policies

| metric   | policy   |   median_pct |   geomean_pct |   p90_pct |   max_pct |   cvar10_pct |   within_10_pct |
|:---------|:---------|-------------:|--------------:|----------:|----------:|-------------:|----------------:|
| runtime  | INTEL    |      1709.31 |       1167.36 |   2933.65 |   4063.72 |      3291.22 |            3.92 |
| runtime  | AMD      |       803.11 |        514.77 |   1470.02 |   1976.37 |      1792.47 |            5.88 |
| runtime  | 3090     |         0    |         17.75 |    112.65 |    131.39 |       121.65 |           74.51 |
| runtime  | 5060ti   |       113.45 |        129.2  |    406.27 |    439.32 |       427.35 |           17.65 |
| energy   | INTEL    |       644.56 |        595.8  |    949.81 |   2472.03 |      1386.67 |            3.92 |
| energy   | AMD      |       581.76 |        423.73 |    820.68 |   1024.69 |       888.35 |            3.92 |
| energy   | 3090     |        73.26 |         89.26 |    441.91 |    659.15 |       558.74 |           31.37 |
| energy   | 5060ti   |         0    |         21.79 |    126.84 |    188.5  |       155.99 |           70.59 |
| edp      | INTEL    |     10256.6  |       7164.95 |  25185.4  |  59985.2  |     38729.9  |            3.92 |
| edp      | AMD      |      4803.05 |       2530.69 |  10768.7  |  17642    |     13246.3  |            7.84 |
| edp      | 3090     |         0    |         77.29 |   1022.03 |   1496.56 |      1278.01 |           68.63 |
| edp      | 5060ti   |        31.53 |        122.18 |   1045.62 |   1453.61 |      1199.52 |           31.37 |

Die RTX 3090 ist die beste globale statische Laufzeit- und EDP-Policy; die RTX 5060 Ti ist die beste globale statische Energie-Policy. Trotzdem bleiben die Tail-Kosten groß.

### Wie viel Kontext hilft?

Ein nur workloadabhängiger Laufzeitentscheider wählt bei allen sechs Workloads die RTX 3090 und verbessert daher gegenüber der globalen statischen Policy nichts. Für Energie wählt ein workloadabhängiger Durchschnittsentscheider:

- RTX 5060 Ti für AXPY, Conv2D, REDUCTION und STREAM,
- RTX 3090 für GEMM und STRIDED_GEMM.

Trotz dieser zusätzlichen Kontextinformation erreicht er **610,1 % worst-case Energie-Regret**, weil GEMM/STRIDED_GEMM bei kleinen Größen andere Energiegewinner besitzen. Workloadname allein genügt daher nicht; die Größen- oder Shapeinformation ist entscheidend.

## 6. Neuer harter Befund: Konflikt bedeutet tatsächlich kein nahes gemeinsames Optimum

Für jede der 51 Zellen wurden alle 18 gemessenen Konfigurationen betrachtet: zwei GPUs plus Intel- und AMD-Thread-Sweeps.

- Bei einer 1-%-Grenze existiert in 26/51 Zellen ein gemeinsames Näherungsoptimum.
- Bei 2 %: 26/51.
- Bei 5 %: 27/51.
- Bei 10 %: 29/51.
- Bei 20 %: 32/51.

Der überraschende exakte Zusammenhang bei 5 % lautet:

- **24/24 robuste Konfliktzellen:** keine Konfiguration liegt zugleich innerhalb von 5 % beider Optima.
- **27/27 Nicht-Konfliktzellen:** mindestens eine solche Konfiguration existiert.

Der Zielkonflikt ist damit in diesen Daten nicht bloß ein Artefakt verschiedener Punktgewinner. Er bleibt auch unter einer praktisch großzügigen 5-%-Definition hart.

## 7. Paretofronten: Der Messraum ist groß, die Entscheidungsfront klein

Über alle 918 Konfigurationen:

- mittlere strikte Frontgröße: **1.549** von 18,
- genau ein Paretopunkt in **26/51** Zellen,
- genau zwei in **23/51**,
- CPU auf der Front in nur **6/51** Zellen,
- ausschließlich GPUs auf der Front in **45/51** Zellen,
- im Mittel **91.39 %** der Konfigurationen strikt dominiert.

Das spricht für ein zweistufiges Paperargument: Der vollständige Thread-Sweep ist für die methodische Absicherung wichtig, aber nach der Messung reduziert sich der relevante Entscheidungsraum drastisch.

## 8. CPU-Threadzahl: Near-free savings und teure letzte Prozente

Die laufzeit- und energieoptimale Threadzahl unterscheidet sich in **65 von 102** CPU-Plattform-Zellen.

### Teuerste letzte Laufzeitverbesserungen bei höchstens 5 % Gewinn

| workload     |      size | platform   | runtime_opt_cfg   | energy_opt_cfg   |   runtime_gain_pct |   energy_premium_pct |   marginal_cost_ratio |
|:-------------|----------:|:-----------|:------------------|:-----------------|-------------------:|---------------------:|----------------------:|
| REDUCTION    | 256000000 | INTEL      | 8T                | 4T               |              0.244 |               39.449 |               161.558 |
| REDUCTION    | 128000000 | INTEL      | 8T                | 4T               |              0.832 |               36.912 |                44.379 |
| REDUCTION    |  64000000 | INTEL      | 8T                | 4T               |              1.344 |               36.804 |                27.382 |
| REDUCTION    |  32000000 | INTEL      | 8T                | 4T               |              3.098 |               30.042 |                 9.696 |
| GEMM         |       128 | INTEL      | 10T               | 4T               |              4.998 |               18.343 |                 3.67  |
| GEMM         |      8192 | AMD        | 32T               | 64T              |              4.977 |               13.195 |                 2.651 |
| CONV2D       |         6 | AMD        | 32T               | 64T              |              1.143 |               12.243 |                10.712 |
| STRIDED_GEMM |      8192 | AMD        | 32T               | 64T              |              4.617 |               11.675 |                 2.529 |
| CONV2D       |         4 | AMD        | 32T               | 64T              |              3.091 |                9.318 |                 3.014 |
| CONV2D       |         3 | INTEL      | 20T               | 10T              |              0.059 |                8.994 |               152.499 |

Besonders stark ist eine konsistente Intel-REDUCTION-Serie:

- 32M: 30,04 % Energie für 3,10 % Laufzeit,
- 64M: 36,80 % für 1,34 %,
- 128M: 36,91 % für 0,83 %,
- 256M: 39,45 % für 0,244 %.

Die Grenzkosten steigen mit der Größe massiv; bei 256M werden rund **161,6 % relative Energieerhöhung pro 1 % relativer Laufzeitverbesserung** bezahlt.

### Höchste Energieersparnis bei höchstens 1 % Laufzeitverlust

| workload   |      size | platform   | chosen_cfg   | runtime_opt_cfg   |   runtime_penalty_pct |   energy_saving_pct |
|:-----------|----------:|:-----------|:-------------|:------------------|----------------------:|--------------------:|
| REDUCTION  | 256000000 | INTEL      | 4T           | 8T                |                 0.244 |              28.289 |
| STREAM     | 256000000 | INTEL      | 4T           | 8T                |                 0.632 |              27.271 |
| REDUCTION  | 128000000 | INTEL      | 4T           | 8T                |                 0.832 |              26.96  |
| AXPY       | 256000000 | INTEL      | 4T           | 8T                |                 0.081 |              26.337 |
| STREAM     |   8000000 | INTEL      | 4T           | 8T                |                 0.876 |              26.332 |
| AXPY       |   8000000 | INTEL      | 4T           | 8T                |                 0.761 |              25.877 |
| AXPY       |  16000000 | INTEL      | 4T           | 8T                |                 0.339 |              24.539 |
| STREAM     |   4000000 | INTEL      | 8T           | 10T               |                 0.223 |               9.127 |
| CONV2D     |         3 | INTEL      | 10T          | 20T               |                 0.059 |               8.252 |
| STREAM     |   2000000 | AMD        | 16T          | 20T               |                 0.42  |               2.914 |

Das inverse Framing ist deploymentnäher: Bei Intel REDUCTION 256M lassen sich **28,29 % Energie** bei nur **0,244 % Laufzeitverlust** sparen. Bei mehreren AXPY- und STREAM-Fällen liegen 24–27 % Einsparung unter 1 % Laufzeitverlust.

### „Immer maximale Threads“ ist keine sichere Baseline

Über 102 CPU-Plattform-Zellen verursacht maximale Threadzahl:

- median 7,47 % Laufzeit-Regret,
- median 38,52 % Energie-Regret,
- median 63,06 % EDP-Regret,
- maximal 458,20 % Energie-Regret,
- maximal 2108,73 % EDP-Regret.

## 9. Nichtmonotone Gewinner-Topologie

Die gemeldeten Sequenzen werden bestätigt:

- AXPY Laufzeit: 5060 Ti → AMD → 3090,
- STREAM Laufzeit: 5060 Ti → AMD → 3090,
- REDUCTION Laufzeit: 3090 → 5060 Ti → 3090,
- GEMM Energie: Intel → 5060 Ti → 3090.

REDUCTION enthält damit eine echte Rückkehr eines früheren Gewinners. Eine einzelne monotone Größenschwelle kann diese Sequenz nicht darstellen.

Conv2D darf nicht als geordnete Größenachse interpretiert werden. Dennoch ist die Shape-Abhängigkeit extrem: Der Energiegewinner alterniert über die sechs Shapes vollständig zwischen RTX 3090 und RTX 5060 Ti, während Laufzeit und EDP bei allen sechs Shapes an die RTX 3090 gehen.

## 10. Rangliste der stärksten Befunde

| finding                                                  | statement                                                                                                                                                 |   total |
|:---------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------|--------:|
| Large-regime objective split                             | All 15 AXPY/STREAM/REDUCTION cells from 16M to 256M choose RTX 3090 for runtime and RTX 5060 Ti for energy.                                               |      38 |
| Static GPU policy is costly in opposite objectives       | Always 3090: median 72.382% energy regret; always 5060 Ti: median 110.564% runtime regret.                                                                |      38 |
| Power overwhelms race-to-idle                            | In the 15-cell regime the 3090 is 2.122x faster but draws 3.696x median board power and uses 1.738x median energy.                                        |      38 |
| EDP is strongly speed-oriented here                      | EDP point winner equals runtime point winner in 48/51 cells; in the large regime its 3090 advantage ranges from about 3% for REDUCTION to 23% for STREAM. |      38 |
| Hard conflict equals no 5%-joint optimum                 | Exactly the 24 robust conflict cells lack any measured configuration within 5% of both optima; all 27 non-conflict cells have one.                        |      37 |
| Winner topology is non-monotone                          | REDUCTION runtime returns 3090→5060 Ti→3090; Conv2D energy alternates 3090/5060 Ti across all six shapes.                                                 |      37 |
| The last CPU performance fraction is extremely expensive | Intel REDUCTION 32M–256M spends 30.0–39.4% more energy for only 3.10% down to 0.244% runtime improvement.                                                 |      35 |
| Workload name alone is insufficient for energy placement | A workload-only average policy still reaches 610% worst-case all-platform energy regret because winner sequences change with size.                        |      35 |
| Most measured configurations are decision-irrelevant     | 91.4% of the 18 configurations per cell are strictly Pareto dominated; the strict frontier averages 1.55 points.                                          |      33 |
| Maximum CPU threads is a poor default                    | Across 102 CPU workload-size-platform cells, max threads has 38.5% median energy regret and up to 458%; EDP regret reaches 2109%.                         |      33 |

## 11. Literaturabgrenzung

| Nächste Arbeit | Was sie bereits zeigt | Was diese Kampagne zusätzlich liefert | Risiko |
|---|---|---|---|
| HEP GPU energy efficiency, arXiv:2604.27523 | Höchster GPU-Durchsatz ist nicht höchste Energieeffizienz; zehn GPUs, ein HLT1-Workload | sechs harmonisierte Kernelfamilien, Größen-/Shapeachsen, CPUs, Threadwahl, Sessionrobustheit und Regret | Existenzclaim ist besetzt |
| Wattlytics, arXiv:2604.08182 | Workloadabhängige GPU-/TCO-Entscheidungen für HPC-Anwendungen | direkte Sessionmessungen, kontrollierte Kernelregime, harte Zielkonflikte statt TCO-Modellfokus | „workload-dependent selection“ nicht als Neuheit claimen |
| Tchakoute et al., arXiv:2505.03398 / IEEE Access | sechs Kernel, Intel/AMD/NVIDIA, Energie/Laufzeit/EDP unter DVFS, Governor und Power Cap | feste Stock-Zustände, Placement über Plattform × Größe × Thread, praktische Ties und Policy-Regret | nächster Evaluationskonkurrent |
| SuperMUC-NG Phase 2, arXiv:2606.23265 | größenabhängige CPU/GPU-Energiecrossovers in realen HPC-Anwendungen | consumer GPUs, semantisch harmonisierte Kernel, objective-dependent map, 5×10-Design | Crossovers selbst sind nicht neu |
| Watt Counts, arXiv:2604.09048 | hardwareabhängige Energieentscheidungen über 50 LLMs und zehn GPUs | allgemeiner kontrollierter Kernelraum plus CPU-/Threadebene | nicht mit „largest benchmark“ konkurrieren |

Der sicherste Novelty-Satz lautet sinngemäß:

> *Prior work establishes that throughput-optimal and energy-optimal hardware can differ. We contribute a session-robust map of when this divergence occurs across harmonized kernel families, problem scales, platforms, and CPU thread configurations, and quantify the regret of ignoring that context.*

## 12. Reviewer-Angriffstest

### Angriff 1: „CPU und GPU messen unterschiedliche Energiedomänen“

**Berechtigt.** CPU nutzt Package-RAPL, GPU NVML-Boardenergie inklusive VRAM. Deshalb dürfen CPU-vs-GPU-Zahlen nicht als identische Systemenergiebilanz verkauft werden. Der Hauptbefund 23/24 und das 15/15-Regime stehen jedoch vollständig im symmetrischen GPU-only-Raum.

### Angriff 2: „GPU-resident ignoriert PCIe und Allokation“

**Berechtigt.** Der Claim gilt für wiederholte, resident ausgeführte Kernel. Keine Transfer-inclusive Offloading- oder One-shot-Placement-Behauptung.

### Angriff 3: „Nur zwei GPUs und zwei CPUs“

Reichweitengrenze, aber kein interner Validitätsfehler. Die Arbeit kartiert die untersuchte Kampagne; sie isoliert weder Generation noch Architektur kausal.

### Angriff 4: „Die 15 Zellen sind nicht unabhängig“

Korrekt. Sie werden als strukturell verwandte Zellen beschrieben. Die unabhängigen statistischen Einheiten sind die Sessions innerhalb einer Zelle.

### Angriff 5: „Post-selection bei optimalen CPU-Threads“

Die native-best Ratio-Intervalle sind deskriptiv. Für einen konfirmatorischen Claim müsste die Auswahl in Train-Sessions erfolgen und in Holdout-Sessions bewertet werden. Die vorhandene 5/5-Richtungsstabilität mildert, ersetzt aber nicht vollständig eine verschachtelte Auswahl.

### Angriff 6: „Power-Limit der RTX 3090 ist konfundiert“

Das Stock-Power-Limit ist Teil des gemessenen Plattformzustands. Keine reine Architektur- oder Generationserklärung behaupten. Dass die Gewinnerfolge innerhalb desselben GPU-Paars über Workloads und Shapes wechseln, verhindert eine monokausale Erklärung durch den konstanten Plattformzustand.

## 13. Paperempfehlung

### Hauptgeschichte

**Objective-dependent placement map plus policy regret.**

### Contributions

1. Auditierte, semantisch harmonisierte 4-Plattform-/6-Workload-Messbasis mit Sessiondesign.
2. Praktisch und statistisch getrennte Laufzeit-/Energie-/EDP-Placementkarte über 51 Zellen.
3. Quantifizierung harter Zielkonflikte und statischer Policy-Regrets.
4. Zweite Entscheidungsebene CPU-Threadzahl mit near-free savings und expensive-last-percent-Fällen.
5. Reproduzierbare Robustheits-, Pareto- und Energiedomänenanalyse.

### Ergebnisreihenfolge

1. vollständige Placement Map,
2. 15/15-Großregime,
3. Power-Zeit-Zerlegung,
4. Policy-Regret und erforderlicher Kontext,
5. CPU-Thread-Trade-offs,
6. EDP, Pareto und negative Heuristikergebnisse.

### Abstract-Zahlen

- 24/51 harte Konflikte,
- 23/24 davon GPU-only,
- 15/15 großes AXPY/STREAM/REDUCTION-Regime,
- 2,12× schneller versus 42,5 % weniger Energie,
- 72,4 % beziehungsweise 110,6 % medianes statisches GPU-Regret,
- optional: 28,3 % CPU-Energieeinsparung für 0,24 % Laufzeitverlust.

## 14. Grenzen dieser Auswertungsstufe

Vollständig reproduziert wurden die vorgegebenen elf Claims sowie mehrere explorative Cross-Analysen aus den aktuellen Zusammenfassungs- und Sessiondateien. Noch nicht als konfirmatorisch abgeschlossen gelten:

- verschachteltes Leave-one-session-out mit erneuter CPU-Threadauswahl innerhalb jedes Trainingsfolds,
- hierarchische Bootstrap-CIs für alle neu erzeugten Cross-Workload-Metriken,
- Intel Package-vs-Package+DRAM-Sensitivität, sofern die nötigen DRAM-Werte in allen Workloads verfügbar sind,
- vollständige Shape-Feature-Regression für Conv2D,
- formale Leave-one-workload-out-Prädiktionsmodelle,
- externe Messgerätevalidierung.

Diese Punkte sollten den Hauptclaim nicht blockieren, aber sauber zwischen Haupttext, Appendix und Future Work verteilt werden.
