# Analisi della varianza

L’analisi della varianza, o ANOVA, è una tecnica statistica utilizzata per studiare la dipendenza di una variabile numerica $Y$ da una o più variabili esplicative di tipo **categorico**. L’idea di fondo è confrontare le medie di $Y$ tra diversi gruppi e stabilire se le differenze osservate possano essere attribuite al caso oppure siano statisticamente significative.

A differenza della regressione lineare classica, in cui le covariate sono numeriche, nell’ANOVA i regressori assumono un numero finito di modalità, come gruppi, livelli o trattamenti. Il modello statistico sottostante assume che la variabile risposta sia affetta da **rumore additivo gaussiano omoschedastico**, cioè errori indipendenti, normalmente distribuiti e con varianza costante.

A seconda del numero di fattori considerati, si distinguono diverse configurazioni. Nell’ANOVA a una via è presente un solo fattore categorico; nell’ANOVA a due vie entrano in gioco due fattori; infine, nell’ANOVA a due vie con repliche si hanno più osservazioni per ciascuna combinazione dei livelli dei fattori.

## ANOVA a una via

Nel caso più semplice si considera un unico fattore categoriale con $m$ livelli. Ogni osservazione $Y_{ij}$ appartiene al gruppo $i$ ed è la $j$-esima replica all’interno di quel gruppo. Il modello è

$$
Y_{ij} = \mu_i + \varepsilon_{ij}, \qquad \varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2),
$$

dove $\mu_i$ rappresenta la media del gruppo $i$ e $\sigma^2$ è la varianza comune a tutti i gruppi. L’indice $i = 1,\dots,m$ identifica i gruppi, mentre $j = 1,\dots,n_i$ indica le osservazioni disponibili nel gruppo $i$.

L’ipotesi nulla del test ANOVA a una via afferma che tutte le medie di gruppo coincidano,

$$
H_0:\ \mu_1 = \mu_2 = \dots = \mu_m,
$$

mentre l’alternativa sostiene che almeno una media sia diversa. In termini interpretativi, il test verifica se la variabilità osservata tra le medie dei gruppi sia sufficientemente grande rispetto alla variabilità interna ai gruppi. Se così non fosse, le differenze tra gruppi possono essere spiegate come semplici fluttuazioni casuali; in caso contrario, si conclude che la variabile risposta $Y$ dipende in modo significativo dal gruppo di appartenenza.

## Stima dei parametri

Nel modello ANOVA a una via, per ciascun gruppo $i$ si stimano i parametri tramite le statistiche campionarie. In particolare, la media campionaria $\bar Y_i$ fornisce una stima naturale della media di gruppo $\mu_i$, mentre la varianza campionaria $S_i^2$ stima la varianza comune $\sigma^2$.

Sotto le ipotesi del modello, la media campionaria di ciascun gruppo ha distribuzione normale,

$$
\bar Y_i \sim \mathcal{N}\!\left( \mu_i, \frac{\sigma^2}{n_i} \right),
$$

poiché è la media di $n_i$ osservazioni indipendenti gaussiane con varianza $\sigma^2$. Le varianze campionarie $S_i^2$ sono invece stimatori non distorti della stessa varianza $\sigma^2$, ma basati solo sulle osservazioni del singolo gruppo.

Per sfruttare l’ipotesi di varianza comune, si introduce una stima globale ottenuta aggregando l’informazione di tutti i gruppi. A questo scopo si definisce la **devianza within** (o devianza interna ai gruppi),

$$
\mathrm{SS}_W = \sum_{i=1}^m \sum_{j=1}^{n_i} \bigl(Y_{ij} - \bar Y_i\bigr)^2,
$$

che misura la variabilità delle osservazioni attorno alle rispettive medie di gruppo. Dividendo questa quantità per i gradi di libertà residui $N - m$, con $N = \sum_{i=1}^m n_i$, si ottiene la stima pooled della varianza,

$$
S_p^2 = \frac{\mathrm{SS}_W}{N - m}.
$$

Sotto le ipotesi del modello, questa statistica ha una distribuzione chi-quadrato scalata,

$$
\frac{(N - m) S_p^2}{\sigma^2} \sim \chi^2_{N - m},
$$

risultato che è fondamentale per la costruzione del test ANOVA.

Accanto alla devianza interna, si considera la **devianza totale**,

$$
\mathrm{SS}_Y = \sum_{i=1}^m \sum_{j=1}^{n_i} \bigl(Y_{ij} - \bar Y\bigr)^2,
$$

dove $\bar Y$ è la media complessiva di tutte le osservazioni. Questa quantità misura la variabilità totale di $Y$ senza tenere conto della struttura a gruppi.

La parte di variabilità attribuibile alle differenze tra i gruppi è descritta dalla **devianza between**,

$$
\mathrm{SS}_B = \sum_{i=1}^m n_i \bigl(\bar Y_i - \bar Y\bigr)^2,
$$

che confronta le medie di gruppo con la media globale, pesandole per le rispettive numerosità.

Queste tre devianze sono legate dalla classica **identità delle devianze**,

$$
\mathrm{SS}_Y = \mathrm{SS}_B + \mathrm{SS}_W,
$$

che formalizza l’idea centrale dell’ANOVA: la variabilità totale della risposta può essere scomposta in una componente dovuta alle differenze tra gruppi e una componente dovuta alla variabilità interna ai gruppi.

## Il test ANOVA

Per stabilire se le medie di gruppo siano tutte uguali si formalizza il problema come un test di ipotesi. L’ipotesi nulla afferma che non vi siano differenze sistematiche tra i gruppi,

$$
H_0: \mu_1 = \mu_2 = \dots = \mu_m,
$$

mentre l’ipotesi alternativa sostiene che almeno una delle medie differisca dalle altre.

L’idea centrale del test ANOVA è confrontare due stime della varianza $\sigma^2$: una basata sulla variabilità **tra** i gruppi e una basata sulla variabilità **interna** ai gruppi. La prima è ottenuta dividendo la devianza between per i suoi gradi di libertà,

$$
S_B^2 = \frac{\mathrm{SS}_B}{m - 1},
$$

mentre la seconda coincide con la stima pooled già introdotta,

$$
S_p^2 = \frac{\mathrm{SS}_W}{N - m}.
$$

La statistica test è quindi il loro rapporto,

$$
F = \frac{S_B^2}{S_p^2}
  = \frac{\mathrm{SS}_B / (m - 1)}{\mathrm{SS}_W / (N - m)}.
$$

Sotto l’ipotesi nulla, entrambe le quantità al numeratore e al denominatore stimano la stessa varianza $\sigma^2$, e il loro rapporto segue una distribuzione di Fisher–Snedecor con $m-1$ e $N-m$ gradi di libertà,

$$
F \sim F(m - 1,\ N - m).
$$

Il criterio decisionale consiste nel confrontare il valore osservato della statistica con il quantile superiore della distribuzione $F$. Se il valore di $F$ risulta sufficientemente grande, ossia maggiore del quantile critico $F_{\alpha}(m-1, N-m)$ per un livello di significatività $\alpha$, si rifiuta l’ipotesi nulla. In tal caso si conclude che la variabilità tra le medie di gruppo è troppo elevata per essere spiegata dal solo rumore interno, e quindi che almeno un gruppo presenta una media diversa dalle altre.

## Aspetti operativi

Dal punto di vista pratico, i dati per un’analisi ANOVA possono essere organizzati secondo due strutture principali. Nel **formato wide** ogni gruppo è rappresentato da una colonna distinta, ciascuna contenente le osservazioni relative a quel livello del fattore. Nel **formato long** (o stacked), invece, tutte le osservazioni sono raccolte in un’unica colonna numerica, affiancata da una colonna categoriale che indica il gruppo di appartenenza. Quest’ultimo formato è in genere preferito dai software statistici e risulta più flessibile quando si passa a modelli con più fattori.

Una volta stimato il modello e svolto il test ANOVA, è fondamentale verificare a posteriori la validità delle ipotesi su cui il modello si basa. Questo controllo viene effettuato analizzando i **residui**, ossia le differenze tra le osservazioni e le corrispondenti medie di gruppo. In particolare, ci si aspetta che i residui siano approssimativamente normali, che mostrino una variabilità simile nei diversi gruppi (omoschedasticità) e che non presentino valori anomali particolarmente influenti.

Qualora queste condizioni risultino violate, una strategia comune consiste nell’applicare una trasformazione alla variabile risposta $Y$, come il logaritmo o la radice quadrata. Tali trasformazioni possono ridurre l’asimmetria, stabilizzare la varianza e rendere più plausibili le assunzioni del modello.

Infine, se il test ANOVA non risulta significativo, non vi è evidenza statistica di differenze tra le medie di gruppo. In questo caso è ragionevole trattare l’intero insieme di dati come un unico campione proveniente da una stessa distribuzione gaussiana, ignorando il fattore categoriale senza perdita informativa rilevante.
