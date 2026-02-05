# Funzioni di Perdita e Inferenza

## Cross-Entropy Loss

La **Cross-Entropy Loss** (entropia incrociata) è una funzione di perdita utilizzata quando la variabile target è **categorica**, in particolare nei problemi di classificazione multiclasse. Il suo impiego è strettamente connesso a modelli probabilistici che, per ogni osservazione, stimano una distribuzione di probabilità sulle classi possibili.

Si consideri un dataset formato da $n$ osservazioni e $m$ categorie. Ogni osservazione appartiene a una sola categoria ed è rappresentata tramite un vettore **one-hot** $b(i) \in \{0,1\}^m$, tale che $b_j(i)=1$ se l’osservazione $i$ appartiene alla classe $j$ e $b_j(i)=0$ altrimenti.  

Indichiamo con $p_j(i)$ la probabilità che l’osservazione $i$ appartenga alla classe $j$. Per ogni $i$, queste quantità definiscono una distribuzione discreta, quindi soddisfano i vincoli
$$
p_j(i) \ge 0, 
\qquad 
\sum_{j=1}^m p_j(i) = 1.
$$

Grazie alla codifica one-hot, la probabilità dell’osservazione $i$ può essere scritta in forma compatta come
$$
P(Y(i)=j) = p_j(i) = \prod_{k=1}^m p_k(i)^{b_k(i)}.
$$
Il prodotto seleziona automaticamente il termine corrispondente alla classe osservata, poiché tutti gli altri esponenti sono nulli.

La **log-verosimiglianza** dell’intero campione si ottiene sommando i contributi delle singole osservazioni e assume la forma
$$
\ell = \sum_{i=1}^n \sum_{j=1}^m b_j(i)\,\log p_j(i).
$$
Massimizzare questa quantità equivale a rendere massima la probabilità assegnata dal modello alle classi effettivamente osservate.

### Caso senza dipendenza da $x$

Se le probabilità non dipendono dall’input, cioè $p_j(i)=P_j$ è costante per tutte le osservazioni, il problema si riduce alla stima dei parametri di una distribuzione categorica. In questo caso la massimizzazione della log-verosimiglianza porta alla stima
$$
\hat P_j = \frac{O_j}{n},
$$
dove $O_j$ indica il numero di osservazioni appartenenti alla classe $j$. La stima coincide quindi con la frequenza empirica, risultato coerente con l’interpretazione probabilistica del modello.

### Caso con dipendenza da $x$

Quando la probabilità della classe dipende dall’input $x(i)$, si introduce un modello parametrico del tipo
$$
p_j(i) = \pi_j(x(i); \alpha,\beta,\gamma),
$$
che associa a ciascuna osservazione una distribuzione di probabilità sulle classi. Un esempio fondamentale è fornito dalla **softmax**, definita come
$$
\pi_j(x) = \frac{\exp(\eta_j(x))}{\sum_{k=1}^m \exp(\eta_k(x))},
\qquad
\eta_j(x) = w_j^\top x + c_j.
$$
La softmax garantisce automaticamente che le probabilità siano non negative e sommino a uno. In questo contesto, la Cross-Entropy Loss coincide con il negativo della log-verosimiglianza e la stima dei parametri avviene numericamente, tipicamente tramite algoritmi di ottimizzazione iterativa come la discesa del gradiente.


### Cross-Entropy come funzione di perdita

Poiché l’addestramento dei modelli di classificazione avviene tramite **minimizzazione** di una funzione obiettivo, la log-verosimiglianza viene riscritta cambiando segno e normalizzando rispetto al numero di osservazioni. Si ottiene così la **cross-entropy loss**, definita come
$$
\text{loss}_{\text{CE}}(\alpha,\beta,\gamma)
= -\frac{1}{n}\sum_{i=1}^n \sum_{j=1}^m b_j(i)\,\log \pi_j(x(i);\alpha,\beta,\gamma).
$$
Questa quantità rappresenta il negativo della log-verosimiglianza media e misura quanto la distribuzione di probabilità predetta dal modello si discosta dalla distribuzione vera, che nel caso di classificazione supervisionata è codificata tramite vettori one-hot.

La stessa espressione può essere interpretata in termini di **entropia incrociata** tra la distribuzione target $b(i)$ e la distribuzione predetta $\pi(x(i))$. Per una singola osservazione vale
$$
H(b,\pi) = -\sum_{j=1}^m b_j \log \pi_j,
$$
e la funzione di perdita complessiva non è altro che la media di questa quantità sul campione:
$$
\text{loss}_{\text{CE}} = \frac{1}{n}\sum_{i=1}^n H\bigl(b(i),\pi(x(i))\bigr).
$$

Dal punto di vista interpretativo, minimizzare la cross-entropy equivale a rendere massima la probabilità assegnata dal modello alla classe corretta. L’uso del logaritmo fa sì che predizioni errate ma molto “sicure”, cioè con probabilità alta assegnata alla classe sbagliata, vengano penalizzate in modo particolarmente severo.

Nella pratica computazionale è necessario prestare attenzione alla stabilità numerica. In genere si evita il problema di $\log 0$ introducendo un piccolo termine $\varepsilon$ nelle probabilità oppure lavorando direttamente con formulazioni numericamente stabili della softmax. Inoltre, in presenza di dataset sbilanciati o di rumore nelle etichette, si possono introdurre tecniche come il *label smoothing* o pesi di classe, che modificano la loss per rendere l’addestramento più robusto.

## Mean Squared Error Loss (MSE)

La **Mean Squared Error (MSE)** è una funzione di perdita tipicamente utilizzata quando la variabile risposta è quantitativa e si assume un modello **gaussiano** per gli errori. In questo contesto si ipotizza che ciascuna osservazione segua una distribuzione normale
$$
Y(i) \sim \mathcal{N}(\mu(i), \sigma^2(i)),
$$
dove $\mu(i)$ rappresenta la media condizionata, eventualmente dipendente dagli input, mentre $\sigma^2(i)$ indica la varianza.

Sotto questa ipotesi, la log-verosimiglianza del campione può essere scritta come
$$
\ell
= \sum_{i=1}^n \left[
-\log \sigma(i)
- \log \sqrt{2\pi}
- \frac{(Y(i)-\mu(i))^2}{2\sigma^2(i)}
\right].
$$
Questa espressione mette in evidenza come lo scarto quadratico tra osservazioni e media giochi un ruolo centrale nel modello gaussiano.

### Parametri costanti

Nel caso più semplice si assume che $\mu(i)=\mu$ e $\sigma(i)=\sigma$ per ogni osservazione. La massimizzazione della log-verosimiglianza conduce alle stime classiche: la media campionaria come stimatore di $\mu$ e la varianza campionaria con denominatore $n$ come stimatore di $\sigma^2$ nel senso della massima verosimiglianza. In questo scenario la MSE coincide con la varianza empirica attorno alla media stimata.

### Caso omoschedastico (varianza costante)

Un caso fondamentale in regressione è quello in cui la media dipende dagli input $x(i)$, mentre la varianza è costante, cioè $\sigma^2(i)=\sigma^2$. Indicando con $\mu(x(i);\alpha,\beta,\gamma)$ un modello parametrico per la media, la parte della log-verosimiglianza che dipende da tali parametri è, a meno di costanti additive e moltiplicative, proporzionale alla somma dei quadrati dei residui. Ne segue che massimizzare la log-verosimiglianza equivale a minimizzare la **MSE**:
$$
\text{loss}_{\text{MSE}}(\alpha,\beta,\gamma)
= \frac{1}{n}\sum_{i=1}^n \bigl(Y(i)-\mu(x(i);\alpha,\beta,\gamma)\bigr)^2.
$$

Una volta stimata la media, la stima di massima verosimiglianza della deviazione standard è data da
$$
\hat{\sigma}
= \sqrt{\text{loss}_{\text{MSE}}(\alpha,\beta,\gamma)},
$$
mentre la corrispondente stima della varianza risulta
$$
\hat{\sigma}^2_{\text{MLE}}
= \frac{1}{n}\sum_{i=1}^n \bigl(Y(i)-\hat{\mu}(i)\bigr)^2.
$$
Dal punto di vista statistico questo stimatore è distorto; lo stimatore **non distorto** della varianza utilizza invece il denominatore $n-p$, dove $p$ è il numero di parametri stimati nel modello per la media.

### Caso eteroschedastico (dipendenza completa da $x$)

Nel caso più generale anche la varianza dipende dagli input, cioè $\sigma^2(i)=\sigma^2(x(i))$. In questa situazione la funzione obiettivo coerente con il modello gaussiano è il negativo della log-verosimiglianza:
$$
-\ell(\theta)
= \frac{1}{2}\sum_{i=1}^n \left[
\log\bigl(2\pi \sigma^2(i)\bigr)
+ \frac{(Y(i)-\mu(i))^2}{\sigma^2(i)}
\right].
$$
Questo criterio corrisponde a una forma di **weighted least squares**, in cui ciascun residuo è pesato con l’inverso della sua varianza. La stima congiunta dei parametri della media e della varianza non ammette in generale una soluzione in forma chiusa e richiede l’uso di metodi di ottimizzazione iterativi.

## Regressione Lineare Semplice

Nel modello di **regressione lineare semplice** si assume che le variabili indipendenti $x_i$ siano deterministiche, mentre le variabili dipendenti $Y_i$ siano casuali. L’ipotesi di base è che ciascuna osservazione segua un modello gaussiano con media lineare e varianza costante:
$$
Y_i \sim \mathcal{N}(\mu_i, \sigma^2),
\qquad
\mu_i = \beta_0 + \beta_1 x_i.
$$
In questa formulazione $\beta_0$ rappresenta l’intercetta del modello, mentre $\beta_1$ quantifica l’effetto medio di una variazione unitaria di $x$ sulla risposta $Y$.

Assumendo indipendenza delle osservazioni e omoschedasticità, la massimizzazione della log-verosimiglianza gaussiana è equivalente alla minimizzazione della **Mean Squared Error**, definita come
$$
\text{loss}_{\text{MSE}}(\beta_0, \beta_1)
= \frac{1}{n} \sum_{i=1}^n \bigl(Y_i - \beta_0 - \beta_1 x_i\bigr)^2.
$$
La stima dei parametri si riduce quindi a un problema di minimi quadrati.

La soluzione analitica di questo problema fornisce le stime di massima verosimiglianza dei coefficienti, che coincidono con le stime dei minimi quadrati ordinari:
$$
\hat{\beta}_1
= \frac{\sum_{i=1}^n x_i Y_i - n \bar{x}\,\bar{Y}}
{\sum_{i=1}^n x_i^2 - n \bar{x}^2},
\qquad
\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{x},
$$
dove $\bar{x}$ e $\bar{Y}$ indicano le medie campionarie delle variabili indipendente e dipendente. La stima di $\beta_1$ può essere interpretata come il rapporto tra la covarianza campionaria tra $x$ e $Y$ e la varianza campionaria di $x$.

Una volta stimati i coefficienti, si procede alla stima della varianza degli errori. Lo stimatore di massima verosimiglianza utilizza il denominatore $n$ ed è quindi distorto. Lo stimatore **non distorto** della varianza degli errori è invece
$$
S_e^2
= \frac{1}{n - 2} \sum_{i=1}^n \bigl(Y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i\bigr)^2,
$$
dove il termine $n-2$ rappresenta i gradi di libertà residui dopo la stima dei due parametri $\beta_0$ e $\beta_1$.

## Teorema di Cochran

Il **teorema di Cochran** fornisce la base teorica per comprendere come la **variabilità totale dei dati** possa essere scomposta in modo rigoroso nei modelli lineari sotto ipotesi gaussiane. L’idea centrale è che, quando le osservazioni seguono una distribuzione normale con **varianza costante** e i parametri vengono stimati all’interno di un modello lineare, la parte dei dati spiegata dal modello e la parte residua possono essere trattate come componenti **indipendenti**, ciascuna con una struttura probabilistica ben definita. Questo risultato giustifica formalmente la scomposizione della varianza utilizzata in regressione lineare, **ANOVA** e **test F**.

Si consideri un vettore aleatorio $X=(X_1,\dots,X_n)\in\mathbb{R}^n$ con componenti indipendenti tali che
$$
X_i \sim \mathcal{N}(\mu_i,\sigma^2),
$$
e si assuma che il vettore delle medie $\mu=(\mu_1,\dots,\mu_n)$ appartenga a un sottospazio vettoriale $V\subset\mathbb{R}^n$ di dimensione $k$. In questo contesto, la stima di massima verosimiglianza di $\mu$ si ottiene come **proiezione ortogonale** del vettore osservato $X$ sul sottospazio $V$. Indicando con $\pi_V(X)$ tale proiezione, vale
$$
\hat{\mu}=\pi_V(X).
$$
Dal punto di vista geometrico, questa stima è quella che minimizza la distanza euclidea tra $X$ e l’insieme dei vettori ammissibili per $\mu$, coerentemente con il criterio dei minimi quadrati.

La componente di $X$ non spiegata dal modello è il residuo $X-\pi_V(X)$. La sua norma al quadrato
$$
W=\|X-\pi_V(X)\|^2
$$
rappresenta la **somma dei quadrati residui**. Il teorema di Cochran afferma che la quantità $W$ è **indipendente** dalla componente stimata $\pi_V(X)$ e che, normalizzata per la varianza comune $\sigma^2$, segue una distribuzione chi-quadro con $n-k$ gradi di libertà:
$$
\frac{W}{\sigma^2}\sim\chi^2(n-k).
$$

L’indipendenza tra la parte spiegata dal modello e la parte residua, insieme alla distribuzione chi-quadro della somma dei quadrati residui, costituisce il fondamento teorico della **scomposizione della varianza**. È proprio questo risultato che rende possibile confrontare in modo corretto la variabilità spiegata dal modello con quella attribuibile al rumore, utilizzando distribuzioni note e strumenti di inferenza statistica affidabili nei modelli di regressione lineare.