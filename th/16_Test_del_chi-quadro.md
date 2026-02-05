# Test del Chi-quadro

Il **test del Chi-quadro** è una famiglia di test statistici utilizzati per valutare se i dati osservati sono compatibili con una distribuzione teorica prefissata. L’idea di fondo è confrontare ciò che si osserva nel campione con ciò che ci si aspetterebbe se un certo modello probabilistico fosse vero, misurando la discrepanza tra osservato e atteso.

## Test del Chi-quadro elementare

Si consideri un campione $X_1,\dots,X_n$ di variabili aleatorie indipendenti, generate da una distribuzione incognita $\varphi$, che può assumere $K$ valori distinti. L’obiettivo è verificare se $\varphi$ coincide con una distribuzione teorica nota $\varphi_0$.

L’ipotesi nulla afferma che i dati seguano esattamente la distribuzione teorica,
$$
H_0:\ \varphi = \varphi_0,
$$
mentre l’ipotesi alternativa sostiene che la distribuzione reale sia diversa,
$$
H_1:\ \varphi \neq \varphi_0.
$$

L’idea operativa del test è la seguente. Per ciascuno dei $K$ possibili valori, si conta quante osservazioni del campione assumono quel valore: questo numero è la **frequenza osservata** $O_j$ (e dipende solo dai dati).

In parallelo, assumendo vera l’ipotesi nulla $H_0$, si calcola quante osservazioni ci si aspetterebbe in quella stessa categoria sulla base della distribuzione teorica. Se sotto $H_0$ la probabilità della $j$-esima categoria è $p_j$, allora la **frequenza attesa** è
$$
A_j = n \cdot p_j,
$$
dove $n$ è la numerosità campionaria.

Il confronto tra $O_j$ e $A_j$ è il cuore del test. Se $H_0$ è compatibile con i dati, le frequenze osservate dovrebbero essere vicine a quelle attese, salvo fluttuazioni casuali. Se invece il modello teorico non descrive bene i dati, emergono differenze sistematiche tra osservato e atteso in una o più categorie.

## Statistica di Pearson

La statistica del test, detta **statistica di Pearson**, è definita come
$$
W = \sum_{j=1}^K \frac{(O_j - A_j)^2}{A_j}.
$$
Questa quantità cresce quanto più le frequenze osservate si discostano da quelle teoriche, pesando lo scarto in modo relativo alla frequenza attesa.

Sotto l’ipotesi nulla e assumendo che il campione sia sufficientemente grande (in modo che le frequenze attese non siano troppo piccole), la statistica $W$ ha approssimativamente una distribuzione $\chi^2$ con $K-1$ gradi di libertà. Questo risultato permette di confrontare il valore osservato di $W$ con i quantili della distribuzione $\chi^2$ e decidere se rigettare o meno $H_0$.

#### Statistica alternativa: $G$

Una variante più moderna della statistica di Pearson è la **statistica $G$**, spesso detta *likelihood ratio statistic*. Essa nasce dal confronto tra la verosimiglianza del modello sotto l’ipotesi nulla e quella del modello saturato, cioè perfettamente aderente ai dati osservati. La sua espressione è
$$
G = 2 \sum_{j=1}^K O_j \log \frac{O_j}{A_j}.
$$
Dal punto di vista interpretativo, anche $G$ misura quanto le frequenze osservate si discostano da quelle attese, ma lo fa in termini di informazione e verosimiglianza anziché tramite scarti quadratici. Sotto l’ipotesi nulla e in regime asintotico, la statistica $G$ ha approssimativamente una distribuzione $\chi^2$ con $K-1$ gradi di libertà, risultando quindi equivalente al test di Pearson dal punto di vista decisionale per campioni grandi.

## Estensione a distribuzioni con molti valori o continue

Quando la distribuzione da testare è continua, come nel caso della normale o dell’esponenziale, oppure quando il numero di valori distinti è molto grande, il test del Chi-quadro non si applica direttamente alle singole osservazioni. In questi casi si suddivide lo spazio dei valori possibili in $K$ intervalli, detti *bin*, e si riconduce il problema a un caso discreto.

Per ciascun bin $B_j$, la frequenza osservata è
$$
O_j = \#\{X_i \in B_j\},
$$
mentre la frequenza attesa sotto l’ipotesi nulla è
$$
A_j = n \cdot P_{H_0}(X \in B_j),
$$
dove la probabilità è calcolata usando la distribuzione teorica ipotizzata. Il test procede poi esattamente come nel caso discreto, calcolando una statistica di tipo $\chi^2$ (di Pearson o $G$).

Una scelta cruciale riguarda la costruzione dei bin. In pratica è consigliabile definirli in modo che le frequenze attese risultino il più possibile omogenee, idealmente equiprobabili sotto $H_0$, così da rispettare più facilmente le condizioni di applicabilità e migliorare la qualità dell’approssimazione.

## Estensione a famiglie parametriche

In molti casi non si dispone di una distribuzione teorica completamente specificata, ma si vuole verificare se i dati provengono da una certa **famiglia parametrica**, come ad esempio la famiglia delle distribuzioni normali, lasciando però ignoti i parametri. La situazione tipica è il test di gaussianità, in cui si ipotizza
$$
X \sim \mathcal{N}(\mu,\sigma^2),
$$
ma i parametri $\mu$ e $\sigma^2$ non sono noti a priori.

La procedura segue la stessa logica del test di adattamento visto prima, con una differenza fondamentale: i parametri della distribuzione teorica vengono stimati a partire dal campione. In pratica si calcolano le stime campionarie $\hat{\mu}$ e $\hat{\sigma}$ e si costruiscono le frequenze attese usando la distribuzione $\mathcal{N}(\hat{\mu},\hat{\sigma}^2)$. Su queste frequenze si calcola poi la statistica di Pearson $W$ oppure la statistica $G$.

Un aspetto cruciale riguarda i **gradi di libertà**. Ogni parametro stimato dai dati riduce l’informazione disponibile per il test. Di conseguenza, se il campione è suddiviso in $K$ classi e si stimano $p$ parametri, la statistica del test segue approssimativamente una distribuzione
$$
\chi^2(K - 1 - p).
$$
Nel caso della normale, in cui si stimano due parametri ($\mu$ e $\sigma$), i gradi di libertà diventano $K - 3$.

## Test su tabelle di contingenza

Il test del Chi-quadro trova un’applicazione molto importante nello studio dell’indipendenza tra due variabili categoriali. In questo contesto i dati vengono organizzati in una **tabella di contingenza**, in cui ogni cella contiene la frequenza osservata $O_{ij}$ corrispondente alla combinazione della $i$-esima modalità della prima variabile con la $j$-esima modalità della seconda.

L’ipotesi nulla afferma che le due variabili siano indipendenti, mentre l’ipotesi alternativa sostiene l’esistenza di un’associazione tra di esse. Sotto l’ipotesi di indipendenza, la frequenza attesa nella cella $(i,j)$ si ottiene combinando i totali marginali di riga e di colonna:
$$
A_{ij} = \frac{(\text{totale riga } i)\,(\text{totale colonna } j)}{N},
$$
dove $N$ è la numerosità totale del campione.

La discrepanza tra frequenze osservate e attese può essere misurata tramite la statistica di Pearson
$$
W = \sum_{i=1}^m \sum_{j=1}^n \frac{(O_{ij} - A_{ij})^2}{A_{ij}},
$$
oppure tramite la statistica $G$
$$
G = 2 \sum_{i=1}^m \sum_{j=1}^n O_{ij} \log \frac{O_{ij}}{A_{ij}}.
$$
In entrambi i casi, sotto l’ipotesi nulla e per campioni sufficientemente grandi, la statistica segue approssimativamente una distribuzione $\chi^2$ con $(m-1)(n-1)$ gradi di libertà, dove $m$ e $n$ sono rispettivamente il numero di righe e di colonne della tabella.

## Versioni esatte del test

Quando le frequenze attese risultano troppo piccole, l’approssimazione asintotica alla distribuzione $\chi^2$ non è più affidabile. In queste situazioni è necessario ricorrere a **versioni esatte** del test, che non si basano su limiti asintotici ma sulla distribuzione reale della statistica sotto l’ipotesi nulla.

Un primo approccio consiste nella **simulazione Monte Carlo**. Si generano molti campioni artificiali assumendo vera $H_0$, si calcola per ciascuno la statistica del test (ad esempio $W$ o $G$) e si costruisce così una distribuzione empirica della statistica sotto l’ipotesi nulla. Il p-value viene stimato come la proporzione di simulazioni in cui la statistica simulata è almeno grande quanto quella osservata.

Un secondo approccio, concettualmente più diretto ma computazionalmente più oneroso, è l’**enumerazione esaustiva**. In questo caso si considerano tutte le possibili configurazioni di conteggi compatibili con i vincoli imposti (ad esempio i totali di riga e colonna) e si calcola esattamente la distribuzione della statistica del test. Questo metodo è praticabile solo per problemi di dimensioni molto ridotte, poiché il numero di configurazioni cresce rapidamente.

### Test esatto di Fisher–Irwin

Un caso particolarmente importante di test esatto è il **test di Fisher–Irwin**, applicabile alle tabelle di contingenza $2 \times 2$ con totali fissati. In questo contesto, sotto l’ipotesi nulla di indipendenza tra le due variabili, l’intera tabella è determinata dal valore di una sola cella, ad esempio $O_{11}$.

La distribuzione esatta di $O_{11}$ sotto $H_0$ è una **distribuzione ipergeometrica**. Indicando con $a$ il totale della prima riga, con $c$ e $d$ i totali delle due colonne e con $n$ il totale complessivo, vale
$$
P(O_{11} = k) = \frac{\binom{c}{k}\,\binom{d}{a - k}}{\binom{n}{a}}.
$$
Questa formula fornisce la probabilità esatta di osservare il valore $k$ nella cella $(1,1)$, condizionatamente ai totali marginali.

Il calcolo del **p-value** dipende dalla forma della distribuzione. Se la distribuzione è simmetrica, il p-value bilaterale si ottiene raddoppiando la probabilità di osservare un valore almeno così estremo nella coda considerata,
$$
p = 2 \cdot P(O_{11} \le k_{\text{oss}}).
$$
Nel caso generale, in cui la distribuzione non è simmetrica, il p-value si calcola sommando le probabilità di tutti gli esiti che risultano almeno tanto improbabili quanto quello osservato, secondo la distribuzione ipergeometrica sotto $H_0$.
