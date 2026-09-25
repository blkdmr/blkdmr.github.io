# Analisi delle componenti principali

La PCA (Principal Component Analysis) è una trasformazione lineare che permette di ridurre la dimensionalità dei dati mantenendo quanta più informazione possibile. L’idea è quella di individuare nuove variabili, dette **componenti principali**, ottenute come combinazioni lineari delle variabili originali, tali da spiegare in modo ottimale la variabilità dei dati. Operativamente, la PCA può essere interpretata come una **rotazione del sistema di riferimento** che rende la struttura dei dati più semplice.

Si consideri la matrice di covarianza dei dati $X$, indicata con $\Sigma = C(X)$. Questa matrice è quadrata, simmetrica e definita non negativa; di conseguenza ammette una decomposizione spettrale. Esistono quindi $m$ autovettori $v_j \in \mathbb{R}^m$ e autovalori associati $\lambda_j$ tali che

$$
\Sigma v_j = \lambda_j v_j, \qquad j = 1,\dots,m.
$$

Gli autovettori associati a una matrice simmetrica sono ortogonali tra loro e possono essere scelti a norma unitaria. In particolare vale

$$
v_j \cdot v_k =
\begin{cases}
1 & \text{se } j = k, \\
0 & \text{se } j \neq k.
\end{cases}
$$

Raccogliendo gli autovettori in una matrice

$$
V = [v_1, v_2, \dots, v_m],
$$

si ottiene una matrice ortogonale, per cui

$$
V^T V = I \qquad \text{e} \qquad V^{-1} = V^T.
$$

La matrice $V$ rappresenta quindi una rotazione nello spazio delle variabili. In termini di autovalori e autovettori, la matrice di covarianza può essere scritta come

$$
\Sigma = \sum_{i=1}^m \lambda_i \, v_i v_i^T.
$$

La trasformazione PCA è definita introducendo le nuove variabili

$$
Y = V^T X,
$$

che corrispondono alle coordinate di $X$ nel nuovo sistema di riferimento dato dagli autovettori. In questo nuovo spazio la matrice di covarianza risulta diagonalizzata,

$$
C(Y) = V^T \Sigma V = \Lambda = \operatorname{diag}(\lambda_1, \lambda_2, \dots, \lambda_m),
$$

dove ciascun autovalore $\lambda_j$ rappresenta la varianza spiegata dalla $j$-esima componente principale. Le componenti sono quindi scorrelate e ordinate per importanza decrescente in base al valore degli autovalori.

Dal punto di vista operativo, esistono due approcci principali alla PCA. Nel primo approccio i dati vengono prima traslati nell’origine sottraendo la media, poi ruotati tramite la PCA calcolata sulla matrice di covarianza, ed eventualmente standardizzati in un secondo momento. Nel secondo approccio, invece, si effettua prima la traslazione nell’origine e una standardizzazione delle variabili, quindi si applica la PCA sulla matrice di correlazione $g(X)$, seguita da una standardizzazione finale.

Il secondo approccio è generalmente preferibile quando le variabili originali hanno scale molto diverse, poiché evita che le componenti principali siano dominate dalle variabili con varianza numericamente più grande e conduce a un’analisi più equilibrata.

## Factor Analysis

L’analisi fattoriale ha l’obiettivo di descrivere un insieme potenzialmente ampio di variabili osservate tramite un numero ridotto di **fattori latenti**, interpretabili come dimensioni concettuali sottostanti e tra loro indipendenti. L’idea è che le variabili originali siano manifestazioni osservabili di pochi fattori non direttamente misurabili.

In un contesto come quello delle misure corporee, ad esempio, è naturale pensare che molte variabili siano influenzate da caratteristiche comuni. Un fattore potrebbe rappresentare la dimensione globale del corpo, un altro la larghezza, un altro ancora l’età o il grado di sviluppo muscolare, mentre fattori residui possono catturare differenze più specifiche o locali. In questo modo, strutture complesse vengono ricondotte a poche componenti interpretabili.

Quando il numero di variabili $m$ è elevato, molte delle componenti ottenute tramite PCA presentano una varianza molto bassa. Tali componenti contribuiscono poco alla variabilità complessiva dei dati e possono essere trascurate senza perdita informativa rilevante, riducendo così la dimensionalità effettiva da $m$ a un numero più contenuto $k$.

I **factor loadings**, ossia i coefficienti che legano ciascuna variabile osservata ai fattori, descrivono il peso e il segno del contributo di ogni variabile all’interno di un fattore. Valori elevati in modulo indicano che la variabile è fortemente associata a quel fattore, facilitandone l’interpretazione sostantiva.

Dal punto di vista della varianza, la PCA conserva la varianza totale del sistema. In particolare, la somma delle varianze delle variabili originali coincide con la somma delle varianze delle componenti trasformate,

$$
\mathrm{Var}(x_1) + \dots + \mathrm{Var}(x_m)
=
\mathrm{Var}(y_1) + \dots + \mathrm{Var}(y_m).
$$

Gli autovalori $\lambda_j$ della matrice di covarianza (o di correlazione) misurano la varianza spiegata da ciascuna componente e sono ordinati in modo decrescente. Uno strumento grafico comunemente utilizzato per decidere quante componenti o fattori trattenere è lo **scree plot**, che rappresenta gli autovalori in funzione dell’indice della componente e consente di individuare un punto di flesso oltre il quale il contributo informativo delle componenti diventa trascurabile.

## Distribuzione Gaussiana Multidimensionale

Un vettore aleatorio $X = (X_1, \dots, X_p)$ segue una distribuzione gaussiana multidimensionale con parametri $(\mu, \Sigma)$, indicata con $N(\mu, \Sigma)$, se la sua distribuzione è completamente determinata da un vettore medio e da una matrice di covarianza. Il vettore medio $\mu \in \mathbb{R}^p$ raccoglie le medie delle singole componenti, mentre la matrice di covarianza $\Sigma \in \mathbb{R}^{p \times p}$ descrive sia le varianze marginali sia le dipendenze lineari tra le componenti.

La funzione di densità di probabilità associata a $X$ è data da

$$
f_X(x) = \frac{1}{(2\pi)^{p/2} \sqrt{\det \Sigma}}
\exp\!\left(
-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)
\right),
$$

dove il termine quadratico nell’esponente misura la distanza di $x$ dalla media $\mu$ tenendo conto della struttura di correlazione imposta da $\Sigma$. La matrice $\Sigma$ deve essere simmetrica e definita positiva, in modo da garantire l’esistenza dell’inversa e la normalizzazione della densità.

Se le componenti di $X$ sono indipendenti, tutte le covarianze incrociate sono nulle e la matrice $\Sigma$ risulta diagonale. In questo caso la densità multidimensionale si fattorizza nel prodotto delle densità marginali e le curve di livello della distribuzione sono ellissi (o iper-ellissi) allineate con gli assi coordinati.

Quando invece le componenti non sono indipendenti, la matrice di covarianza contiene termini fuori diagonale. Geometricamente, ciò si traduce in ellissi di livello ruotate e deformate, il cui orientamento e allungamento riflettono le correlazioni e le diverse scale di variabilità delle variabili originali.

## Legge di Dirichlet

La legge di Dirichlet è una distribuzione di probabilità definita su vettori di **proporzioni**, ed è utilizzata quando si vuole modellare l’incertezza su come una quantità totale venga suddivisa tra più categorie. È particolarmente importante in ambito bayesiano, dove compare spesso come distribuzione *a priori* per vettori di probabilità, ad esempio nei modelli di clustering, nei modelli a miscela e nei topic model.

Formalmente, la distribuzione di Dirichlet genera vettori aleatori

$$
X = (X_1, \dots, X_m)
$$

tali che ogni componente sia non negativa e che la somma totale valga uno,

$$
X_i \ge 0, \qquad \sum_{i=1}^m X_i = 1.
$$

Si scrive

$$
X \sim \text{Dirichlet}(\alpha),
$$

dove il parametro $\alpha = (\alpha_1, \dots, \alpha_m)$ soddisfa $\alpha_i > 0$ per ogni $i$. I parametri $\alpha_i$ controllano la forma della distribuzione e riflettono il peso relativo attribuito a ciascuna categoria.

La funzione di densità di probabilità è data da

$$
f_X(x) = C_\alpha \, x_1^{\alpha_1 - 1} \cdots x_m^{\alpha_m - 1},
$$

dove $C_\alpha$ è una costante di normalizzazione che dipende dal vettore $\alpha$ e garantisce che la densità integri a uno sul simplesso $\{x : x_i \ge 0,\ \sum x_i = 1\}$.

La media della distribuzione di Dirichlet ha una forma particolarmente semplice ed è pari a

$$
\mathbb{E}(X_i) = \frac{\alpha_i}{\sum_{j=1}^m \alpha_j},
$$

oppure, in forma vettoriale,

$$
\mathbb{E}(X) = \frac{\alpha}{\sum_{i=1}^m \alpha_i}.
$$

Questo mostra che il parametro $\alpha$ può essere interpretato come un insieme di *pseudoconteggi*: il rapporto tra ciascun $\alpha_i$ e la loro somma determina il valore medio delle proporzioni.

Nel caso particolare $m = 2$, la distribuzione di Dirichlet si riduce alla distribuzione **Beta**, che modella una singola proporzione su $[0,1]$. La Dirichlet può quindi essere vista come una generalizzazione multivariata della Beta, adatta a descrivere simultaneamente più probabilità che devono sommare a uno.

## Distribuzione Multinomiale

La distribuzione multinomiale descrive il comportamento dei **conteggi** associati a più categorie quando si ripete uno stesso esperimento un numero fissato di volte. Può essere vista come una naturale estensione della distribuzione binomiale al caso in cui gli esiti possibili non siano due, ma $m \ge 2$. Un esempio tipico è il lancio ripetuto di un dado: ogni lancio produce uno tra $m=6$ risultati possibili e, dopo $n$ lanci, si è interessati a contare quante volte compare ciascuna faccia.

Formalmente, si considera un vettore aleatorio

$$
X = (X_1, \dots, X_m),
$$

dove $X_i$ rappresenta il numero di volte in cui si osserva l’esito appartenente alla categoria $i$ su un totale di $n$ prove indipendenti. Il modello è parametrizzato da un vettore di probabilità

$$
p = (p_1, \dots, p_m), \qquad \sum_{i=1}^m p_i = 1,
$$

che descrive la probabilità di ciascun esito in una singola prova. Si scrive allora

$$
X \sim \text{Multinomiale}(n, p).
$$

La funzione di massa di probabilità è data da

$$
P(X = x)
=
\frac{n!}{x_1! \cdots x_m!}
\, p_1^{x_1} \cdots p_m^{x_m},
\qquad
\sum_{i=1}^m x_i = n.
$$

Il coefficiente combinatorio tiene conto di tutti i possibili ordini in cui i diversi esiti possono verificarsi nelle $n$ prove, mentre il prodotto dei termini $p_i^{x_i}$ pesa ciascuna configurazione in base alle probabilità delle singole categorie.

Ciascuna componente $X_i$ ha distribuzione binomiale marginale con parametri $(n, p_i)$, ma le componenti non sono indipendenti tra loro. Il vincolo $\sum_i X_i = n$ implica infatti che un aumento in un conteggio debba essere compensato da una diminuzione in almeno un altro. La distribuzione multinomiale è quindi lo strumento naturale per modellare esperimenti con più di due esiti possibili, quando l’interesse è rivolto alla distribuzione congiunta dei conteggi.