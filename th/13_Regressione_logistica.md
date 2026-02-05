# Regressione logistica

La regressione logistica viene utilizzata quando la variabile risposta è **dicotomica**, tipicamente codificata come $0/1$. Per ogni osservazione $i=1,\dots,N$, associata a un vettore di input $x^{(i)}=(x_1^{(i)},\dots,x_p^{(i)})$, si assume che la risposta $Z_i$ segua una distribuzione di Bernoulli con parametro $p_i$:
$$
Z_i \sim \text{Bernoulli}(p_i), \qquad p_i = f\!\big(x^{(i)}\big).
$$

La probabilità $p_i$ è modellata tramite una **sigmoide** applicata a un predittore lineare. In particolare si pone

$$
p_i = \sigma(\eta_i), \qquad 
\eta_i = \beta_0 + \beta_1 x_1^{(i)} + \cdots + \beta_p x_p^{(i)},
$$
dove la funzione sigmoide è definita come
$$
\sigma(z) = \frac{1}{1+e^{-z}}.
$$
In questo modo il predittore lineare $\eta_i$ può assumere qualunque valore reale, mentre la sigmoide garantisce che $p_i \in (0,1)$.

## Stima dei parametri e funzione obiettivo

Il vettore dei parametri $\beta=(\beta_0,\dots,\beta_p)$ viene stimato tramite **massima verosimiglianza**. La log-verosimiglianza associata al campione è
$$
\ell(\beta) 
= \sum_{i=1}^N \Big[ Z_i \log p_i + (1-Z_i)\log(1-p_i) \Big],
\qquad 
p_i=\sigma(\eta_i).
$$

Massimizzare questa quantità è equivalente a minimizzare la **cross-entropy**, ossia la negativa della log-verosimiglianza media:
$$
\text{loss}_{\text{CE}}(\beta) 
= -\frac{1}{N}\sum_{i=1}^N \Big[ Z_i \log p_i + (1-Z_i)\log(1-p_i) \Big].
$$
Questa funzione obiettivo è convessa rispetto ai parametri e coincide con la funzione di perdita comunemente usata nei modelli di classificazione binaria.

Dal punto di vista computazionale, la stima dei parametri richiede metodi di ottimizzazione numerica, come Newton–Raphson o il metodo IRLS (Iteratively Reweighted Least Squares), oppure algoritmi di discesa del gradiente. In pratica si utilizzano formulazioni **numericamente stabili** della loss, ad esempio basate su trasformazioni tipo log-sum-exp, per evitare problemi di overflow o underflow.

Pur non esistendo una distribuzione esatta in forma chiusa per gli stimatori $\hat\beta$, gli stimatori di massima verosimiglianza sono **asintoticamente normali**. Questo consente di costruire intervalli di confidenza e test di ipotesi (Wald, Likelihood Ratio, Score) utilizzando la matrice di **informazione di Fisher**. Un aspetto critico è la presenza di **separazione completa o quasi** dei dati: in tali casi l’MLE può non esistere o divergere, e si ricorre a tecniche di penalizzazione, come la correzione di Firth, o alla regolarizzazione.

## Predizione

Dato un nuovo vettore di input $x$, il modello di regressione logistica restituisce direttamente una **probabilità** di appartenenza alla classe positiva:
$$
P(Z=1\mid x)=\sigma\!\big(\beta_0+\beta_1 x_1+\cdots+\beta_p x_p\big).
$$
Questa quantità rappresenta una stima di $P(Z=1\mid X=x)$ e non una decisione discreta. La classificazione si ottiene introducendo una **soglia** sulla probabilità, ad esempio $0.5$, ma tale scelta non è intrinseca al modello. In pratica la soglia va calibrata in funzione dell’obiettivo applicativo, tenendo conto del compromesso tra errori di tipo diverso, spesso tramite curve ROC o Precision–Recall e funzioni di costo asimmetriche.

## Regressione logistica multinomiale

Quando la variabile risposta assume **più di due categorie** ($m>2$), $Z_i\in\{1,\dots,m\}$, la regressione logistica si estende al caso multiclasse. Si utilizza una **codifica one-hot** della risposta, indicata con $b^{(i)}=(b^{(i)}_1,\dots,b^{(i)}_m)$, dove un solo elemento vale 1 e gli altri 0. In questo caso si assume
$$
B^{(i)} \sim \text{Multinomiale}\!\big(1;\ \pi(x^{(i)},W)\big),
$$
dove $\pi(x^{(i)},W)$ è il vettore delle probabilità di classe e $W$ è la matrice dei parametri del modello. Inserendo esplicitamente l’intercetta, si può considerare $W\in\mathbb{R}^{(p+1)\times m}$.

Per ciascuna classe $k$ si definisce un **logit** lineare
$$
g_k^{(i)} = w_{0k} + w_{1k} x_1^{(i)} + \cdots + w_{pk} x_p^{(i)},
$$
e le probabilità di appartenenza alle classi sono ottenute tramite la funzione **softmax**:
$$
\pi_k\!\big(x^{(i)},W\big) = 
\frac{e^{g_k^{(i)}}}{\sum_{h=1}^m e^{g_h^{(i)}}},
\qquad \sum_{k=1}^m \pi_k(\cdot)=1.
$$
La softmax generalizza la sigmoide al caso multiclasse, garantendo probabilità positive che sommano a uno.

La **log-verosimiglianza** del campione assume la forma
$$
\ell(W) = \sum_{i=1}^N \sum_{k=1}^m b^{(i)}_k \log \pi_k\!\big(x^{(i)},W\big),
$$
e la funzione obiettivo da minimizzare è la **cross-entropy media**:
$$
\text{loss}(W) = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^m b^{(i)}_k \log \pi_k\!\big(x^{(i)},W\big).
$$

Un aspetto teorico rilevante è l’**identificabilità** del modello: i logit sono definiti a meno di una traslazione comune, che non modifica le probabilità softmax. Per ottenere stime univoche è quindi necessario imporre un vincolo, ad esempio fissando una classe di riferimento (modello baseline-category logit) oppure imponendo condizioni di somma o di centratura sui parametri.

## One-vs-Rest

Un approccio alternativo alla regressione logistica multinomiale consiste nello stimare **$m$ modelli binari distinti**, uno per ciascuna classe, confrontando la classe $k$ contro il resto delle classi. Questo schema, noto come *one-vs-rest*, è concettualmente semplice e facile da implementare, poiché riduce il problema multiclasse a una collezione di problemi binari standard. Tuttavia, le probabilità stimate dai singoli modelli non sono coerenti tra loro e in generale **non sommano a 1**, per cui non ammettono un’interpretazione probabilistica congiunta. Per questo motivo il metodo è spesso usato come baseline o in contesti applicativi semplici, mentre risulta meno rigoroso rispetto alla formulazione multinomiale con softmax.