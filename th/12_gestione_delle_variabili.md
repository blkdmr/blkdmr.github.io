# Gestione delle variabili

Quando si costruisce un modello statistico, non solo di regressione, è essenziale trattare correttamente le **variabili** in base alla loro natura. In generale si distinguono variabili dicotomiche, categoriche e numeriche, e questa distinzione influisce direttamente sul tipo di modello utilizzabile e sull’interpretazione dei parametri.

Le **variabili dicotomiche** hanno due soli livelli, come sì/no o vero/falso. Le **variabili categoriche** presentano più di due modalità, ad esempio la stagione o la regione di provenienza. Le **variabili numeriche**, infine, rappresentano quantità misurabili e possono essere continue o discrete.

## Variabili dicotomiche

### Come variabili esplicative

Quando una variabile dicotomica è usata come ingresso, viene quasi sempre codificata con i valori **0 e 1**, anche se in teoria qualsiasi coppia di numeri distinti sarebbe possibile. Questa scelta rende immediata l’interpretazione dei coefficienti.

Se, ad esempio, $x_i=1$ indica “femmina” e $x_i=0$ “maschio”, allora il coefficiente $\beta_1$ associato a $x_i$ misura la differenza media nella risposta $Y$ tra il gruppo femminile e il gruppo di riferimento (maschile), a parità delle altre variabili. Per questo motivo le dicotomiche sono molto comuni anche nei **disegni sperimentali**, dove permettono confronti diretti tra trattamento e controllo.

### Come variabili di risposta

Una variabile dicotomica non è adatta come risposta in un modello di regressione lineare classica, perché le ipotesi di normalità e omoschedasticità risultano violate. In questi casi si utilizzano modelli specifici per la classificazione, come la **regressione logistica** nel caso binario, l’analisi discriminante o, più in generale, modelli di machine learning per la classificazione.

## Variabili categoriche

Le variabili categoriche si distinguono in **nominali** e **ordinali**. Le nominali non hanno un ordine naturale tra le categorie, come nel caso della regione di provenienza o del tipo di carburante. Le ordinali, invece, presentano un ordine intrinseco, ad esempio il livello di istruzione o il grado di gravità di un sintomo.

Talvolta le variabili ordinali vengono codificate con numeri interi crescenti, ma questa scelta va fatta con cautela, perché introduce implicitamente una nozione di distanza tra le categorie che potrebbe non essere giustificata dal fenomeno osservato.

### Come variabili di risposta

Quando la variabile risposta è categorica, il modello di regressione lineare non è appropriato. A seconda del numero di categorie, si ricorre alla regressione logistica binaria o multinomiale, oppure a modelli di classificazione più generali come alberi decisionali, foreste casuali o reti neurali.

### Come variabili di ingresso

In un modello di regressione, una variabile categorica con $k$ livelli non può essere inserita direttamente come numero. Deve essere trasformata in **$k-1$ variabili dicotomiche**, dette *dummy*, scegliendo una categoria come riferimento. Questa scelta evita problemi di collinearità e consente di interpretare i coefficienti come differenze rispetto al livello di riferimento.

## Codifica one-hot con categoria di riferimento

Nel caso di una variabile categorica con $k$ livelli, la codifica one-hot richiede di scegliere una categoria come **riferimento** (o *default*) e di costruire solo $k-1$ variabili binarie. La categoria di riferimento non compare esplicitamente nel modello, ma è assorbita nel termine noto.

Se, ad esempio, la variabile *stagione* ha quattro livelli (inverno, primavera, estate, autunno) e si sceglie **inverno** come default, il modello lineare assume la forma
$$
Y = \beta_0 + \beta_{\text{pri}} x_{\text{pri}} + \beta_{\text{est}} x_{\text{est}} + \beta_{\text{aut}} x_{\text{aut}} + \varepsilon.
$$
In questa parametrizzazione, $\beta_0$ rappresenta il valore atteso di $Y$ per la categoria di riferimento (inverno), mentre ciascun coefficiente associato alle dummy misura la **differenza media rispetto al default**.

Considerando, ad esempio, le seguenti stime:

| Coefficiente | Valore |
|---|---|
| $\beta_0$ (inverno) | 16.4 |
| $\beta_{\text{pri}}$ (primavera) | 5.4 |
| $\beta_{\text{est}}$ (estate) | 6.7 |
| $\beta_{\text{aut}}$ (autunno) | 1.8 |

i valori attesi della risposta risultano:

| Stagione   | $E[Y]$ |
|---|---|
| Inverno    | 16.4 |
| Primavera  | 21.8 |
| Estate     | 23.1 |
| Autunno    | 18.2 |

L’interpretazione è immediata: ciascun coefficiente indica di quanto la media della risposta cambia rispetto all’inverno. Questa codifica consente anche di formulare test di ipotesi mirati, ad esempio verificare se l’autunno è indistinguibile dall’inverno tramite $H_0:\beta_{\text{aut}}=0$, oppure confrontare direttamente primavera ed estate con $H_0:\beta_{\text{pri}}=\beta_{\text{est}}$.

## Effetto della selezione stepwise

Quando si applica una procedura di **stepwise backward**, può accadere che una delle variabili dummy venga eliminata dal modello. Questo va interpretato come evidenza del fatto che quella categoria non è statisticamente distinguibile dalla categoria di riferimento. Se, ad esempio, la dummy per l’autunno viene rimossa, il modello sta suggerendo che autunno e inverno possono essere considerati equivalenti rispetto alla risposta.

In termini di codifica, la situazione diventa:

| Stagione   | primavera | estate |
|---|---|---|
| Primavera  | 1 | 0 |
| Estate     | 0 | 1 |
| Autunno    | 0 | 0 |
| Inverno    | 0 | 0 |

Autunno viene quindi trattato **come inverno**, poiché entrambe le categorie condividono la stessa combinazione di dummy.

Se l’interesse è confrontare direttamente due categorie non di riferimento, come primavera ed estate, ci sono due possibilità concettualmente equivalenti. La prima consiste nel formulare un test lineare sui parametri, ad esempio $H_0:\beta_{\text{pri}}=\beta_{\text{est}}$. La seconda è cambiare la categoria di riferimento, ad esempio ponendo estate come default, e ricodificare il modello.

Va infine ricordato che ogni categoria aggiuntiva aumenta il numero di parametri $p$. È quindi importante monitorare il rapporto $n/p$: in presenza di categorie rare può essere opportuno accorparle in macro-categorie, oppure valutare l’esclusione di alcune osservazioni, per mantenere il modello stabile e interpretabile.

## Variabili numeriche

Le variabili numeriche rappresentano quantità misurabili e, a differenza delle categoriche, possono entrare direttamente in un modello di regressione. Tuttavia, il loro **significato statistico** e la loro **scala** influenzano il modo in cui è opportuno utilizzarle.

### Variabili di tipo “differenza”

Le variabili di tipo differenza sono quelle per cui ha senso interpretare direttamente una variazione additiva. Esempi tipici sono statura, peso, quoziente intellettivo o consumi espressi in unità fisiche standard. In questi casi non sono richieste trasformazioni particolari: il coefficiente di regressione misura semplicemente la variazione media della risposta associata a un incremento unitario della variabile.

### Variabili di tipo “rapporto”

Le variabili di tipo rapporto sono definite su valori **positivi** e spesso presentano una distribuzione fortemente asimmetrica o estesa su più ordini di grandezza, come reddito, popolazione o fatturato. In questi casi è frequente applicare una **trasformazione logaritmica**.

In un modello lineare del tipo
$$
Y = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p + \varepsilon,
$$
se $x_1$ è una variabile di tipo rapporto, l’uso di $\log x_1$ può rendere la relazione più vicina alla linearità e ridurre l’asimmetria della distribuzione, migliorando il comportamento dei residui.

### Effetti delle trasformazioni

Le **trasformazioni lineari**, come il cambio di unità di misura o una riscalatura affine, non modificano le conclusioni statistiche fondamentali del modello, come i test di ipotesi o il valore di $R^2$, anche se cambiano il valore numerico dei coefficienti.

Le **trasformazioni non lineari**, come il logaritmo o la radice quadrata, hanno invece un impatto più sostanziale: possono migliorare la normalità dei residui, la stabilità della varianza e la linearità della relazione tra risposta e predittori, rendendo il modello più adeguato alle ipotesi teoriche.

Un caso frequente è la regressione logaritmica sulla risposta:
$$
\log Y = \beta_0 + \beta_1 x_1 + \cdots,
$$
che, riscritta sulla scala originale, diventa
$$
Y = e^{\beta_0}\cdot e^{\beta_1 x_1}\cdot \cdots.
$$
In questa formulazione i coefficienti hanno un’interpretazione **moltiplicativa**. Ad esempio, se $\beta_1=\log(1.4)$, un aumento unitario di $x_1$ è associato a un incremento medio di circa **+40\%** di $Y$, a parità delle altre variabili.