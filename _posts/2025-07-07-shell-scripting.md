---
layout: post
title: Introduzione allo Shell Scripting
categories: [Linux]
---


Lo shell script è uno strumento potente e versatile che permette di automatizzare operazioni e task ripetitivi sui sistemi operativi Unix/Linux e macOS. Grazie agli script, puoi risparmiare tempo, ridurre gli errori e migliorare notevolmente la produttività. In questo post esploreremo cos'è uno shell script, perché è importante imparare a usarlo e ti guiderò passo passo verso la scrittura del tuo primo script.


Ecco i punti principali che affronteremo:

* Cos’è uno shell script
* Le basi della sintassi della shell
* Variabili, parametri e argomenti
* Strutture di controllo (if, for, while)
* Funzioni e modularità
* Alcuni esempi pratici
* Buone pratiche di programmazione con la shell

---

## Cos’è uno Shell Script?

Uno **shell script** è un file di testo contenente una serie di comandi interpretati da una shell (come Bash, Zsh o altre). La shell esegue in sequenza i comandi scritti nello script, proprio come se fossero digitati manualmente nel terminale.

Esempio di un semplice shell script (salvato in `script.sh`):

```bash
#!/bin/bash

echo "Ciao, mondo!"
date
```

* La prima riga (`#!/bin/bash`) indica al sistema operativo che interpreterà lo script con la shell Bash.
* Le successive righe contengono i comandi da eseguire.

Per rendere eseguibile lo script, è necessario modificare i permessi con il comando:

```bash
chmod +x script.sh
```

Per eseguire lo script, basta digitare:

```bash
./script.sh
```

---

## Le basi della sintassi della shell

La sintassi di uno shell script è semplice, diretta e potente. Alcune regole generali:

* Ogni comando è separato da una nuova riga.
* I commenti si indicano con `#`.
* Gli spazi sono importanti: separano comandi e argomenti.
* Le variabili sono definite e richiamate tramite `$`.

Ecco un esempio:

```bash
# Commento: definizione variabile
NOME="Marco"

# Stampa il valore della variabile
echo "Ciao, $NOME!"
```

---

## Variabili, parametri e argomenti

Le variabili nello shell scripting possono essere definite semplicemente con la sintassi:

```bash
VARIABILE="valore"
```

Non aggiungere spazi prima o dopo il segno `=`.

Per utilizzare il valore di una variabile, anteponi il simbolo `$` al nome della variabile, ad esempio `$VARIABILE`.

### Argomenti dello script

Quando lanci uno script, puoi passargli degli argomenti:

```bash
./script.sh arg1 arg2 arg3
```

Nello script puoi accedere agli argomenti con `$1`, `$2`, `$3`, ecc.

Esempio di script con argomenti:

```bash
#!/bin/bash

echo "Primo argomento: $1"
echo "Secondo argomento: $2"
```

---

## Strutture di controllo

Le strutture di controllo (if, for, while) permettono di prendere decisioni o iterare azioni.

### Struttura condizionale `if`

Sintassi base:

```bash
if [ condizione ]; then
    # comandi se vero
else
    # comandi se falso
fi
```

Esempio:

```bash
#!/bin/bash

ETA=20

if [ $ETA -ge 18 ]; then
    echo "Maggiorenne"
else
    echo "Minorenne"
fi
```

Operatori di confronto comuni:

* `-eq` uguale
* `-ne` non uguale
* `-lt` minore
* `-le` minore o uguale
* `-gt` maggiore
* `-ge` maggiore o uguale

### Ciclo `for`

Sintassi base:

```bash
for variabile in lista_elementi; do
    # comandi
done
```

Esempio:

```bash
#!/bin/bash

for nome in Marco Anna Luca; do
    echo "Ciao, $nome!"
done
```

### Ciclo `while`

Sintassi base:

```bash
while [ condizione ]; do
    # comandi
done
```

Esempio pratico:

```bash
#!/bin/bash

CONTO=1
while [ $CONTO -le 5 ]; do
    echo "Numero: $CONTO"
    ((CONTO++))
done
```

---

## Funzioni e modularità

Le funzioni permettono di organizzare meglio il codice:

```bash
#!/bin/bash

saluta(){
    echo "Ciao, $1!"
}

# Richiama la funzione
saluta "Marco"
saluta "Anna"
```

Le funzioni rendono lo script più leggibile e riutilizzabile.

---

## Alcuni esempi pratici

### Backup di file

Un semplice script di backup:

```bash
#!/bin/bash

DATA=$(date +%Y%m%d)
BACKUP_DIR="/tmp/backup_$DATA"

mkdir -p $BACKUP_DIR
cp ~/Documenti/*.txt $BACKUP_DIR

echo "Backup completato in $BACKUP_DIR"
```

### Monitoraggio dello spazio disco

```bash
#!/bin/bash

USO_DISCO=$(df -h / | awk 'NR==2 {print $5}')

echo "Utilizzo del disco principale: $USO_DISCO"
```

---

## Buone pratiche di programmazione shell

Seguire buone pratiche ti aiuterà a scrivere script robusti e facili da mantenere:

* Usa sempre `#!/bin/bash` in cima allo script.
* Commenta in maniera chiara e sintetica.
* Controlla sempre il risultato dei comandi critici.
* Mantieni lo script breve e semplice.
* Usa nomi di variabili descrittivi e coerenti.

---

## Conclusione e consigli utili

Lo shell scripting è una competenza molto utile per automatizzare il lavoro quotidiano e ottimizzare il tempo. Dopo aver esplorato i fondamenti, il prossimo passo è sperimentare con script via via più complessi e integrare altri strumenti di sistema (come `grep`, `sed` e `awk`) per sfruttare al massimo la potenza della shell.

Per approfondire ulteriormente, puoi consultare:

* [Shell Scripting Tutorial](https://www.shellscript.sh)