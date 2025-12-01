---
title: Lidt teori og begreber
publish: true
---
## Underfitting og overfitting
![[Pasted image 20251130023443.png]]

Hvor godt en statistisk model passer til et sæt kendte observationer. En statistisk model er en matematisk repræsentation af data fra den virkelige verden, som hjælper med at komme med forudsigelser. Disse tre begreber er grundlæggende inden for maskinlæring, da de vedrører en models ydeevne på træningsdata og usete data.

**Undertilpasset**
Det første diagram illustrerer en undertilpasset model. Modellen ikke lærer mønstrene i træningsdataene godt nok. Denne enkelhed fører til en høj fejlprocent på både træningsdata og nye, usete data. Modellen, som er repræsenteret ved den lige linje, fanger ikke den underliggende tendens i dataene og behandler dem, som om de var lineært separerbare.

**Optimal**
Det andet diagram viser en optimal model, hvor linjen nøjagtigt adskiller de to klasser af datapunkter. Denne model finder en god balance og generaliserer godt ud fra træningsdataene. Det betyder, at den sandsynligvis vil klare sig godt på nye, usete data, fordi den har lært det underliggende mønster uden at blive alt for påvirket af støj og outliers.

**Overtilpasset**
Det sidste diagram illustrerer en overtilpasset model, som er for kompleks og passer for godt til træningsdataene. Modellen er så tilpasset, at den ikke kun lærer det underliggende mønster, men også støjen og outliers i træningsdataene, som det ses af dens uberegnelige opførsel omkring nogle få datapunkter. Resultatet er, at selvom den klarer sig godt på træningsdataene, vil den sandsynligvis klare sig dårligt på nye, usete data.

---
## Bias og variance

**Bias (Fejlagtige antagelser)** 
Bias er fejlen, der opstår, fordi din model gør forsimplede antagelser om den virkelige verden for at gøre målfunktionen lettere at lære. En model med høj bias ignorerer relevante relationer mellem features og outputtet. I praksis betyder det, at modellen ikke fanger den underliggende trend i dataene. Hvis dine data danner en parabel (en kurve), men du tvinger en lineær regression (en ret linje) ned over dem, har du indbygget en systematisk fejl. Uanset hvor meget data du fodrer den med, vil den aldrig ramme rigtigt, fordi dens grundlæggende verdensbillede er for simpelt. Dette fører direkte til underfitting.

**Variance (Følsomhed over for støj)** 
Variance angiver, hvor meget modellens estimat af målfunktionen ville ændre sig, hvis vi brugte et andet træningsdatasæt. Ideelt set burde modellen ikke ændre sig meget fra det ene sæt til det andet. Høj variance betyder, at modellen lægger for meget vægt på de specifikke datapunkter i træningssættet – inklusiv den tilfældige støj. Den "memorerer" dataene i stedet for at generalisere. Ser den et nyt datasæt, fejler den totalt, fordi den har optimeret sig selv til at passe perfekt til de mikroskopiske særheder i det første sæt. Dette er mekanismen bag overfitting.

**Relationen: Bias-Variance Tradeoff** 
De to begreber er modsatrettede kræfter. Du kan ikke bare minimere begge to isoleret set; du betaler for reduktion af den ene med en stigning i den anden.

Hvis du øger modellens kompleksitet (f.eks. tilføjer flere parametre eller lag i et neuralt netværk), falder din bias, fordi modellen bliver bedre til at fange komplekse mønstre. Prisen er, at din variance stiger, fordi modellen bliver hypersensitiv over for hver eneste lille variation i træningsdataene.

Gør du modellen simplere for at stabilisere den (lavere variance), stiger din bias, fordi modellen mister evnen til at beskrive komplekse sammenhænge.

Målet i data science er ikke at eliminere bias eller variance – det er umuligt – men at finde det punkt på kurven (Total Error), hvor summen af bias og variance er lavest. Det er det punkt, hvor modellen er kompleks nok til at forstå signalet, men robust nok til at ignorere støjen.

---
## Softmax
Softmax er en normaliserings-mekanisme. Den tager en vektor af "logits" (rå, vilkårlige tal fra netværkets sidste lag, som kan være negative eller uendeligt store) og tvinger dem ned i en pæn sandsynlighedsfordeling mellem 0 og 1.

#### Matematikken (The "Squashing" Function)
Formlen gør to ting: Gør tallene positive (eksponentiering) og normaliserer dem (division).

For et givent tal $x_i$ i din input-vektor ser formlen sådan ud:

$$\sigma(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

1. **Tælleren ($e^{x_i}$):** Vi tager tallet og opløfter $e$ (Eulers tal $\approx 2.718$) i det.

    - _Hvorfor?_ Eksponentialfunktionen gør alle tal positive (selv negative logits bliver positive). Vigtigere endnu: Den forstørrer forskelle. Et tal, der er lidt større end et andet, får en _meget_ større værdi efter $e^x$. Det tvinger modellen til at være mere "selvsikker" (confident).

2. **Nævneren ($\sum e^{x_j}$):** Vi lægger alle de eksponentierede tal sammen.

3. **Divisionen:** Vi deler det enkelte tal med totalsummen. Det garanterer, at summen af alle outputs altid er præcis 1 (100%).

#### Eksempel
**Logits:** $[2.0, \quad 0.5, \quad -1.0]$

##### Trin 1: Eksponentiering ($e^x$)
Her sker det vigtige: $e$ opløftet i et negativt tal bliver aldrig negativt eller nul. Det bliver en brøk ($e^{-1} = 1/e$). Det sikrer, at selv den mest usandsynlige klasse har en teoretisk chance (større end 0).

- $e^{2.0} \approx 7.39$
- $e^{0.5} \approx 1.65$
- $e^{-1.0} \approx 0.37$ (Bemærk: Positivt tal, men under 1)

##### Trin 2: Sum
Vi lægger de nye, positive værdier sammen.

$7.39 + 1.65 + 0.37 = 9.41$

##### Trin 3: Normalisering
Nu dividerer vi som før for at få procenterne.

- **Klasse A (Logit 2.0):** $7.39 / 9.41 \approx 0.785$ ($\mathbf{78.5\%}$)
- **Klasse B (Logit 0.5):** $1.65 / 9.41 \approx 0.175$ ($\mathbf{17.5\%}$)
- **Klasse C (Logit -1.0):** $0.37 / 9.41 \approx 0.040$ ($\mathbf{4.0\%}$)

**Resultat:** Summen er stadig 1.0 (med afrunding).

Det negative input ($-1.0$) dræbte ikke beregningen. Softmax skubbede det bare ned i bunden af hierarkiet. Det er derfor, sprogmodeller aldrig siger, at sandsynligheden for et ord er 0%. Der er altid en mikroskopisk chance for, at det næste ord er "kartoffel", selvom konteksten er kvantefysik.
#### I Sprogmodeller

I en LLM er Softmax det sidste skridt, før man ser et ord.

1. **Kontekst:** "København er hovedstaden i...".

2. **Logits:** Modellen beregner en logit-værdi for _hvert eneste ord_ (egentlig tokens, men for at danne sig en konceptuel forståelse, kan man tænke på det som ord) i sit vokabularium.

- "Danmark": Logit 15.4
- "Sverige": Logit 8.2
- "Havet": Logit -3.0

3. **Softmax:** Disse 50.000 tal køres gennem Softmax.

- "Danmark" bliver til 0.99 (99% sandsynlighed).
- "Sverige" bliver til 0.009 (0.9%).
- Resten deler resten.

Når man justerer temperatur i en LLM-API, piller man faktisk ved tallene, _før_ de rammer Softmax. En høj temperatur dividerer logits ned (gør dem tættere på hinanden), så Softmax giver en fladere fordeling $\rightarrow$ modellen vælger oftere noget uventet ("Sverige"). En lav temperatur fremhæver forskellene $\rightarrow$ modellen vælger det mest oplagte ("Danmark").

---