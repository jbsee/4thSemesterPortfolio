---
title: System Development Methodology Draft
publish: true
---
## Lineær opstart
Målet er hurtigst muligt at komme igennem den lineære del af projektet, mens vi stadig benytter relevante teknikker, for at sikre gode forudsætninger for langsigtet verificering og validitet.
#### Iteration 0

- **Forventningsafstemning**
	- Indledende **kravindsamling **- hvad har vi med at gøre?
	- **Foranalyse/analyse**
		- **Hvis** vi har brug for overordnede analyser, som omhandler hele systemet/projektet (BMC, BPMN, Risikoanalyse (kompleksitet/usikkerhed)), udføres de nu.
- **HLD**
		- Lav 1-2 usecases hver, smæk det hurtigt ind i en lofi-domænemodel-skitse, lav en hurtig DCD inkl. lag (pakkediagram). Sikrer en fælles vision for systemet uden at vi synker ned i design-sumpen.
- **MVP** (Minimal viable product)
		- Når vi har en fælles vision for det endelige system definerer vi hvilke dele af systemet, der er nødvendige for at kunne udrulle en MVP. Den agerer bibel indtil vi har realiseret den:
		 Vi arbejder ud fra en lofi-filosofi, hvor vi sigter efter at have noget, der kan køre på en telefon/web app hurtigst muligt med kun de allermest nødvendige features, men **som giver reel værdi til kunden**.
-  **Udviklingsplan**
	- Vi dokumenterer valgt arkitektur, sprog, mønstre, konventioner, WIP, DOD, DOT, testfilosofi etc.

Alle hifi artefakter/modeller bliver udviklet pr. MVP.

## Inkrementel levering gennem evolutionære prototyper
Formålet er hurtigst muligt at udrulle noget, der både er meningsfuldt og testbart, hvilket giver mulighed for løbende feedback.

#### Inkrementel levering

- **Vi udruller hver gang vi har realiseret en MVP.** Efter realisering defineres den næste MVP.

#### Evolutionære prototyper
-  Vi arbejder **iterativt på MVP-niveau**. Dvs. det daglige arbejde går ud på at øge fidelity på det eksisterende system samtidig med at vi tilføjer nye lofi-features. 
- Hvilke features/fidelity vi arbejder på, bestemmes ud fra en **prioriteret liste over ønskede fidelity/features**. Prioritering sker så vidt muligt udelukkende **på baggrund af anslået kunde-værdi (burde kunne udledes af indledende krav-indsamling)** - og så alligevel ikke helt, fordi det kan give mening også at prioritere at alle har noget at lave inden for deres specialisering. Formålet er at sikre, at projektet både giver mening individuelt og på gruppe/produkt-niveau.
- Der kan evt. laves en Gantt-inspireret critical path for hver MVP, så der kan planlægges efter evt. afhængigheder (Kan måske gøres uformelt under prioritering - men afhængigheder er gode at have in mente).
- MVP'er kan siges at agere sprints, men disse kan variere i længde og kan sagtens vare flere uger. Derfor bruges kanban-boardet til daglig projektstyring - ingen sprint deadline - vi arbejder os igennem vores prioriterede MVP-tasks indtil vi når i mål.
	- Progress-evaluering hver fredag? Nye udfordringer eller krav? Er vi stadig på sporet, går det fremad i det ønskede tempo, er der noget vi bliver nødt til at udskyde til næste MVP? Evt. ændringer i task-prioriteter

#### HLD/LLD
Vi laver de artefakter, vi finder nødvendige for at kunne gå i gang med implementering - hverken flere eller færre. Disse udarbejdes med så meget fidelity, som vi finder gavnligt. Ligesom vi hele tiden integrerer ny kode med det samlede system, udbygger vi hele tiden de eksisterende modeller. Der skulle således gerne være (nogenlunde) overensstemmelse mellem kode og artefakter når vi udruller en MVP. Der kan evt. laves en lille artefakt-opsamling/høj-prioritet-artefakt tasks efter hver udrulning, hvor alt opdateres, så artefakterne ikke rådner.


#### Værdi-prioritering
Det er værd at bemærke, at vores overordnede mål for at opfylde kundekrav kan være afhængige at ikke-funktionelle delmål. F.eks. kan det give mening først at prioritere proof of concept af arkitekturen, da ingen værdi kan leveres til kunden, hvis vi sejler rundt i arkitektur i to måneder.


## Udfordringer
- Den mest åbenlyse er **ingen hårde deadlines**: Vi har ikke en Gantt-tidslinje eller timeboxing, så **selvdisciplin** er nok den største driver, hvis vi bruger denne model. Evt. kan ugentlige kanban-board-throughput-parametre udvikles hvis nødvendigt?
- Endeligt mål kan føles fluffy. Selvom vi har krav til systemet, vi skal opfylde, lægges der aldrig konkrete planer for mere end den næste MVP.
- Prioritering kan blive en diskussionsklub - sporbarhed til krav essentiel som tie-breaker.
- Arbejdsfordeling kan være tricky. Hvor meget skal der laves hvor hurtigt, hvem gør hvad, har alle nok at lave/ingen for lidt at lave. Gælder om at undgå flaskehalse og sikre, at det overordnede projekt ikke stagnerer. Der kan evt. laves planer på forhånd til afhjælpning af problemer, f.eks. en handleplan for flaskehalse - alle der kan hjælpe, pauser med det samme hvad de laver, og prioriterer flaskehalsen indtil afhjulpet.


## Fordele
 - Omvendt er fordelen, at vi bare kan arbejde mod et mål uden at blive stressede over at skulle færdiggøre et system til en bestemt dato. Ved at arbejde med levering gennem evolutionære prototyper, sikrer vi, at vi kommer til at have en eller anden brugbar version af vores system, når vi rammer projekt-deadlinen. Spørgsmålet er hvor mange features og hvor høj fidelity, vi har nået at lave før deadlinen.
 - Mulighed for **løbende brugertests** gennem MVP-udrulninger.
 - Fleksibilitet til at *spike*; *pivotere*, hvis kravene ændrer sig; *defer tough decisions* indtil vi har opnået et godt nok grundlag for beslutninger/bliver nødt til at beslutte noget.