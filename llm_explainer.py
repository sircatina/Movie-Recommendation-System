
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class LLMExplainer:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY lipseste din fisierul .env"
            )

        self.client = Groq(api_key=api_key)

        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b"
        )

    def explain(
        self,
        movie_title,
        movie_genres,
        predicted_score,
        preferred_genres,
        method="Hybrid"
    ):
        valid_methods = {
            "Hybrid",
            "Hybrid Adaptive",
            "Matrix Factorization",
            "Genre Model",
            "Cold Start"
        }

        if method not in valid_methods:
            raise ValueError(
                f"Metoda de recomandare necunoscuta: {method}"
            )

        movie_genres = list(movie_genres or [])
        preferred_genres = list(preferred_genres or [])

        preferred_set = {
            genre.casefold()
            for genre in preferred_genres
        }

        common_genres = [
            genre
            for genre in movie_genres
            if genre.casefold() in preferred_set
        ]

        other_genres = [
            genre
            for genre in movie_genres
            if genre.casefold() not in preferred_set
        ]

        genres_text = (
            ", ".join(movie_genres)
            if movie_genres
            else "Necunoscute"
        )

        preferences_text = (
            ", ".join(preferred_genres)
            if preferred_genres
            else "Nu sunt disponibile"
        )

        common_genres_text = (
            ", ".join(common_genres)
            if common_genres
            else "Niciunul"
        )

        other_genres_text = (
            ", ".join(other_genres)
            if other_genres
            else "Niciunul"
        )

        if method == "Cold Start":
            method_description = (
                "Utilizatorul nu are istoric de evaluari. "
                "A selectat explicit genurile preferate. "
                "Filmele sunt clasate folosind potrivirea "
                "genurilor si ratingul mediu regularizat. "
                "Nu exista informatii despre filmele "
                "vizionate anterior de acest utilizator."
            )

            score_description = (
                f"Scor intern: {predicted_score:.3f}/1. "
                "Nu reprezinta probabilitatea ca filmul "
                "sa fie apreciat."
            )

        elif method == "Matrix Factorization":
            method_description = (
                "Recomandarea este calculata exclusiv "
                "din tiparele invatate din evaluarile "
                "utilizatorilor si ale filmelor. "
                "Genurile nu sunt folosite direct "
                "pentru calculul acestei predictii. "
                "Nu exista dovezi ca un anumit gen "
                "a determinat recomandarea."
            )

            score_description = (
                f"Rating estimat: {predicted_score:.2f}/5."
            )

        elif method == "Genre Model":
            method_description = (
                "Recomandarea se bazeaza pe preferintele "
                "estimate ale utilizatorului pentru "
                "genurile filmului, calculate din "
                "evaluarile anterioare. "
                "Genurile preferate furnizate reprezinta "
                "cele mai bine cotate genuri din profil."
            )

            score_description = (
                f"Rating estimat: {predicted_score:.2f}/5."
            )

        elif method == "Hybrid":
            method_description = (
                "Recomandarea combina tiparele din "
                "evaluarile utilizatorilor cu preferintele "
                "estimate pentru genurile filmelor. "
                "Cele doua componente au ponderi fixe. "
                "Genurile preferate furnizate sunt "
                "cele mai bine cotate din profil."
            )

            score_description = (
                f"Rating estimat: {predicted_score:.2f}/5."
            )

        else:
            method_description = (
                "Recomandarea combina tiparele din "
                "evaluarile utilizatorilor cu preferintele "
                "estimate pentru genurile filmelor. "
                "Contributia celor doua componente "
                "depinde de numarul de evaluari "
                "disponibile pentru utilizator. "
                "Genurile preferate furnizate sunt "
                "cele mai bine cotate din profil."
            )

            score_description = (
                f"Rating estimat: {predicted_score:.2f}/5."
            )

        if method == "Cold Start":
            explanation_guidance = """
Utilizatorul si-a declarat explicit preferintele.

Explica recomandarea pornind de la genurile comune.

Daca filmul are toate genurile selectate,
poti spune ca se potriveste tuturor genurilor alese.

Nu mentiona ca utilizatorul a apreciat anterior
anumite filme sau genuri.

Poti mentiona alte genuri ale filmului,
dar numai ca informatii descriptive.
"""
        elif method == "Matrix Factorization":
            explanation_guidance = """
Explica faptul ca filmul este recomandat
pe baza tiparelor identificate in evaluari.

Nu atribui recomandarea genurilor filmului.

Nu pretinde ca ai identificat utilizatori
concreti cu gusturi similare.

Nu interpreta factorii latenti.
"""
        elif method == "Genre Model":
            explanation_guidance = """
Explica recomandarea prin preferintele
estimate pentru genurile filmului.

Daca exista genuri comune, mentioneaza-le.

Nu inventa evaluari concrete ale utilizatorului.

Nu afirma ca genurile neincluse in lista
preferintelor sunt neapreciate.
"""
        else:
            explanation_guidance = """
Explica recomandarea prin combinarea
tiparelor din evaluari cu informatiile
despre preferintele pentru genuri.

Daca exista genuri comune, poti mentiona
aceasta potrivire.

Nu afirma ca potrivirea genurilor este
singurul motiv al recomandarii.

Nu inventa contributii numerice ale modelelor.
"""

        prompt = f"""
Scrie o explicatie naturala pentru o persoana
care foloseste o aplicatie de recomandare de filme.

Intrebarea la care raspunzi este:
"De ce imi recomandati acest film?"

INFORMATII DISPONIBILE:

Titlu: {movie_title}

Genurile filmului:
{genres_text}

Genurile asociate preferintelor utilizatorului:
{preferences_text}

Genuri comune:
{common_genres_text}

Alte genuri ale filmului:
{other_genres_text}

Metoda interna:
{method}

Mecanismul recomandarii:
{method_description}

{score_description}

INSTRUCTIUNI SPECIFICE:
{explanation_guidance}

REGULI OBLIGATORII:

1. Scrie exclusiv in limba romana.

2. Adreseaza-te direct utilizatorului,
la persoana a doua singular.

3. Scrie 1-2 propozitii clare si naturale.

4. Incepe cu:
"Iti recomandam filmul ..."

5. Foloseste doar informatiile furnizate.

6. Explica motivele reale pentru care
filmul a fost selectat de sistem.

7. Nu inventa informatii despre poveste,
actori, regizor, premii, atmosfera,
calitate artistica sau originalitate.

8. Nu transforma simpla prezenta a unor
genuri intr-o apreciere subiectiva.

9. Nu folosi formulari precum:
"aduce un plus de originalitate",
"ofera o experienta captivanta",
"imbina perfect genurile",
"este un film exceptional",
decat daca exista dovezi explicite.

10. Nu afirma ca filmul are evaluari bune
sau este popular fara date care sa
sustina aceste afirmatii.

11. Nu inventa comportamente ale
utilizatorului sau filme vizionate.

12. Nu afirma ca utilizatorul va aprecia
cu siguranta filmul.

13. Nu mentiona scoruri numerice.

14. Nu mentiona denumirile tehnice
ale metodelor de recomandare.

15. Nu folosi jargon precum:
factori latenti, scor euristic,
regularizare, ponderi sau algoritm.

16. Nu inventa relatii cauzale intre
genuri si recomandare daca metoda
nu foloseste direct genurile.

17. Daca mentionezi genuri suplimentare,
prezinta-le neutru, fara sa le atribui
automat valoare sau calitate.

18. Evita formularile repetitive,
exagerate sau promotionale.

19. Nu folosi expresii precum
"sistemul estimeaza" sau
"recomandarea se bazeaza pe metoda".

20. Nu adauga titluri, liste,
introduceri sau concluzii.

Returneaza exclusiv explicatia finala.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Esti un asistent pentru o aplicatie "
                        "de recomandare de filme. "
                        "Scrii explicatii scurte, naturale "
                        "si adresate direct utilizatorului. "
                        "Respecti strict datele primite. "
                        "Nu inventezi caracteristici ale "
                        "filmelor, motive de recomandare "
                        "sau preferinte ale utilizatorului. "
                        "Eviti limbajul tehnic si "
                        "afirmatiile subiective nejustificate."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=800
        )

        explanation = response.choices[0].message.content

        if not explanation or not explanation.strip():
            raise RuntimeError(
                "LLM-ul nu a returnat o explicatie. "
                "Verifica limita de tokenuri sau "
                "raspunsul modelului."
            )

        return explanation.strip()
