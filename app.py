
import streamlit as st

from recommender import MovieRecommender
from data_utils import GENRES
from llm_explainer import LLMExplainer


st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

@st.cache_resource
def load_recommender():
    return MovieRecommender()


recommender = load_recommender()

def get_movie_genres(item_id):
    """
    Extrage genurile unui film din MovieLens.
    """
    movie_rows = recommender.movies[
        recommender.movies["item_id"] == int(item_id)
    ]

    if movie_rows.empty:
        return []

    movie = movie_rows.iloc[0]

    return [
        genre
        for genre in GENRES
        if genre != "unknown"
        and genre in movie.index
        and movie[genre] == 1
    ]


def get_user_preferred_genres(user_id, top_k=3):
    """
    Extrage genurile cu cele mai mari scoruri
    din profilul utilizatorului.
    """
    profile = recommender.profiles.get(
        int(user_id), {}
    )

    if not profile:
        return []

    valid_genres = [
        genre
        for genre in profile
        if genre in GENRES
        and genre != "unknown"
    ]

    return sorted(
        valid_genres,
        key=lambda genre: profile[genre],
        reverse=True
    )[:top_k]


def generate_explanation(
    movie_title,
    movie_genres,
    score,
    preferred_genres,
    method
):
    """
    Genereaza explicatia folosind Groq.
    """
    explainer = LLMExplainer()

    return explainer.explain(
        movie_title=movie_title,
        movie_genres=movie_genres,
        predicted_score=float(score),
        preferred_genres=preferred_genres,
        method=method
    )


def clear_existing_results():
    st.session_state.existing_recommendations = None
    st.session_state.existing_explanation = None
    st.session_state.existing_context = None


def clear_new_results():
    st.session_state.new_recommendations = None
    st.session_state.new_explanation = None
    st.session_state.new_context = None


def clear_existing_explanation():
    st.session_state.existing_explanation = None


def clear_new_explanation():
    st.session_state.new_explanation = None


defaults = {
    "existing_recommendations": None,
    "existing_explanation": None,
    "existing_context": None,
    "new_recommendations": None,
    "new_explanation": None,
    "new_context": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("Movie Recommendation System")

st.write(
    "Sistem de recomandare bazat pe factorizare "
    "matriciala, preferinte pentru genuri, "
    "filtrare hibrida fixa si adaptiva, "
    "cu explicatii generate prin Groq LLM."
)

tab1, tab2 = st.tabs([
    "Utilizator existent",
    "Utilizator nou"
])

with tab1:

    st.subheader("Recomandari personalizate")

    user_ids = sorted(
        recommender.mf.known_users
    )

    user_id = st.selectbox(
        "Selecteaza utilizatorul",
        user_ids,
        key="existing_user",
        on_change=clear_existing_results
    )

    method = st.selectbox(
        "Metoda de recomandare",
        [
            "Hybrid",
            "Hybrid Adaptive",
            "Matrix Factorization",
            "Genre Model"
        ],
        key="existing_method",
        on_change=clear_existing_results
    )

    n = st.slider(
        "Numar recomandari",
        min_value=5,
        max_value=20,
        value=10,
        key="existing_count",
        on_change=clear_existing_results
    )

    if method == "Hybrid":
        st.caption(
            "Hybrid Fixed: combina Matrix Factorization "
            "si Genre Model folosind o pondere fixa."
        )

    elif method == "Hybrid Adaptive":
        st.caption(
            "Hybrid Adaptive: ajusteaza ponderea "
            "celor doua modele in functie de numarul "
            "de evaluari ale utilizatorului."
        )

    elif method == "Matrix Factorization":
        st.caption(
            "Model colaborativ bazat pe factorizare "
            "matriciala si factori latenti."
        )

    elif method == "Genre Model":
        st.caption(
            "Model bazat pe preferintele estimate "
            "ale utilizatorului pentru genurile filmelor."
        )

    if st.button(
        "Genereaza recomandari",
        key="existing",
        type="primary"
    ):

        try:
            with st.spinner(
                "Calculam recomandarile..."
            ):
                recommendations = (
                    recommender.recommend_existing(
                        int(user_id),
                        n,
                        method
                    )
                )

            st.session_state.existing_recommendations = (
                recommendations
            )

            st.session_state.existing_context = {
                "user_id": int(user_id),
                "method": method,
                "n": n
            }

            st.session_state.existing_explanation = None

        except Exception as error:
            clear_existing_results()
            st.error(
                "Nu am putut genera recomandarile. "
                f"Detalii: {error}"
            )

    recommendations = (
        st.session_state.existing_recommendations
    )

    context = st.session_state.existing_context

    if (
        recommendations is not None
        and context is not None
    ):

        st.divider()

        st.subheader(
            f"Top {len(recommendations)} recomandari "
            f"pentru utilizatorul {context['user_id']}"
        )

        st.write(
            f"Metoda utilizata: {context['method']}"
        )

        if recommendations.empty:

            st.warning(
                "Nu au fost gasite recomandari."
            )

        else:

            st.dataframe(
                recommendations[
                    ["title", "score"]
                ].rename(columns={
                    "title": "Film",
                    "score": "Rating estimat (1-5)"
                }),
                hide_index=True,
                use_container_width=True
            )

            st.caption(
                "Scorurile sunt predictii ale modelului "
                "selectat, nu ratinguri reale."
            )

            st.divider()

            st.subheader(
                "Explica recomandarea cu AI"
            )

            st.write(
                "Selecteaza unul dintre filmele recomandate "
                "pentru a obtine o explicatie in limba romana."
            )

            selected_index = st.selectbox(
                "Film recomandat",
                options=list(range(len(recommendations))),
                format_func=lambda i: (
                    f"{recommendations.iloc[i]['title']} "
                    f"({recommendations.iloc[i]['score']:.2f}/5)"
                ),
                key="existing_movie_selection",
                on_change=clear_existing_explanation
            )

            if st.button(
                "Explica recomandarea",
                key="explain_existing"
            ):

                selected_movie = (
                    recommendations.iloc[selected_index]
                )

                movie_title = str(
                    selected_movie["title"]
                )

                item_id = int(
                    selected_movie["item_id"]
                )

                predicted_score = float(
                    selected_movie["score"]
                )

                movie_genres = get_movie_genres(
                    item_id
                )

                preferred_genres = (
                    get_user_preferred_genres(
                        context["user_id"]
                    )
                )

                try:

                    with st.spinner(
                        "Groq genereaza explicatia..."
                    ):

                        explanation = generate_explanation(
                            movie_title=movie_title,
                            movie_genres=movie_genres,
                            score=predicted_score,
                            preferred_genres=preferred_genres,
                            method=context["method"]
                        )

                    st.session_state.existing_explanation = (
                        explanation
                    )

                except Exception as error:

                    st.session_state.existing_explanation = None

                    st.error(
                        "Nu am putut genera explicatia. "
                        f"Detalii: {error}"
                    )

            if st.session_state.existing_explanation:

                st.info(
                    st.session_state.existing_explanation
                )

                st.caption(
                    "Explicatie generata de LLM folosind "
                    "informatiile disponibile despre film, "
                    "utilizator si metoda de recomandare."
                )


with tab2:

    st.subheader(
        "Recomandari pentru utilizatori noi"
    )

    st.write(
        "Pentru utilizatorii fara istoric de evaluari, "
        "sistemul foloseste genurile selectate si "
        "ratingurile medii regularizate ale filmelor."
    )

    selected_genres = st.multiselect(
        "Ce genuri de filme preferi?",
        [
            genre
            for genre in GENRES
            if genre != "unknown"
        ],
        key="new_genres",
        on_change=clear_new_results
    )

    n_new = st.slider(
        "Numar recomandari",
        min_value=5,
        max_value=20,
        value=10,
        key="new_count",
        on_change=clear_new_results
    )

    if st.button(
        "Recomanda filme",
        key="new",
        type="primary"
    ):

        if not selected_genres:

            clear_new_results()

            st.warning(
                "Selecteaza cel putin un gen."
            )

        else:

            try:

                with st.spinner(
                    "Calculam recomandarile..."
                ):

                    recommendations = (
                        recommender.recommend_new(
                            selected_genres,
                            n_new
                        )
                    )

                st.session_state.new_recommendations = (
                    recommendations
                )

                st.session_state.new_context = {
                    "genres": list(selected_genres),
                    "n": n_new
                }

                st.session_state.new_explanation = None

            except Exception as error:

                clear_new_results()

                st.error(
                    "Nu am putut genera recomandarile. "
                    f"Detalii: {error}"
                )

    recommendations_new = (
        st.session_state.new_recommendations
    )

    context_new = st.session_state.new_context

    if (
        recommendations_new is not None
        and context_new is not None
    ):

        st.divider()

        st.subheader(
            f"Top {len(recommendations_new)} recomandari"
        )

        if recommendations_new.empty:

            st.warning(
                "Nu au fost gasite recomandari."
            )

        else:

            st.dataframe(
                recommendations_new[
                    [
                        "title",
                        "genre_match",
                        "movie_mean",
                        "score"
                    ]
                ].rename(columns={
                    "title": "Film",
                    "genre_match": "Potrivire genuri",
                    "movie_mean": "Rating mediu regularizat",
                    "score": "Scor recomandare (0-1)"
                }),
                hide_index=True,
                use_container_width=True
            )

            st.caption(
                "Scorul de recomandare (0-1) reflecta potrivirea "
                "cu genurile selectate si evaluarile primite de film. "
                "Un scor mai mare indica o recomandare mai bine clasata, "
                "nu probabilitatea ca filmul sa iti placa."
            )

            st.divider()

            st.subheader(
                "Explica recomandarea cu AI"
            )

            st.write(
                "Afla de ce filmul a fost recomandat "
                "pe baza preferintelor declarate."
            )

            selected_new_index = st.selectbox(
                "Film recomandat",
                options=list(
                    range(len(recommendations_new))
                ),
                format_func=lambda i: (
                    f"{recommendations_new.iloc[i]['title']} "
                    f"(scor "
                    f"{recommendations_new.iloc[i]['score']:.3f})"
                ),
                key="new_movie_selection",
                on_change=clear_new_explanation
            )

            if st.button(
                "Explica recomandarea",
                key="explain_new"
            ):

                selected_movie = (
                    recommendations_new.iloc[
                        selected_new_index
                    ]
                )

                movie_title = str(
                    selected_movie["title"]
                )

                item_id = int(
                    selected_movie["item_id"]
                )

                score = float(
                    selected_movie["score"]
                )

                movie_genres = get_movie_genres(
                    item_id
                )

                try:

                    with st.spinner(
                        "Groq genereaza explicatia..."
                    ):

                        explanation = generate_explanation(
                            movie_title=movie_title,
                            movie_genres=movie_genres,
                            score=score,
                            preferred_genres=context_new["genres"],
                            method="Cold Start"
                        )

                    st.session_state.new_explanation = (
                        explanation
                    )

                except Exception as error:

                    st.session_state.new_explanation = None

                    st.error(
                        "Nu am putut genera explicatia. "
                        f"Detalii: {error}"
                    )

            if st.session_state.new_explanation:

                st.info(
                    st.session_state.new_explanation
                )


st.divider()

st.caption(
    "Movie Recommendation System | "
    "Matrix Factorization | Genre Model | "
    "Hybrid Fixed | Hybrid Adaptive | Groq LLM"
)
