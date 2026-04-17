import streamlit as st
import json
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests
from ddgs import DDGS
from newspaper import Article
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -----------------------------
# Sources presse française
# -----------------------------
FRENCH_NEWS_DOMAINS = [

# centre / généralistes
"lemonde.fr",
"francetvinfo.fr",
"ouest-france.fr",

# centre droit
"lefigaro.fr",
"lesechos.fr",

# gauche
"liberation.fr",
"nouvelobs.com",

# droite
"valeursactuelles.com",

# droite radicale / extrême droite
"fdesouche.com",
"ripostelaique.com",
"boulevardvoltaire.fr",
"egaliteetreconciliation.fr",
"reseauinternational.net",

# international
"france24.com",
"rfi.fr"
]

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from streamlit_mic_recorder import speech_to_text
    MICRO_AVAILABLE = True
except Exception:
    speech_to_text = None
    MICRO_AVAILABLE = False


# -----------------------------
# Configuration page
# -----------------------------
st.set_page_config(
    page_title="DOXA Detector",
    page_icon="🧠",
    layout="wide",
)

st.image("banner2.png", use_container_width=True)
st.caption("Laboratoire de calibration cognitive — M = (G + N) − D")
st.markdown("---")

st.markdown("""
<style>
div[data-testid="stProgressBar"] > div > div > div > div {
    height: 20px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Textes FR uniques
# -----------------------------
T = {
    "settings": "Réglages",
    "load_example": "Charger l'exemple",
    "show_method": "Afficher la méthode",
    "hard_fact_score_scale": "Échelle du Hard Fact Score",
    "scale_0_5": "très fragile",
    "scale_6_9": "douteux",
    "scale_10_14": "plausible mais à recouper",
    "scale_15_20": "structurellement robuste",
    "topic_section": "Analyse de plusieurs articles par sujet",
    "topic": "Sujet à analyser",
    "topic_placeholder": "ex. : intelligence artificielle",
    "analyze_topic": "📰 Analyser 10 articles sur ce sujet",
    "searching": "Recherche et analyse des articles en cours...",
    "articles_analyzed": "articles analysés.",
    "analyzed_articles": "Articles analysés",
    "avg_hard_fact": "Moyenne Hard Fact",
    "avg_classic_score": "Moyenne score classique",
    "topic_doxa_index": "Indice de doxa du sujet",
    "high": "Élevé",
    "medium": "Moyen",
    "low": "Faible",
    "credibility_score_dispersion": "Dispersion des scores de crédibilité",
    "article_label": "Article",
    "no_exploitable_articles_found": "Aucun article exploitable trouvé pour ce sujet.",
    "enter_keyword_first": "Entrez d'abord un mot-clé ou un sujet.",
    "url": "Analyser un article par URL",
    "load_url": "🌐 Charger l'article depuis l'URL",
    "article_loaded_from_url": "Article chargé depuis l'URL.",
    "unable_to_retrieve_text": "Impossible de récupérer le texte de cette URL.",
    "paste_url_first": "Collez d'abord une URL.",
    "paste": "Collez ici un article ou un texte",
    "analyze": "🔍 Analyser l'article",
    "manual_paste": "copier-coller manuel",
    "loaded_url_source": "article chargé par URL",
    "text_source": "Source du texte",
    "paste_text_or_load_url": "Collez un texte ou chargez une URL, puis cliquez sur « 🔍 Analyser l'article ».",
    "classic_score": "Score classique",
    "improved_score": "Score amélioré",
    "hard_fact_score": "Hard Fact Score",
    "help_classic_score": "M = (G + N) − D",
    "help_improved_score": "Ajout de V et pénalité R",
    "help_hard_fact_score": "Contrôle plus dur des affirmations et des sources",
    "credibility_gauge": "Jauge de crédibilité",
    "fragile": "Fragile",
    "fragile_message": "Le texte présente de fortes fragilités structurelles ou factuelles.",
    "doubtful": "Douteux",
    "doubtful_message": "Le texte contient quelques éléments crédibles, mais reste très incertain.",
    "plausible": "Plausible",
    "plausible_message": "Le texte paraît globalement plausible, mais demande encore vérification.",
    "robust": "Robuste",
    "robust_message": "Le texte présente une base structurelle et factuelle plutôt solide.",
    "score": "Score",
    "verdict": "Verdict",
    "summary": "Résumé de l'analyse",
    "strengths_detected": "Forces détectées",
    "few_strong_signals": "Peu de signaux forts repérés.",
    "weaknesses_detected": "Fragilités détectées",
    "no_major_weakness": "Aucune fragilité majeure repérée par l'heuristique.",
    "presence_of_source_markers": "Présence de marqueurs de sources ou de données",
    "verifiability_clues": "Indices de vérifiabilité repérés : liens, chiffres, dates ou pourcentages",
    "text_contains_nuances": "Le texte contient des nuances, limites ou contrepoints",
    "text_evokes_robust_sources": "Le texte évoque des sources potentiellement robustes ou institutionnelles",
    "some_claims_verifiable": "Certaines affirmations sont assez bien ancrées pour être vérifiées proprement",
    "overly_assertive_language": "Langage trop assuré ou absolutiste",
    "notable_emotional_sensational_charge": "Charge émotionnelle ou sensationnaliste notable",
    "almost_total_absence_of_verifiable_elements": "Absence quasi totale d'éléments vérifiables",
    "text_too_short": "Texte trop court pour soutenir sérieusement une affirmation forte",
    "multiple_claims_very_fragile": "Plusieurs affirmations centrales sont très fragiles au regard des indices présents",
    "hard_fact_checking_by_claim": "Fact-checking des affirmations",
    "claim": "Affirmation",
    "status": "Statut",
    "verifiability": "Vérifiabilité",
    "risk": "Risque",
    "number": "Nombre",
    "date": "Date",
    "named_entity": "Nom propre",
    "attributed_source": "Source attribuée",
    "yes": "Oui",
    "no": "Non",
    "to_verify": "À vérifier",
    "rather_verifiable": "Plutôt vérifiable",
    "very_fragile": "Très fragile",
    "low_credibility": "Crédibilité basse",
    "prudent_credibility": "Crédibilité prudente",
    "rather_credible": "Plutôt crédible",
    "strong_credibility": "Crédibilité forte",
    "paste_longer_text": "Collez un texte un peu plus long pour obtenir une cartographie fine des affirmations.",
    "llm_analysis": "Analyse de mécroyance pour systèmes",
    "llm_intro": "Cette section applique les modèles dérivés du traité pour évaluer la posture cognitive d'un système.",
    "overconfidence": "Surconfiance (asymétrie)",
    "calibration": "Calibration relative (ratio)",
    "revisability": "Révisabilité (R)",
    "cognitive_closure": "Clôture cognitive",
    "interpretation": "Interprétation",
    "llm_metrics": "Métriques dérivées",
    "zone_closure": "Zone de clôture cognitive : la certitude excède l’ancrage cognitif.",
    "zone_stability": "Zone de stabilité révisable : la mécroyance accompagne sans dominer.",
    "zone_lucidity": "Zone de lucidité croissante : le doute structure la cognition.",
    "zone_rare": "Zone rare : cognition hautement intégrée et réflexive.",
    "zone_pansapience": "Pan-sapience hypothétique : horizon limite d’une cognition presque totalement révisable.",
    "zone_asymptote": "Asymptote idéale : totalité du savoir et de l’intégration, sans rigidification.",
    "out_of_spectrum": "Valeur hors spectre théorique.",
    "external_corroboration_module": "🔎 Module de corroboration externe",
    "external_corroboration_caption": "Ce module cherche des sources externes susceptibles de confirmer, nuancer ou contredire les affirmations centrales du texte collé.",
    "corroboration_in_progress": "Recherche de corroborations en cours...",
    "generated_query": "Requête générée",
    "no_strong_sources_found": "Aucune source suffisamment solide trouvée pour cette affirmation.",
    "no_corroboration_found": "Aucune corroboration exploitable trouvée.",
    "corroborated": "Corroborée",
    "mixed": "Mitigée",
    "not_corroborated": "Non corroborée",
    "insufficiently_documented": "Insuffisamment documentée",
    "corroboration_verdict": "Verdict de corroboration",
    "match_score": "Score de correspondance",
    "contradiction_signal": "Signal de contradiction",
    "detected": "Détecté",
    "not_detected": "Non détecté",
    "ai_module": "Module IA",
    "ai_module_caption": "L’IA relit l’analyse heuristique et formule une lecture critique plus synthétique.",
    "generate_ai_analysis": "✨ Générer l’analyse IA",
    "ai_unavailable": "Module IA indisponible : clé OpenAI absente ou bibliothèque non installée.",
    "ai_analysis_result": "Analyse IA",
    "method": "Méthode",
    "original_formula": "Formule originelle",
    "articulated_knowledge_density": "G : densité de savoir articulé — sources, chiffres, noms, références, traces vérifiables.",
    "integration": "N : intégration — contexte, nuances, réserves, cohérence argumentative.",
    "assertive_rigidity": "D : rigidité assertive — certitudes non soutenues, emballement rhétorique.",
    "disclaimer": "Cette app ne remplace ni un journaliste, ni un chercheur, ni un greffier du réel. Mais elle retire déjà quelques masques au texte qui parade.",
}


# -----------------------------
# Triangle cognitif 3D
# -----------------------------
def plot_cognitive_triangle_3d(G: float, N: float, D: float):
    G_pt = [10, 0, 0]
    N_pt = [0, 10, 0]
    D_pt = [0, 0, 10]
    P = [G, N, D]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    verts = [[G_pt, N_pt, D_pt]]
    tri = Poly3DCollection(verts, alpha=0.18, edgecolor="black", linewidths=1.5)
    ax.add_collection3d(tri)

    ax.plot([G_pt[0], N_pt[0]], [G_pt[1], N_pt[1]], [G_pt[2], N_pt[2]], linewidth=2)
    ax.plot([N_pt[0], D_pt[0]], [N_pt[1], D_pt[1]], [N_pt[2], D_pt[2]], linewidth=2)
    ax.plot([D_pt[0], G_pt[0]], [D_pt[1], G_pt[1]], [D_pt[2], G_pt[2]], linewidth=2)

    ax.scatter(*G_pt, s=80)
    ax.scatter(*N_pt, s=80)
    ax.scatter(*D_pt, s=80)

    ax.text(G_pt[0] + 0.3, G_pt[1], G_pt[2], "G", fontsize=12, weight="bold")
    ax.text(N_pt[0], N_pt[1] + 0.3, N_pt[2], "N", fontsize=12, weight="bold")
    ax.text(D_pt[0], D_pt[1], D_pt[2] + 0.3, "D", fontsize=12, weight="bold")

    ax.scatter(*P, s=140, marker="o")
    ax.text(P[0] + 0.2, P[1] + 0.2, P[2] + 0.2, "Texte", fontsize=11, weight="bold")

    ax.plot([0, G], [0, 0], [0, 0], linestyle="--", linewidth=1)
    ax.plot([0, 0], [0, N], [0, 0], linestyle="--", linewidth=1)
    ax.plot([0, 0], [0, 0], [0, D], linestyle="--", linewidth=1)
    ax.plot([0, G], [0, N], [0, D], linestyle=":", linewidth=1.5)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)

    ax.set_xlabel("G — gnōsis")
    ax.set_ylabel("N — nous")
    ax.set_zlabel("D — doxa")
    ax.set_title("Triangle cognitif 3D")
    ax.view_init(elev=24, azim=35)

    return fig


# -----------------------------
# OpenAI client
# -----------------------------
def get_openai_client() -> Optional["OpenAI"]:
    if OpenAI is None:
        return None

    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


client = get_openai_client()


# -----------------------------
# Header
# -----------------------------
st.title("DOXA Detector")

with st.container(border=True):
    st.subheader("Analyser la fiabilité d’un texte")
    st.write(
        "DOXA Detector aide à comprendre si un texte repose sur des faits solides "
        "ou sur une rhétorique persuasive."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1️⃣ Coller un texte")
        st.write("Copiez un article ou un extrait dans la zone d’analyse.")

    with col2:
        st.markdown("### 2️⃣ Analyser")
        st.write("L’application examine les sources, les affirmations et la nuance.")

    with col3:
        st.markdown("### 3️⃣ Comprendre")
        st.write("Obtenez un score de crédibilité et une analyse des affirmations.")

    st.caption(
        "Cet outil n’affirme pas si un texte est vrai ou faux : "
        "il aide simplement à mieux comprendre la solidité de l’information."
    )


# -----------------------------
# Modèle de cognition
# -----------------------------
class Cognition:
    def __init__(self, gnosis: float, nous: float, doxa: float):
        self.G = self.clamp(gnosis)
        self.N = self.clamp(nous)
        self.D = self.clamp(doxa)
        self.M = self.compute_mecroyance()

    @staticmethod
    def clamp(value: float, min_val: float = 0.0, max_val: float = 10.0) -> float:
        return max(min_val, min(max_val, value))

    def compute_mecroyance(self) -> float:
        return (self.G + self.N) - self.D

    def interpret(self) -> str:
        m = self.M
        if m < 0:
            return T["zone_closure"]
        if 0 <= m <= 10:
            return T["zone_stability"]
        if 10 < m <= 17:
            return T["zone_lucidity"]
        if 17 < m < 19:
            return T["zone_rare"]
        if 19 <= m < 20:
            return T["zone_pansapience"]
        if m == 20:
            return T["zone_asymptote"]
        return T["out_of_spectrum"]


# -----------------------------
# Exemple
# -----------------------------
SAMPLE_ARTICLE = (
    "L'intelligence artificielle va remplacer 80% des emplois d'ici 2030, selon une étude choc publiée hier par le cabinet GlobalTech. "
    "Le rapport de 45 pages affirme que les secteurs de la finance et de la santé seront les plus touchés. "
    "\"C'est une révolution sans précédent\", déclare Jean Dupont, expert en robotique. "
    "Cependant, certains économistes comme Marie Curie restent prudents : \"Il faut nuancer ces chiffres, car de nouveaux métiers vont apparaître.\" "
    "L'étude précise que 12 millions de postes pourraient être créés en Europe. "
    "Malgré cela, l'inquiétude grandit chez les salariés qui craignent pour leur avenir. "
    "Il est absolument certain que nous allons vers une crise sociale majeure si rien n'est fait immédiatement."
)



# -----------------------------
# Helpers
# -----------------------------
def clamp(n: float, minn: float, maxn: float) -> float:
    return max(min(maxn, n), minn)


def compute_linguistic_suspicion(text: str) -> dict:
    """
    Amplificateur linguistique simple pour le mensonge brut.
    Retourne un facteur L entre 1.0 et 2.0 environ.
    """
    if not text:
        return {
            "L": 1.0,
            "rhetorical_pressure": 0,
            "absolute_claims": 0,
            "vague_authority": 0,
            "dramatic_framing": 0,
            "lack_of_nuance": 0,
            "trigger_count": 0,
        }

    t = text.lower()

    rhetorical_pressure_terms = [
        "clearly", "obviously", "without doubt", "there is no doubt",
        "the truth is", "everyone knows", "it is certain", "undeniable",
        "il est évident", "sans aucun doute", "la vérité est",
        "tout le monde sait", "il est certain", "indéniable"
    ]

    absolute_claim_terms = [
        "always", "never", "everyone", "nobody", "all", "none",
        "toujours", "jamais", "tout le monde", "personne", "tous", "aucun"
    ]

    vague_authority_terms = [
        "experts say", "sources say", "insiders say", "many specialists",
        "according to sources", "internal sources", "reports confirm",
        "les experts disent", "des sources affirment", "selon des sources",
        "des spécialistes", "des rapports confirment", "sources internes"
    ]

    dramatic_framing_terms = [
        "shocking truth", "what they don't want you to know", "unbelievable",
        "hidden truth", "explosive revelation", "scandalous",
        "vérité choquante", "ce qu'on ne veut pas que vous sachiez",
        "incroyable", "vérité cachée", "révélation explosive", "scandaleux"
    ]

    nuance_terms = [
        "may", "might", "could", "perhaps", "possibly", "suggests", "appears",
        "peut", "pourrait", "peut-être", "possiblement", "semble", "suggère"
    ]

    def count_hits(terms):
        return sum(1 for term in terms if term in t)

    rhetorical_pressure = count_hits(rhetorical_pressure_terms)
    absolute_claims = count_hits(absolute_claim_terms)
    vague_authority = count_hits(vague_authority_terms)
    dramatic_framing = count_hits(dramatic_framing_terms)
    nuance_hits = count_hits(nuance_terms)

    lack_of_nuance = 2 if nuance_hits == 0 else 1 if nuance_hits <= 2 else 0

    raw_score = (
        rhetorical_pressure
        + absolute_claims
        + vague_authority
        + dramatic_framing
        + lack_of_nuance
    )

    L = 1.0 + min(raw_score / 8.0, 1.0)

    return {
        "L": round(L, 3),
        "rhetorical_pressure": rhetorical_pressure,
        "absolute_claims": absolute_claims,
        "vague_authority": vague_authority,
        "dramatic_framing": dramatic_framing,
        "lack_of_nuance": lack_of_nuance,
        "trigger_count": raw_score,
    }


# -----------------------------
# Bibliothèques rhétoriques
# -----------------------------

CERTITUDE_PERFORMATIVE = [
    "il est évident",
    "il est clair",
    "une chose est claire",
    "les faits sont clairs",
    "les faits parlent d'eux-mêmes",
    "personne ne peut nier",
    "nul ne peut nier",
    "il ne fait aucun doute",
    "sans aucun doute",
    "cela ne fait aucun doute",
    "chacun peut voir",
    "tout le monde voit bien",
    "la réalité est simple",
    "la vérité est simple",
    "il est absolument certain",
    "c'est une certitude",
    "il est certain",
    "c'est incontestable",
    "c'est indiscutable",
    "c'est incontestablement vrai",
    "clearly",
    "it is clear",
    "it is obvious",
    "there is no doubt",
    "without any doubt",
    "facts speak for themselves",
    "nobody can deny"
]

AUTORITE_VAGUE = [
    "selon des experts",
    "de nombreux experts",
    "des experts confirment",
    "les experts disent",
    "les spécialistes s'accordent",
    "des spécialistes l'affirment",
    "plusieurs analystes pensent",
    "les observateurs constatent",
    "selon certains spécialistes",
    "des sources indiquent",
    "selon des sources",
    "des sources proches du dossier",
    "des sources internes",
    "des rapports suggèrent",
    "des rapports confirment",
    "les études montrent",
    "des études montrent",
    "les chiffres montrent",
    "les indicateurs montrent",
    "tout indique que",
    "il ressort que",
    "according to sources",
    "experts say",
    "specialists confirm",
    "many analysts believe",
    "reports confirm",
    "studies show"
]

DRAMATISATION = [
    "crise majeure",
    "crise historique",
    "crise sans précédent",
    "catastrophe imminente",
    "catastrophe annoncée",
    "menace historique",
    "menace grave",
    "danger immense",
    "danger majeur",
    "situation explosive",
    "situation critique",
    "tournant historique",
    "moment décisif",
    "moment historique",
    "choc politique",
    "effondrement",
    "effondrement annoncé",
    "chaos",
    "dérive grave",
    "urgence nationale",
    "révolution sans précédent",
    "historic threat",
    "major collapse",
    "unprecedented crisis",
    "critical situation",
    "historic turning point"
]

GENERALISATION = [
    "tout le monde sait",
    "tout le monde voit bien",
    "tout le monde comprend",
    "chacun sait",
    "chacun comprend",
    "les français savent",
    "les citoyens pensent",
    "les gens comprennent",
    "les gens savent",
    "les familles voient bien",
    "personne n'ignore",
    "nul n'ignore",
    "nous savons tous",
    "on sait tous que",
    "everyone knows",
    "everyone understands",
    "people know",
    "ordinary people see it",
    "nobody ignores"
]

NATURALISATION = [
    "il n'y a pas d'alternative",
    "il n'existe pas d'alternative",
    "c'est la seule solution",
    "c'est la seule voie",
    "c'est inévitable",
    "c'est nécessaire",
    "nous devons agir",
    "nous devons continuer",
    "nous n'avons pas le choix",
    "il faut poursuivre",
    "il faut aller plus loin",
    "il faut avancer",
    "il faut tenir bon",
    "il faut continuer sur cette voie",
    "la seule politique possible",
    "la seule réponse possible",
    "no alternative",
    "there is no alternative",
    "the only solution",
    "the only path",
    "it is unavoidable",
    "necessary reform"
]

ENNEMI_ABSTRAIT = [
    "certaines forces",
    "des forces hostiles",
    "des intérêts puissants",
    "certains groupes",
    "des groupes d'influence",
    "des acteurs étrangers",
    "des puissances extérieures",
    "les élites",
    "les technocrates",
    "les lobbies",
    "les réseaux",
    "ceux qui veulent affaiblir le pays",
    "ceux qui refusent le changement",
    "les ennemis de l'intérieur",
    "les ennemis du peuple",
    "hostile forces",
    "external actors",
    "powerful interests",
    "hidden networks",
    "enemies within"
]

VICTIMISATION = [
    "on nous empêche d'agir",
    "on veut nous faire taire",
    "on refuse d'entendre le peuple",
    "le peuple est abandonné",
    "les français sont abandonnés",
    "nous sommes attaqués",
    "nous sommes affaiblis",
    "nous sommes pénalisés",
    "le pays est sacrifié",
    "nos efforts sont méprisés",
    "ordinary people are ignored",
    "the people have been abandoned",
    "we are being silenced",
    "we are under attack"
]

MORALISATION = [
    "c'est une question de responsabilité",
    "c'est notre devoir",
    "nous avons le devoir",
    "nous devons être à la hauteur",
    "ce serait irresponsable",
    "il serait irresponsable",
    "notre devoir moral",
    "nous n'avons pas le droit d'échouer",
    "nous devons protéger nos enfants",
    "nous devons défendre l'avenir",
    "it is our duty",
    "it would be irresponsible",
    "we must protect our children"
]

URGENCE = [
    "il faut agir maintenant",
    "il faut agir immédiatement",
    "sans attendre",
    "avant qu'il ne soit trop tard",
    "il est encore temps",
    "nous devons agir vite",
    "immédiatement",
    "dès maintenant",
    "urgence absolue",
    "time is running out",
    "we must act now",
    "before it is too late",
    "immediately"
]

PROMESSE_EXCESSIVE = [
    "nous allons tout changer",
    "nous allons changer la vie",
    "nous allons redresser le pays",
    "nous allons sauver l'économie",
    "nous allons protéger tout le monde",
    "nous garantirons l'avenir",
    "nous garantirons la prospérité",
    "nous garantirons la sécurité",
    "nous allons rétablir l'ordre",
    "we will fix everything",
    "we will restore prosperity",
    "we will guarantee security"
]

POPULISME_ANTI_ELITE = [
    "les élites ont trahi",
    "les élites méprisent le peuple",
    "le peuple contre les élites",
    "les puissants contre le peuple",
    "les technocrates",
    "les bureaucrates de bruxelles",
    "la caste",
    "l'oligarchie",
    "les élites mondialisées",
    "le système est verrouillé",
    "ceux d'en haut",
    "la finance décide de tout",
    "les banques gouvernent",
    "les marchés imposent leur loi",
    "ordinary people versus the elite",
    "the elite has failed",
    "the establishment betrayed the people",
    "the system is rigged"
]

PROGRESSISME_IDENTITAIRE = [
    "les dominations systémiques",
    "la violence systémique",
    "le racisme systémique",
    "les discriminations structurelles",
    "les privilèges invisibles",
    "les privilèges blancs",
    "les privilèges de classe",
    "déconstruire les normes",
    "déconstruire les stéréotypes",
    "remettre en cause les normes",
    "les identités minorisées",
    "les corps minorisés",
    "les personnes marginalisées",
    "les vécus minoritaires",
    "intersection des oppressions",
    "les rapports de domination",
    "check your privilege",
    "systemic oppression",
    "structural discrimination",
    "deconstruct gender norms",
    "marginalized voices",
    "lived experience matters",
    "the personal is political"
]

SOCIALISME_COMMUNISME = [
    "les travailleurs exploités",
    "la lutte des classes",
    "le capital détruit",
    "le capital exploite",
    "les possédants",
    "les exploiteurs",
    "la bourgeoisie",
    "le patronat prédateur",
    "les riches doivent payer",
    "reprendre les richesses",
    "socialiser les moyens de production",
    "redistribuer les richesses",
    "mettre fin au capitalisme",
    "abolir l'exploitation",
    "protéger les services publics contre le marché",
    "workers are exploited",
    "class struggle",
    "the ruling class",
    "end capitalism",
    "redistribute wealth",
    "the wealthy must pay",
    "public ownership"
]

CONFUSION_DELEGITIMATION = [
    "tout populisme est d'extrême droite",
    "le populisme mène toujours au fascisme",
    "toute critique est réactionnaire",
    "toute opposition est haineuse",
    "qui n'est pas avec nous est contre nous",
    "refuser cette réforme c'est refuser le progrès",
    "critiquer cela c'est être raciste",
    "critiquer cela c'est être sexiste",
    "critiquer cela c'est être transphobe",
    "toute réserve est suspecte",
    "there is only one acceptable position",
    "any criticism is hate",
    "if you disagree you are on the wrong side of history"
]

# -----------------------------
# Bibliothèques rhétoriques
# -----------------------------
AUTORITE_ACADEMIQUE_VAGUE = [
    "selon plusieurs études",
    "selon certaines études",
    "selon une étude récente",
    "selon des chercheurs",
    "selon certains chercheurs",
    "selon plusieurs chercheurs",
    "plusieurs études suggèrent",
    "plusieurs travaux suggèrent",
    "les analyses montrent",
    "les analyses suggèrent",
    "les données montrent",
    "les données indiquent",
    "les données disponibles",
    "les recherches montrent",
    "les recherches suggèrent",
    "la littérature scientifique",
    "le consensus scientifique",
    "de nombreux spécialistes",
    "certains spécialistes",
    "de nombreux experts",
    "certains experts",
    "plusieurs experts",
    "de nombreux analystes",
    "plusieurs analystes",
    "the data suggests",
    "available data shows",
    "research suggests",
    "studies suggest",
    "experts agree",
    "many specialists"
]

DILUTION_RESPONSABILITE = [
    "il ne s'agit pas d'accuser",
    "il ne s'agit pas de blâmer",
    "il ne s'agit pas de désigner",
    "personne ne cherche à accuser",
    "il faut simplement reconnaître",
    "il faut seulement reconnaître",
    "il s'agit simplement de constater",
    "il s'agit seulement de constater",
    "il convient de reconnaître",
    "il faut admettre que",
    "il serait naïf d'ignorer",
    "ignorer cette réalité reviendrait à",
    "ce n'est pas une accusation",
    "sans mettre en cause quiconque",
    "sans désigner de coupable",
    "without blaming anyone",
    "this is not about blaming",
    "it is simply necessary to recognize",
    "it would be naive to ignore"
]
CAUSALITE_IMPLICITE = [
    "depuis que",
    "depuis l'introduction de",
    "depuis la mise en place de",
    "depuis l'arrivée de",
    "suite à",
    "à cause de",
    "en raison de",
    "cela a conduit à",
    "cela explique",
    "cela montre que",
    "ce qui prouve que",
    "ce qui démontre que",
    "ce qui explique que",
    "c'est pourquoi",
    "d'où",
    "ce qui entraîne",
    "ce qui conduit à",
    "ce qui provoque",
    "which explains",
    "this proves that",
    "this shows that",
    "this leads to",
]
def detect_political_patterns(text: str):
    """
    Détecte des manœuvres discursives politiques ou rhétoriques
    à partir de bibliothèques d'expressions.
    Retourne :
    - total_score : nombre total d'occurrences détectées
    - results : nombre d'occurrences par catégorie
    - matched_terms : expressions effectivement trouvées
    """
    if not text:
        return 0, {}, {}

    t = text.lower()

    categories = {
        "certitude": CERTITUDE_PERFORMATIVE,
        "autorite": AUTORITE_VAGUE,
        "autorite_academique": AUTORITE_ACADEMIQUE_VAGUE,
        "dramatisation": DRAMATISATION,
        "generalisation": GENERALISATION,
        "naturalisation": NATURALISATION,
        "ennemi": ENNEMI_ABSTRAIT,
        "victimisation": VICTIMISATION,
        "moralisation": MORALISATION,
        "urgence": URGENCE,
        "promesse": PROMESSE_EXCESSIVE,
        "populisme": POPULISME_ANTI_ELITE,
        "progressisme_identitaire": PROGRESSISME_IDENTITAIRE,
        "socialisme_communisme": SOCIALISME_COMMUNISME,
        "delegitimation": CONFUSION_DELEGITIMATION,
        "dilution": DILUTION_RESPONSABILITE,
    }

    results = {}
    matched_terms = {}
    total_score = 0

    for name, terms in categories.items():
        hits = [term for term in terms if term in t]
        results[name] = len(hits)
        matched_terms[name] = hits
        total_score += len(hits)

    return total_score, results, matched_terms


def compute_rhetorical_pressure(results: dict) -> float:
    """
    Calcule une pression rhétorique pondérée entre 0.0 et 1.0
    à partir des catégories détectées.
    """
    weights = {
        "certitude": 1.2,
        "autorite": 1.0,
        "dramatisation": 1.3,
        "generalisation": 1.1,
        "naturalisation": 1.4,
        "ennemi": 1.5,
    }

    weighted_score = 0.0

    for cat, count in results.items():
        weighted_score += count * weights.get(cat, 1.0)

    return min(weighted_score / 10, 1.0)


def interpret_rhetorical_pressure(value: float):
    """
    Traduit la pression rhétorique en étiquette + couleur.
    """
    if value < 0.20:
        return "Faible", "#16a34a"
    elif value < 0.40:
        return "Modérée", "#ca8a04"
    elif value < 0.70:
        return "Élevée", "#f97316"
    else:
        return "Très élevée", "#dc2626"
def compute_propaganda_gauge(
    lie_gauge: float,
    rhetorical_pressure: float,
    political_pattern_score: int,
    closure: float
) -> float:
    """
    Jauge propagandiste globale entre 0 et 1.
    Combine :
    - tension cognitive
    - pression rhétorique
    - motifs politiques/idéologiques détectés
    - fermeture cognitive
    """
    pattern_factor = min(political_pattern_score / 8, 1.0)
    closure_factor = min(closure / 1.2, 1.0)

    score = (
        0.30 * lie_gauge +
        0.35 * rhetorical_pressure +
        0.20 * pattern_factor +
        0.15 * closure_factor
    )

    return min(max(score, 0.0), 1.0)


def interpret_propaganda_gauge(value: float):
    """
    Traduit l'indice propagandiste en étiquette + couleur + commentaire.
    """
    if value < 0.20:
        return "Très faible", "#16a34a", "Le texte ne présente pas de structure propagandiste marquée."
    elif value < 0.40:
        return "Faible", "#84cc16", "Le discours peut orienter légèrement la perception, sans verrouillage fort."
    elif value < 0.60:
        return "Modéré", "#ca8a04", "Le texte contient plusieurs éléments compatibles avec une mise en orientation du lecteur."
    elif value < 0.80:
        return "Élevé", "#f97316", "Le discours semble fortement orienté et cherche à imposer un cadrage interprétatif."
    else:
        return "Très élevé", "#dc2626", "Le texte présente une structure fortement propagandiste ou de verrouillage idéologique."       

def interpret_discursive_profile(
    lie_gauge: float,
    rhetorical_pressure: float,
    propaganda_gauge: float
) -> str:
    """
    Verdict global basé sur les 3 jauges discursives.
    """
    if propaganda_gauge >= 0.75 and rhetorical_pressure >= 0.60:
        return "Structure discursive fortement propagandiste"
    elif lie_gauge >= 0.65 and rhetorical_pressure >= 0.45:
        return "Structure discursive manipulatoire probable"
    elif propaganda_gauge >= 0.45 or rhetorical_pressure >= 0.45:
        return "Discours fortement orienté"
    elif lie_gauge < 0.40 and rhetorical_pressure < 0.35:
        return "Discours plutôt sincère ou peu verrouillant"
    else:
        return "Discours ambigu ou mixte"

def interpret_closure_gauge(value: float):
    """
    Traduit la clôture cognitive en étiquette + couleur + commentaire.
    """
    if value < 0.40:
        return "Ouverture cognitive", "#16a34a", "Le texte reste assez révisable."
    elif value < 0.75:
        return "Rigidification modérée", "#ca8a04", "Le discours commence à se refermer sur ses certitudes."
    elif value < 1.10:
        return "Clôture élevée", "#f97316", "La certitude domine nettement l’ancrage cognitif."
    else:
        return "Clôture critique", "#dc2626", "Le texte semble fortement verrouillé par sa propre structure."


def render_custom_gauge(value: float, color: str):
    value = max(0.0, min(1.0, value))
    st.markdown(f"""
    <div style="width:100%; margin-top:10px; margin-bottom:10px;">
        <div style="
            width:100%;
            height:26px;
            background:#e5e7eb;
            border-radius:12px;
            overflow:hidden;
            border:1px solid #cbd5e1;
        ">
            <div style="
                width:{value*100}%;
                height:100%;
                background:{color};
                transition:width 0.4s ease;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)       

@st.cache_data(show_spinner=False, ttl=3600)
def extract_article_from_url(url: str) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception:
        return ""


@st.cache_data(show_spinner=False, ttl=1800)
def search_articles_by_keyword(keyword: str, max_results: int = 10) -> List[Dict]:
    articles = []
    seen_urls = set()

    api_key = st.secrets.get("NEWS_API_KEY")
    from_date_iso = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    # -----------------------------
    # 1) Priorité : NewsAPI
    # -----------------------------
    if api_key:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": keyword,
            "language": "fr",
            "sortBy": "publishedAt",
            "pageSize": max_results * 3,
            "apiKey": api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                for art in data.get("articles", []):
                    article_url = art.get("url")
                    title = art.get("title", "Sans titre")
                    source = art.get("source", {}).get("name", "Source inconnue")
                    published_at = art.get("publishedAt", "")

                    if not article_url or article_url in seen_urls:
                        continue

                    seen_urls.add(article_url)

                    articles.append({
                        "title": title,
                        "url": article_url,
                        "source": source,
                        "published_at": published_at,
                    })

                    if len(articles) >= max_results:
                        return articles

        except Exception as e:
            st.warning(f"Erreur NewsAPI : {e}")

    # -----------------------------
    # 2) Fallback DDGS
    # -----------------------------
    try:
        with DDGS() as ddgs:
            query = f"{keyword} actualité France"
            results = list(ddgs.text(query, max_results=max_results * 5))

            for r in results:
                url = r.get("href", "")
                title = r.get("title", "Sans titre")

                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)

                articles.append({
                    "title": title,
                    "url": url,
                    "source": url.split("/")[2] if "://" in url else url,
                    "published_at": "",
                })

                if len(articles) >= max_results:
                    break

    except Exception as e:
        st.warning(f"Erreur DDGS : {e}")

    return articles
# -----------------------------
# Jauge mécroyance / mensonge
# -----------------------------
def compute_lie_gauge(M: float, ME: float):
    delta = ME - M
    amp = 8.0
    strength = min(abs(delta) / amp, 1.0)

    if delta <= 0:
        gauge = 0.5 * (1 - strength)

        if gauge > 0.35:
            label = "Mécroyance modérée"
            color = "#ca8a04"
        else:
            label = "Mécroyance forte"
            color = "#a16207"
    else:
        gauge = 0.5 + (0.5 * strength)

        if gauge < 0.65:
            label = "Mensonge possible"
            color = "#f97316"
        elif gauge < 0.85:
            label = "Mensonge probable"
            color = "#dc2626"
        else:
            label = "Mensonge extrême"
            color = "#991b1b"

    return round(gauge, 3), label, color, round(ME, 2)
    
@dataclass
class Claim:
    text: str
    has_number: bool
    has_date: bool
    has_named_entity: bool
    has_source_cue: bool
    absolutism: int
    emotional_charge: int
    verifiability: float
    risk: float
    status: str


SOURCE_CUES = [
    "selon", "affirme", "déclare", "rapport", "étude", "expert",
    "source", "dit", "écrit", "publié", "annonce", "confirme", "révèle",
]

ABSOLUTIST_WORDS = [
    "toujours", "jamais", "absolument", "certain", "certaine",
    "prouvé", "prouvée", "incontestable", "tous", "aucun",
]

EMOTIONAL_WORDS = [
    "choc", "incroyable", "terrible", "peur", "menace",
    "scandale", "révolution", "urgent", "catastrophe", "crise",
]

NUANCE_MARKERS = [
    "cependant", "pourtant", "néanmoins", "toutefois", "mais",
    "nuancer", "prudence", "possible", "peut-être", "semble",
]

CERTITUDE_PERFORMATIVE = [
    "il est évident",
    "il est clair que",
    "sans aucun doute",
    "il est absolument certain",
    "les faits sont clairs",
    "personne ne peut nier",
    "la réalité est simple",
    "clearly",
    "it is obvious",
    "without any doubt",
    "there is no doubt"
]

AUTORITE_VAGUE = [
    "selon des experts",
    "des sources indiquent",
    "selon certains spécialistes",
    "plusieurs analystes pensent",
    "des rapports suggèrent",
    "according to sources",
    "experts say",
    "insiders say",
    "many specialists"
]

DRAMATISATION = [
    "crise majeure",
    "catastrophe imminente",
    "menace historique",
    "situation explosive",
    "choc politique",
    "crise sans précédent",
    "unprecedented crisis",
    "historic threat",
    "major collapse"
]

GENERALISATION = [
    "tout le monde sait",
    "les citoyens pensent",
    "les gens comprennent",
    "les Français savent",
    "everyone knows",
    "people understand",
    "everyone realizes"
]

NATURALISATION = [
    "il n'y a pas d'alternative",
    "c'est la seule solution",
    "c'est inévitable",
    "nous devons agir",
    "unavoidable",
    "necessary reform",
    "no alternative"
]

ENNEMI_ABSTRAIT = [
    "certaines forces",
    "des intérêts puissants",
    "certains groupes",
    "des acteurs étrangers",
    "hostile forces",
    "external actors"
]

def analyze_claim(sentence: str) -> Claim:
    has_number = bool(re.search(r"\d+", sentence))
    has_date = bool(
        re.search(
            r"\d{4}|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre",
            sentence,
            re.I,
        )
    )
    has_named_entity = bool(re.search(r"[A-Z][a-z]+ [A-Z][a-z]+|[A-Z]{2,}", sentence))
    has_source_cue = any(cue in sentence.lower() for cue in SOURCE_CUES)

    absolutism = sum(1 for word in ABSOLUTIST_WORDS if word in sentence.lower())
    emotional_charge = sum(1 for word in EMOTIONAL_WORDS if word in sentence.lower())

    v_score = clamp((has_number * 5) + (has_date * 5) + (has_named_entity * 5) + (has_source_cue * 5), 0, 20)
    r_score = clamp((absolutism * 7) + (emotional_charge * 7), 0, 20)

    if v_score < 5:
        status = T["very_fragile"]
    elif v_score < 12:
        status = T["to_verify"]
    else:
        status = T["rather_verifiable"]

    return Claim(
        text=sentence,
        has_number=has_number,
        has_date=has_date,
        has_named_entity=has_named_entity,
        has_source_cue=has_source_cue,
        absolutism=absolutism,
        emotional_charge=emotional_charge,
        verifiability=v_score,
        risk=r_score,
        status=status,
    )


def analyze_article(text: str) -> Dict:
    words = text.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 10]
    article_length = len(words)

    source_markers = len(re.findall(r"|".join(re.escape(c) for c in SOURCE_CUES), text.lower()))
    citation_like = len(re.findall(r'"|\'|«|»', text))
    nuance_markers = len(re.findall(r"|".join(re.escape(c) for c in NUANCE_MARKERS), text.lower()))

    G = clamp(source_markers * 1.5 + citation_like * 0.5, 0, 10)
    N = clamp(nuance_markers * 2 + (article_length / 100), 0, 10)

    certainty = len(re.findall(r"certain|absolument|prouvé|évident|incontestable", text.lower()))
    emotional = len(re.findall(r"|".join(re.escape(w) for w in EMOTIONAL_WORDS), text.lower()))

    D = clamp(certainty * 2 + emotional * 1.5, 0, 10)
    M = round((G + N) - D, 1)
    V = clamp(G * 0.8 + N * 0.2, 0, 10)
    R = clamp(D * 0.7 + (emotional * 1.2), 0, 10)
    improved = round((G + N + V) - (D + R), 1)

    claims = [analyze_claim(s) for s in sentences[:15]]
    avg_claim_verifiability = sum(c.verifiability for c in claims) / len(claims) if claims else 0
    avg_claim_risk = sum(c.risk for c in claims) / len(claims) if claims else 0
    source_quality = clamp(source_markers * 3 - (emotional * 2), 0, 20)

    red_flags = []
    if D > 8:
        red_flags.append("Doxa saturée")
    if emotional > 5:
        red_flags.append("Pathos excessif")
    if G < 2:
        red_flags.append("Désert documentaire")
    if article_length < 50:
        red_flags.append("Format indigent")

    hard_fact_score_raw = (
        (0.18 * G + 0.12 * N + 0.20 * V + 0.22 * source_quality + 0.18 * avg_claim_verifiability)
        - (0.16 * D + 0.12 * R + 0.18 * avg_claim_risk + 0.9 * len(red_flags))
    )
    hard_fact_score = round(clamp(hard_fact_score_raw + 8, 0, 20), 1)

    if hard_fact_score < 6:
        verdict = T["low_credibility"]
    elif hard_fact_score < 10:
        verdict = T["prudent_credibility"]
    elif hard_fact_score < 15:
        verdict = T["rather_credible"]
    else:
        verdict = T["strong_credibility"]

    strengths = []
    if source_markers >= 2:
        strengths.append(T["presence_of_source_markers"])
    if citation_like >= 2:
        strengths.append(T["verifiability_clues"])
    if nuance_markers >= 2:
        strengths.append(T["text_contains_nuances"])
    if source_quality >= 12:
        strengths.append(T["text_evokes_robust_sources"])
    if any(c.status == T["rather_verifiable"] for c in claims):
        strengths.append(T["some_claims_verifiable"])

    weaknesses = []
    if certainty >= 3:
        weaknesses.append(T["overly_assertive_language"])
    if emotional >= 2:
        weaknesses.append(T["notable_emotional_sensational_charge"])
    if source_markers == 0 and citation_like == 0:
        weaknesses.append(T["almost_total_absence_of_verifiable_elements"])
    if article_length < 80:
        weaknesses.append(T["text_too_short"])
    weaknesses.extend(red_flags)
    if sum(1 for c in claims if c.status == T["very_fragile"]) >= 2:
        weaknesses.append(T["multiple_claims_very_fragile"])

    if emotional >= 2:
        weaknesses.append(T["notable_emotional_sensational_charge"])
    if source_markers == 0 and citation_like == 0:
        weaknesses.append(T["almost_total_absence_of_verifiable_elements"])
    if article_length < 80:
        weaknesses.append(T["text_too_short"])
    weaknesses.extend(red_flags)
    if sum(1 for c in claims if c.status == T["very_fragile"]) >= 2:
        weaknesses.append(T["multiple_claims_very_fragile"])

    ling = compute_linguistic_suspicion(text)
    L = ling["L"]

    political_pattern_score, political_results, matched_terms = detect_political_patterns(text)
    rhetorical_pressure = compute_rhetorical_pressure(political_results)

    ME_base = max(0, (2 * D) - (G + N))
    ME = round(ME_base * L, 2)

    return {
        "words": len(words),
        "sentences": len(sentences),
        "G": G,
        "N": N,
        "D": D,
        "M": M,
        "ME_base": ME_base,
        "ME": ME,
        "L": L,
        
        "linguistic_trigger_count": ling["trigger_count"],
        "linguistic_pressure_hits": ling["rhetorical_pressure"],
        "absolute_claims": ling["absolute_claims"],
        "vague_authority": ling["vague_authority"],
        "dramatic_framing": ling["dramatic_framing"],
        "lack_of_nuance": ling["lack_of_nuance"],
        "political_pattern_score": political_pattern_score,
        "political_results": political_results,
        "matched_terms": matched_terms,
        "rhetorical_pressure": rhetorical_pressure,
        "V": V,
        "R": R,
        "improved": improved,
        "source_quality": source_quality,
        "avg_claim_risk": avg_claim_risk,
        "avg_claim_verifiability": avg_claim_verifiability,
        "hard_fact_score": hard_fact_score,
        "verdict": verdict,
        "profil_solidite": verdict,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "claims": claims,
        "red_flags": red_flags,
    }


# -----------------------------
# Corroboration
# -----------------------------
def extract_key_sentences_for_corroboration(text: str, max_sentences: int = 5) -> List[str]:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 40]
    scored = []
    for s in sentences:
        score = 0
        if re.search(r"\d+", s):
            score += 2
        if re.search(r"\d{4}|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre", s, re.I):
            score += 2
        if re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}", s):
            score += 2
        if any(word in s.lower() for word in [
            "selon", "affirme", "déclare", "rapport", "étude",
            "expert", "source", "publié", "annonce", "confirme", "révèle"
        ]):
            score += 1
        if any(word in s.lower() for word in [
            "absolument", "certain", "jamais", "toujours",
            "incontestable", "choc", "scandale", "révolution", "urgent"
        ]):
            score += 1
        scored.append((score, s))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [s for _, s in scored[:max_sentences]]


def build_search_query_from_claim(claim: str) -> str:
    claim = re.sub(r"[^\w\s%\-]", " ", claim)
    claim = re.sub(r"\s+", " ", claim).strip()
    words = claim.split()
    important_words = [w for w in words if len(w) > 3][:12]
    return " ".join(important_words)


def extract_claim_features(claim: str) -> Dict:
    numbers = re.findall(r"\d+(?:[.,]\d+)?%?", claim)
    years = re.findall(r"\b(?:19|20)\d{2}\b", claim)
    proper_names = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}", claim)
    words = re.findall(r"\b\w+\b", claim.lower())
    stopwords = {
        "les", "des", "une", "dans", "avec", "pour", "être", "sont", "mais",
        "plus", "comme", "nous", "vous", "sur", "par", "est", "ont", "aux",
        "du", "de", "la", "le", "un", "et", "ou", "en", "à", "au", "ce",
        "ces", "ses", "son", "sa", "qui", "que", "quoi", "dont", "ainsi", "alors",
    }
    keywords = [w for w in words if len(w) > 4 and w not in stopwords]
    return {
        "numbers": list(set(numbers)),
        "years": list(set(years)),
        "proper_names": list(set(proper_names)),
        "keywords": list(dict.fromkeys(keywords))[:12],
    }


def score_match_between_claim_and_result(claim: str, result_text: str) -> Dict:
    features = extract_claim_features(claim)
    rt = result_text.lower()
    number_hits = sum(1 for n in features["numbers"] if n.lower() in rt)
    year_hits = sum(1 for y in features["years"] if y.lower() in rt)
    proper_name_hits = sum(1 for p in features["proper_names"] if p.lower() in rt)
    keyword_hits = sum(1 for k in features["keywords"] if k.lower() in rt)

    score = 0.0
    score += number_hits * 3
    score += year_hits * 2
    score += proper_name_hits * 3
    score += min(keyword_hits, 5) * 1.2

    contradiction_markers = [
        "faux", "trompeur", "incorrect", "inexact",
        "démenti", "réfuté", "aucune preuve",
    ]
    contradiction_signal = any(marker in rt for marker in contradiction_markers)

    return {
        "score": round(score, 1),
        "number_hits": number_hits,
        "year_hits": year_hits,
        "proper_name_hits": proper_name_hits,
        "keyword_hits": keyword_hits,
        "contradiction_signal": contradiction_signal,
    }


def classify_corroboration(matches: List[Dict]) -> str:
    if not matches:
        return "insufficient"

    best_score = max(m["match_score"]["score"] for m in matches)
    contradiction_count = sum(1 for m in matches if m["match_score"]["contradiction_signal"])
    strong_matches = sum(1 for m in matches if m["match_score"]["score"] >= 8)
    medium_matches = sum(1 for m in matches if 4 <= m["match_score"]["score"] < 8)

    if strong_matches >= 2 and contradiction_count == 0:
        return "corroborated"
    if best_score >= 8 and contradiction_count >= 1:
        return "mixed"
    if medium_matches >= 1 or best_score >= 4:
        return "mixed"
    return "not_corroborated"


def display_corroboration_verdict(code: str) -> str:
    if code == "corroborated":
        return f"🟢 {T['corroborated']}"
    if code == "mixed":
        return f"🟠 {T['mixed']}"
    if code == "not_corroborated":
        return f"🔴 {T['not_corroborated']}"
    return f"⚪ {T['insufficiently_documented']}"


def corroborate_claims(text: str, max_claims: int = 5, max_results_per_claim: int = 3) -> List[Dict]:
    claims = extract_key_sentences_for_corroboration(text, max_sentences=max_claims)
    corroboration_results = []

    trusted_domains = [
        "reuters.com", "apnews.com", "bbc.com", "nytimes.com", "theguardian.com",
        "lemonde.fr", "lefigaro.fr", "liberation.fr", "francetvinfo.fr", "lesechos.fr",
        "who.int", "un.org", "worldbank.org", "nature.com", "science.org",
        "elpais.com", "elmundo.es", "dw.com", "spiegel.de",
    ]

    try:
        with DDGS() as ddgs:
            for claim in claims:
                query = build_search_query_from_claim(claim)
                search_results = list(ddgs.text(query, max_results=max_results_per_claim * 5))
                filtered = []
                for r in search_results:
                    url = r.get("href", "")
                    title = r.get("title", "")
                    body = r.get("body", "")
                    combined_text = f"{title} {body}"
                    if any(domain in url for domain in trusted_domains):
                        match_score = score_match_between_claim_and_result(claim, combined_text)
                        filtered.append(
                            {
                                "title": title,
                                "url": url,
                                "snippet": body,
                                "match_score": match_score,
                            }
                        )
                filtered = sorted(filtered, key=lambda x: x["match_score"]["score"], reverse=True)[:max_results_per_claim]
                verdict = classify_corroboration(filtered)
                corroboration_results.append(
                    {
                        "claim": claim,
                        "query": query,
                        "matches": filtered,
                        "verdict": verdict,
                    }
                )
    except Exception as e:
        st.warning(f"Erreur de corroboration : {e}")

    return corroboration_results


# -----------------------------
# IA helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def generate_ai_summary(article_text: str, result: Dict, max_chars: int = 7000) -> str:
    if client is None:
        return ""

    short_text = article_text[:max_chars]
    claims_preview = []
    for c in result.get("claims", [])[:8]:
        claims_preview.append(
            {
                "affirmation": c.text,
                "statut": c.status,
                "verifiabilite": c.verifiability,
                "risque": c.risk,
                "has_number": c.has_number,
                "has_date": c.has_date,
                "has_named_entity": c.has_named_entity,
                "has_source_cue": c.has_source_cue,
            }
        )

    prompt = f"""
Tu es un assistant de lecture critique rigoureux.

Ta tâche :
1. Résumer le profil global de crédibilité du texte.
2. Expliquer la différence entre plausibilité structurelle et robustesse factuelle.
3. Identifier les 3 principales forces.
4. Identifier les 3 principales fragilités.
5. Terminer par un verdict prudent.

Contraintes :
- Sois clair, concis et concret.
- N’invente aucun fait.
- N’affirme pas avec certitude qu’un texte est vrai ou faux sans justification solide.
- Appuie-toi sur les métriques heuristiques ci-dessous.

Analyse heuristique :
{json.dumps({
    'G': result.get('G'),
    'N': result.get('N'),
    'D': result.get('D'),
    'M': result.get('M'),
    'V': result.get('V'),
    'R': result.get('R'),
    'hard_fact_score': result.get('hard_fact_score'),
    'verdict': result.get('verdict'),
    'strengths': result.get('strengths', []),
    'weaknesses': result.get('weaknesses', []),
    'claims': claims_preview,
    'red_flags': result.get('red_flags', []),
}, ensure_ascii=False, indent=2)}

Texte à analyser :
{short_text}
"""

    try:
        response = client.responses.create(model="gpt-4o", input=prompt)
        return response.output_text.strip()
    except Exception as e:
        return f"Erreur IA : {e}"
        
@st.cache_data(show_spinner=False, ttl=1800)
def analyze_multiple_articles(keyword: str, max_results: int = 10) -> List[Dict]:
    articles = search_articles_by_keyword(keyword, max_results)
    results = []

    for art in articles:
        try:
            full_text = extract_article_from_url(art["url"])
            if len(full_text) > 120:
                analysis = analyze_article(full_text)
                results.append(
                    {
                        "Source": art["source"],
                        "Titre": art["title"],
                        "Score classique": analysis["M"],
                        "Hard Fact Score": analysis["hard_fact_score"],
                        "Verdict": analysis["verdict"],
                        "URL": art["url"],
                    }
                )
        except Exception:
            continue

    return results   
    
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_text_for_textarea(url: str) -> str:
    try:
        text = extract_article_from_url(url)
        return (text or "").strip()
    except Exception:
        return ""

# -----------------------------
# Réglages
# -----------------------------
with st.expander(T["settings"], expanded=False):
    use_sample = st.button(T["load_example"])
    show_method = st.toggle(T["show_method"], value=True)
    st.divider()
    st.subheader(T["hard_fact_score_scale"])
    st.markdown(
        f"- **0–5** : {T['scale_0_5']}\n"
        f"- **6–9** : {T['scale_6_9']}\n"
        f"- **10–14** : {T['scale_10_14']}\n"
        f"- **15–20** : {T['scale_15_20']}"
    )

if "article" not in st.session_state:
    st.session_state.article = SAMPLE_ARTICLE
if "article_source" not in st.session_state:
    st.session_state.article_source = "paste"
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_article" not in st.session_state:
    st.session_state.last_article = ""
if "multi_results" not in st.session_state:
    st.session_state.multi_results = []
if "last_keyword" not in st.session_state:
    st.session_state.last_keyword = ""

if use_sample:
    st.session_state.article = SAMPLE_ARTICLE
    st.session_state.article_source = "paste"
    st.session_state.loaded_url = ""


# -----------------------------
# Analyse multi-articles
# -----------------------------
st.subheader(T["topic_section"])
keyword = st.text_input(T["topic"], placeholder=T["topic_placeholder"])

if st.button(T["analyze_topic"], key="analyze_topic"):
    if keyword.strip():
        st.info(T["searching"])
        st.session_state.multi_results = analyze_multiple_articles(keyword.strip(), max_results=10)
        st.session_state.last_keyword = keyword.strip()
    else:
        st.session_state.multi_results = []
        st.warning(T["enter_keyword_first"])

if st.session_state.get("multi_results"):
    df_multi = pd.DataFrame(st.session_state.multi_results).sort_values("Hard Fact Score", ascending=False)

    st.success(f"{len(df_multi)} {T['articles_analyzed']}")

    c1, c2 = st.columns(2)
    c1.metric(T["analyzed_articles"], len(df_multi))
    c2.metric(T["avg_hard_fact"], round(df_multi["Hard Fact Score"].mean(), 1))
    st.metric(T["avg_classic_score"], round(df_multi["Score classique"].mean(), 1))

    ecart_type_hf = df_multi["Hard Fact Score"].std()
    indice_doxa = "high" if ecart_type_hf < 1.5 else ("medium" if ecart_type_hf < 3 else "low")
    st.metric(T["topic_doxa_index"], T[indice_doxa])

    st.subheader(T["credibility_score_dispersion"])
    df_plot = df_multi.copy()
    df_plot["Article"] = [f"{T['article_label']} {i+1}" for i in range(len(df_plot))]
    st.bar_chart(df_plot.set_index("Article")["Hard Fact Score"])
    st.dataframe(df_multi, use_container_width=True, hide_index=True)

    st.markdown("### Actions sur les articles trouvés")

    for i, row in df_multi.reset_index(drop=True).iterrows():
        with st.container(border=True):
            st.markdown(f"### {row['Titre']}")
            st.caption(f"{row['Source']}")

            score = row["Hard Fact Score"]
            if score <= 6:
                color, label = "🔴", "Fragile"
            elif score <= 11:
                color, label = "🟠", "Douteux"
            elif score <= 15:
                color, label = "🟡", "Plausible"
            else:
                color, label = "🟢", "Robuste"

            st.markdown(f"**{color} Score de crédibilité : {score:.1f}/20 — {label}**")
            st.progress(score / 20)

            col1, col2 = st.columns(2)
            with col1:
                st.link_button("🌐 Ouvrir l'article", row["URL"], use_container_width=True)
            with col2:
                if st.button(f"📥 Charger pour analyse", key=f"load_article_{i}"):
                    loaded_text = fetch_text_for_textarea(row["URL"])
                    if loaded_text:
                        st.session_state.article = loaded_text
                        st.session_state.article_source = "url"
                        st.session_state.loaded_url = row["URL"]
                        st.success("Article chargé dans la zone de texte.")
                        st.rerun()
                    else:
                        st.warning("Impossible d'extraire le texte.")
elif st.session_state.get("last_keyword"):
    st.warning(T["no_exploitable_articles_found"])


# -----------------------------
# Chargement URL
# -----------------------------
with st.form("url_form"):
    url = st.text_input(T["url"])
    load_url_submitted = st.form_submit_button(T["load_url"])

if load_url_submitted:
    if url:
        texte = extract_article_from_url(url)
        if texte:
            st.session_state.article = texte
            st.session_state.article_source = "url"
            st.session_state.loaded_url = url
            st.success(T["article_loaded_from_url"])
            st.rerun()
        else:
            st.error(T["unable_to_retrieve_text"])
    else:
        st.warning(T["paste_url_first"])


# -----------------------------
# Zone d’analyse
# -----------------------------
previous_article = st.session_state.article

st.markdown("### Zone d’analyse")

with st.container(border=True):
    st.caption("Collez un texte, chargez une URL, ou dictez directement.")

    if MICRO_AVAILABLE:
        spoken_text = speech_to_text(
            language="fr",
            start_prompt="🎙️ Dicter",
            stop_prompt="⏹️ Stop",
            just_once=True,
            use_container_width=True,
            key="speech_to_text_article"
        )

        if spoken_text:
            st.session_state.article = spoken_text
            st.session_state.article_source = "paste"
            st.success("Texte dicté reçu.")
            st.rerun()
    else:
        st.info("Microphone indisponible sur cette version.")

    with st.form("article_form"):
        article = st.text_area(
            T["paste"],
            key="article",
            height=220,
            label_visibility="collapsed",
            placeholder=T["paste"]
        )
        analyze_submitted = st.form_submit_button(T["analyze"], use_container_width=True)

if article.strip() != previous_article.strip():
    st.session_state.article_source = "paste"

source_label = T["manual_paste"] if st.session_state.get("article_source") == "paste" else T["loaded_url_source"]
st.caption(f"{T['text_source']} : {source_label}")

if st.session_state.get("loaded_url"):
    st.caption(f"URL : {st.session_state.loaded_url}")


# -----------------------------
# Analyse principale
# -----------------------------
if analyze_submitted:
    st.session_state.last_result = analyze_article(article)
    st.session_state.last_article = article

result = st.session_state.last_result
article_for_analysis = st.session_state.last_article

if result:
    col1, col2, col3 = st.columns(3)
    col1.metric(T["classic_score"], result["M"], help=T["help_classic_score"])
    col2.metric(T["improved_score"], result["improved"], help=T["help_improved_score"])
    col3.metric(T["hard_fact_score"], result["hard_fact_score"], help=T["help_hard_fact_score"])

    score = result["hard_fact_score"]
    if score <= 6:
        couleur, etiquette, message = "🔴", T["fragile"], T["fragile_message"]
    elif score <= 11:
        couleur, etiquette, message = "🟠", T["doubtful"], T["doubtful_message"]
    elif score <= 15:
        couleur, etiquette, message = "🟡", T["plausible"], T["plausible_message"]
    else:
        couleur, etiquette, message = "🟢", T["robust"], T["robust_message"]

    st.subheader(f"{couleur} {T['credibility_gauge']} : {etiquette}")
    st.progress(score / 20)
    st.caption(f"{T['score']} : {score}/20 — {message}")
    st.caption("Sur cette échelle, un texte véritablement crédible se situe généralement dans la zone robuste.")

    st.subheader("Diagnostic cognitif")
    life_score = round((result["hard_fact_score"] / 20) * 100, 1)
    mecroyance_bar = max(0.0, min(1.0, (result["M"] + 10) / 30))

    col1, col2 = st.columns(2)
    with col1:
        st.write("Vitalité cognitive")
        st.progress(life_score / 100)
        st.caption(f"{life_score}%")
    with col2:
        st.write("Indice de mécroyance")
        st.progress(mecroyance_bar)
        st.caption(f"M = {result['M']}")

    st.subheader(f"{T['verdict']} : {result['verdict']}")
    st.subheader(T["summary"])

    m1, m2 = st.columns(2)
    m1.metric("G — gnōsis", result["G"])
    m2.metric("N — nous", result["N"])
    m3, m4 = st.columns(2)
    m3.metric("D — doxa", result["D"])
    m4.metric("V — vérifiabilité", result["V"])
    m5, m6 = st.columns(2)
    m5.metric("QS", result["source_quality"])
    m6.metric("RC", round(result["avg_claim_risk"], 1))
    m7, m8 = st.columns(2)
    m7.metric("VC", round(result["avg_claim_verifiability"], 1))
    m8.metric("F", len(result["red_flags"]))

    st.divider()
    st.subheader("Triangle cognitif G-N-D")
    st.caption("Le texte est placé dans l’espace de la cognition : savoir articulé, compréhension intégrée, et certitude assertive.")
    fig_triangle = plot_cognitive_triangle_3d(result["G"], result["N"], result["D"])
    st.pyplot(fig_triangle, use_container_width=True)

    st.subheader("Métriques cognitives")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Indice de mécroyance (M)", round(result["M"], 2))
    with col2:
        st.metric("Indice de mensonge (ME)", round(result["ME"], 2))

    delta_mm = round(result["M"] - result["ME"], 2)
    st.caption(f"Écart cognitif (M − ME) : {delta_mm}")

    if result["M"] > result["ME"] + 1:
        dominant_pattern = "Structure dominante : mécroyance"
    elif result["ME"] > result["M"] + 1:
        dominant_pattern = "Structure dominante : mensonge stratégique"
    else:
        dominant_pattern = "Structure dominante : mixte ou ambiguë"

    st.subheader("Structure cognitive dominante")
    st.write(dominant_pattern)

    if result["ME"] > result["M"] and result["ME"] > 0:
        cognitive_type = "Mensonge stratégique possible"
    elif result["M"] < 0:
        cognitive_type = "Forte mécroyance / clôture cognitive"
    else:
        cognitive_type = "Cognition probablement sincère mais désalignée"

    st.subheader("Interprétation cognitive")
    st.write(cognitive_type)

    if result["M"] - result["ME"] > 3:
        diagnosis = "Structure de mécroyance forte"
    elif result["M"] > result["ME"]:
        diagnosis = "Structure de mécroyance modérée"
    elif abs(result["M"] - result["ME"]) <= 1:
        diagnosis = "Structure cognitive ambiguë"
    else:
        diagnosis = "Tromperie stratégique possible"

    st.subheader("Diagnostic cognitif")
    st.write(diagnosis)

    gauge_value, gauge_label, gauge_color, ME_gauge = compute_lie_gauge(
        result["M"], result["ME"]
    )

    st.write("Tension cognitive (mécroyance vs mensonge)")
    st.caption(
        "Cette jauge indique si le discours relève plutôt d’une erreur sincère "
        "(mécroyance) ou d’une possible manipulation. "
        "Plus la jauge progresse, plus la structure du texte se rapproche du mensonge."
    )

    st.markdown(f"""
    <div style="width:100%; margin-top:10px; margin-bottom:10px;">
        <div style="
            width:100%;
            height:26px;
            background:#e5e7eb;
            border-radius:12px;
            overflow:hidden;
            border:1px solid #cbd5e1;
        ">
            <div style="
                width:{gauge_value*100}%;
                height:100%;
                background:{gauge_color};
                transition:width 0.4s ease;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<b style='color:{gauge_color}'>{gauge_label}</b> — {round(gauge_value*100,1)}%",
        unsafe_allow_html=True
    )

    st.caption("Erreur sincère ⟵⟶ Manipulation probable")

    st.divider()
    st.subheader("Jauge de pression rhétorique")
    st.caption(
        "Cette jauge ne mesure pas un mensonge certain, mais l’intensité des procédés discursifs "
        "susceptibles d’orienter, de verrouiller ou de dramatiser un discours."
    )

    rp = result["rhetorical_pressure"]
    rp_label, rp_color = interpret_rhetorical_pressure(rp)

    st.markdown(f"""
    <div style="width:100%; margin-top:10px; margin-bottom:10px;">
        <div style="
            width:100%;
            height:26px;
            background:#e5e7eb;
            border-radius:12px;
            overflow:hidden;
            border:1px solid #cbd5e1;
        ">
            <div style="
                width:{rp*100}%;
                height:100%;
                background:{rp_color};
                transition:width 0.4s ease;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<b style='color:{rp_color}'>{rp_label}</b> — {round(rp*100, 1)}%",
        unsafe_allow_html=True
    )

    st.caption("Pression rhétorique faible ⟵⟶ Pression rhétorique forte")

    st.divider()
    st.subheader("Jauge propagandiste")
    st.caption(
        "Cette jauge combine la tension cognitive, la pression rhétorique, "
        "les motifs idéologiques détectés et le degré de fermeture cognitive. "
        "Elle aide à estimer si le texte relève d’un simple discours orienté "
        "ou d’une structure plus franchement propagandiste."
    )

    closure_for_discourse = (
        (result["D"] * (1 + len(result["red_flags"]) / 5)) / (result["G"] + result["N"])
        if (result["G"] + result["N"]) > 0 else 10
    )

    propaganda_value = compute_propaganda_gauge(
        lie_gauge=gauge_value,
        rhetorical_pressure=rp,
        political_pattern_score=result["political_pattern_score"],
        closure=closure_for_discourse
    )

    propaganda_label, propaganda_color, propaganda_text = interpret_propaganda_gauge(propaganda_value)

    render_custom_gauge(propaganda_value, propaganda_color)

    st.markdown(
        f"<b style='color:{propaganda_color}'>{propaganda_label}</b> — {round(propaganda_value*100, 1)}%",
        unsafe_allow_html=True
    )

    st.caption("Discours peu orienté ⟵⟶ Structure propagandiste")
    st.caption(propaganda_text)

    discursive_profile = interpret_discursive_profile(
        lie_gauge=gauge_value,
        rhetorical_pressure=rp,
        propaganda_gauge=propaganda_value
    )

    st.subheader("Profil discursif global")
    st.write(discursive_profile)

    with st.expander("Voir les manœuvres discursives détectées", expanded=False):
        if result["political_pattern_score"] == 0:
            st.info("Aucun marqueur rhétorique politique saillant détecté.")
        else:
            st.metric("Score global de manœuvres discursives", result["political_pattern_score"])

            labels = {
                "certitude": "Certitude performative",
                "autorite": "Autorité vague institutionnelle",
                "dramatisation": "Dramatisation politique",
                "generalisation": "Généralisation abusive",
                "naturalisation": "Naturalisation idéologique",
                "ennemi": "Ennemi abstrait",
            }

            for cat, count in result["political_results"].items():
                if count > 0:
                    st.markdown(f"**{labels.get(cat, cat)}** : {count}")
                    st.caption(", ".join(result["matched_terms"][cat]))

    with st.expander(T["strengths_detected"], expanded=True):
        if result["strengths"]:
            for item in result["strengths"]:
                st.success(item)
        else:
            st.info(T["few_strong_signals"])

    with st.expander(T["weaknesses_detected"], expanded=True):
        if result["weaknesses"]:
            for item in result["weaknesses"]:
                st.error(item)
        else:
            st.success(T["no_major_weakness"])

    st.divider()
    st.subheader("Structure cognitive du texte analysé")
    st.info(T["llm_intro"])

    cog = Cognition(result["G"], result["N"], result["D"])
    overconfidence = result["D"] - (result["G"] + result["N"])
    calibration = result["D"] / (result["G"] + result["N"]) if (result["G"] + result["N"]) > 0 else 10
    revisability = (result["G"] + result["N"] + result["V"]) - result["D"]
    closure = (result["D"] * (1 + len(result["red_flags"]) / 5)) / (result["G"] + result["N"]) if (result["G"] + result["N"]) > 0 else 10

    c1, c2 = st.columns(2)
    c1.metric(T["overconfidence"], round(overconfidence, 2))
    c2.metric(T["calibration"], round(calibration, 2))
    c3, c4 = st.columns(2)
    c3.metric(T["revisability"], round(revisability, 2))
    c4.metric(T["cognitive_closure"], round(closure, 2))
    st.divider()
    st.subheader("Jauge de clôture cognitive")

    st.caption(
        "Cette jauge mesure le degré de verrouillage cognitif du texte. "
        "Plus elle monte, plus la certitude domine le savoir et l’intégration."
    )

    closure_gauge = min(closure / 1.5, 1.0)

    closure_label, closure_color, closure_text = interpret_closure_gauge(closure)

    render_custom_gauge(closure_gauge, closure_color)

    st.markdown(
        f"<b style='color:{closure_color}'>{closure_label}</b> — {round(closure,2)}",
        unsafe_allow_html=True
    )

    st.caption("Ouverture cognitive ⟵⟶ Clôture cognitive")

    st.caption(closure_text)
    st.markdown(f"**{T['interpretation']} :** {cog.interpret()}")

    st.subheader(T["hard_fact_checking_by_claim"])
    claims_df = pd.DataFrame(
        [
            {
                T["claim"]: c.text,
                T["status"]: c.status,
                f"{T['verifiability']} /20": c.verifiability,
                f"{T['risk']} /20": c.risk,
                T["number"]: T["yes"] if c.has_number else T["no"],
                T["date"]: T["yes"] if c.has_date else T["no"],
                T["named_entity"]: T["yes"] if c.has_named_entity else T["no"],
                T["attributed_source"]: T["yes"] if c.has_source_cue else T["no"],
            }
            for c in result["claims"]
        ]
    )

    if not claims_df.empty:
        st.dataframe(claims_df, use_container_width=True, hide_index=True)
    else:
        st.info(T["paste_longer_text"])

    st.divider()
    st.subheader(T["ai_module"])
    st.caption(T["ai_module_caption"])

    if client is None:
        st.warning(T["ai_unavailable"])
    else:
        if st.button(T["generate_ai_analysis"], key="generate_ai_analysis"):
            with st.spinner("Analyse IA en cours..."):
                ai_summary = generate_ai_summary(article_for_analysis, result)
            st.subheader(T["ai_analysis_result"])
            st.markdown(ai_summary)

    if st.session_state.get("article_source") == "paste":
        st.divider()
        st.subheader(T["external_corroboration_module"])
        st.caption(T["external_corroboration_caption"])
        with st.spinner(T["corroboration_in_progress"]):
            corroboration = corroborate_claims(article_for_analysis, max_claims=5, max_results_per_claim=3)
        if corroboration:
            for i, item in enumerate(corroboration, start=1):
                title_preview = item["claim"][:140] + ("..." if len(item["claim"]) > 140 else "")
                with st.expander(f"{T['claim']} {i} : {title_preview}", expanded=(i == 1)):
                    st.markdown(f"**{T['corroboration_verdict']} :** {display_corroboration_verdict(item['verdict'])}")
                    st.markdown(f"**{T['generated_query']} :** `{item['query']}`")
                    if item["matches"]:
                        for match in item["matches"]:
                            st.markdown(f"**[{match['title']}]({match['url']})**")
                            st.markdown(
                                f"- **{T['match_score']}** : {match['match_score']['score']}\n"
                                f"- **{T['contradiction_signal']}** : {T['detected'] if match['match_score']['contradiction_signal'] else T['not_detected']}"
                            )
                            if match["snippet"]:
                                st.caption(match["snippet"])
                    else:
                        st.warning(T["no_strong_sources_found"])
        else:
            st.info(T["no_corroboration_found"])
else:
    st.info(T["paste_text_or_load_url"])

# -----------------------------
# Méthode
# -----------------------------
if show_method:
    st.subheader(T["method"])
    st.markdown(
        f"### {T['original_formula']}\n"
        f"`M = (G + N) − D`\n"
        f"- {T['articulated_knowledge_density']}\n"
        f"- {T['integration']}\n"
        f"- {T['assertive_rigidity']}\n\n"
        f"### {T['llm_metrics']}\n"
        f"- **{T['overconfidence']}** : `D - (G + N)`\n"
        f"- **{T['calibration']}** : `D / (G + N)`\n"
        f"- **{T['revisability']}** : `(G + N + V) - D`\n"
        f"- **{T['cognitive_closure']}** : `(D * S) / (G + N)`\n\n"
        f"{T['disclaimer']}"
    )


# -----------------------------
# Laboratoire interactif
# -----------------------------
st.divider()
st.subheader("Laboratoire interactif de la mécroyance")
st.caption(
    "Expérimentez la formule cognitive : M = (G + N) − D. "
    "Modifiez les paramètres pour observer l’évolution des stades cognitifs."
)

g_game = st.slider("G — gnōsis (savoir articulé)", 0.0, 10.0, 5.0, 0.5)
n_game = st.slider("N — nous (intégration vécue)", 0.0, 10.0, 5.0, 0.5)
d_game = st.slider("D — doxa (certitude / saturation)", 0.0, 10.0, 5.0, 0.5)

m_game = round((g_game + n_game) - d_game, 1)

st.markdown(
    f"""
    <div style="
        background:#f1f5f9;
        border-radius:14px;
        padding:18px;
        margin-top:10px;
        border:1px solid #dbe3ec;
        text-align:center;
        font-size:1.3rem;
        font-weight:700;
    ">
        M = ({g_game:.1f} + {n_game:.1f}) − {d_game:.1f} =
        <span style="color:#0b6e4f;">{m_game:.1f}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if m_game < 0:
    stage = "Fermeture cognitive"
    explanation = "La certitude dépasse la compréhension : la pensée se verrouille."
    percent = 10
elif m_game <= 4:
    stage = "Enfance cognitive"
    explanation = "Structure cognitive naissante, encore fragile."
    percent = 25
elif m_game <= 10:
    stage = "Adolescence cognitive"
    explanation = "Cognition stable mais encore agitée."
    percent = 50
elif m_game <= 17:
    stage = "Maturité cognitive"
    explanation = "Équilibre entre savoir, expérience et doute."
    percent = 75
elif m_game < 19:
    stage = "Sagesse structurelle"
    explanation = "État rare d’équilibre cognitif."
    percent = 90
else:
    stage = "Asymptote de vérité"
    explanation = "Horizon théorique de cohérence maximale."
    percent = 100

st.markdown(f"**Stade actuel : {stage}**")
st.progress(percent / 100)
st.caption(f"M = {m_game} — {explanation}")

st.markdown("### Évolution cognitive")

stages = [
    ("Fermeture", -10, 0),
    ("Enfance", 0, 4.1),
    ("Adolescence", 4.1, 10.1),
    ("Maturité", 10.1, 17.1),
    ("Sagesse", 17.1, 19.1),
    ("Asymptote", 19.1, 21),
]

cols = st.columns(len(stages))
for i, (name, low, high) in enumerate(stages):
    active = low <= m_game < high
    with cols[i]:
        if active:
            st.success(name)
        else:
            st.info(name)

st.caption("Lorsque G et N augmentent sans inflation de D, la cognition gagne en revisabilité.")
