# -----------------------------
# Bannière professionnelle
# -----------------------------
import streamlit as st
st.set_page_config(
    page_title="Mécroyance Lab — Fact-checking",
    page_icon="🧠",
    layout="wide",
)

st.image("banner2.png", use_container_width=True)

st.title("DOXA DETECTOR")
st.caption("Laboratoire de calibration cognitive — M = (G + N) − D")

st.markdown("---")

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests
from ddgs import DDGS
from newspaper import Article
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from streamlit_mic_recorder import speech_to_text
def plot_cognitive_triangle_3d(G: float, N: float, D: float):
    """
    Triangle cognitif 3D
    G = gnōsis (savoir articulé)
    N = nous (compréhension intégrée)
    D = doxa (certitude assertive)

    Les valeurs sont attendues entre 0 et 10.
    """

    import requests
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Points de base du triangle
    G_pt = [10, 0, 0]
    N_pt = [0, 10, 0]
    D_pt = [0, 0, 10]

    # Point analysé
    P = [G, N, D]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Triangle principal
    verts = [[G_pt, N_pt, D_pt]]
    tri = Poly3DCollection(verts, alpha=0.18, edgecolor="black", linewidths=1.5)
    ax.add_collection3d(tri)

    # Arêtes du triangle
    ax.plot(
        [G_pt[0], N_pt[0]], [G_pt[1], N_pt[1]], [G_pt[2], N_pt[2]],
        linewidth=2
    )
    ax.plot(
        [N_pt[0], D_pt[0]], [N_pt[1], D_pt[1]], [N_pt[2], D_pt[2]],
        linewidth=2
    )
    ax.plot(
        [D_pt[0], G_pt[0]], [D_pt[1], G_pt[1]], [D_pt[2], G_pt[2]],
        linewidth=2
    )

    # Sommets
    ax.scatter(*G_pt, s=80)
    ax.scatter(*N_pt, s=80)
    ax.scatter(*D_pt, s=80)

    ax.text(G_pt[0] + 0.3, G_pt[1], G_pt[2], "G", fontsize=12, weight="bold")
    ax.text(N_pt[0], N_pt[1] + 0.3, N_pt[2], "N", fontsize=12, weight="bold")
    ax.text(D_pt[0], D_pt[1], D_pt[2] + 0.3, "D", fontsize=12, weight="bold")

    # Point du texte analysé
    ax.scatter(*P, s=140, marker="o")
    ax.text(P[0] + 0.2, P[1] + 0.2, P[2] + 0.2, "Texte", fontsize=11, weight="bold")

    # Projection discrète sur les axes
    ax.plot([0, G], [0, 0], [0, 0], linestyle="--", linewidth=1)
    ax.plot([0, 0], [0, N], [0, 0], linestyle="--", linewidth=1)
    ax.plot([0, 0], [0, 0], [0, D], linestyle="--", linewidth=1)

    # Segment origine -> point
    ax.plot([0, G], [0, N], [0, D], linestyle=":", linewidth=1.5)

    # Limites et labels
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)

    ax.set_xlabel("G — gnōsis")
    ax.set_ylabel("N — nous")
    ax.set_zlabel("D — doxa")

    ax.set_title("Triangle cognitif 3D")

    # Angle de vue
    ax.view_init(elev=24, azim=35)

    return fig

try:
    from openai import OpenAI
except Exception:
    OpenAI = None
    import io
import streamlit as st

def transcribe_audio_with_openai(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    if client is None:
        return ""

    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename  # important pour l'API
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file
        )
        return transcript.text.strip()
    except Exception as e:
        st.error(f"Erreur transcription : {e}")
        return ""

try:
    from streamlit_mic_recorder import speech_to_text
    MICRO_AVAILABLE = True
except Exception:
    MICRO_AVAILABLE = False



# -----------------------------
# OpenAI client
# -----------------------------
def get_openai_client() -> Optional["OpenAI"]:
    if OpenAI is None:
        return None

    api_key = None
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

    if not api_key:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


client = get_openai_client()


# -----------------------------
# Translations
# -----------------------------
translations = {
    "Français": {
        "title": "🧠 Mécroyance Lab — Analyse de crédibilité",
        "intro": "Évaluez la solidité d’un texte, identifiez ses fragilités et examinez la robustesse de ses affirmations.",
        "intro_2": "Mécroyance Lab n’est ni un gadget de vérification ni un simple score automatique. C’est un laboratoire de lecture critique : il cherche moins à bénir un texte qu’à comprendre comment il tient, où il vacille, et jusqu’où il résiste au réel.",
        "language": "Langue / Language",
        "settings": "Réglages",
        "load_example": "Charger l'exemple",
        "show_method": "Afficher la méthode",
        "hard_fact_score_scale": "Échelle du hard fact score",
        "scale_0_5": "très fragile",
        "scale_6_9": "douteux",
        "scale_10_14": "plausible mais à recouper",
        "scale_15_20": "structurellement robuste",
        "topic_section": "Analyse de plusieurs articles par sujet",
        "topic": "Sujet à analyser",
        "topic_placeholder": "ex : intelligence artificielle",
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
        "hard_fact_checking_by_claim": "Fact-checking dur par affirmation",
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
        "llm_analysis": "Analyse de Mécroyance pour LLM",
        "llm_intro": "Cette section applique les modèles dérivés du traité pour évaluer la posture cognitive d'un système (IA ou humain).",
        "overconfidence": "Surconfiance (Asymétrie)",
        "calibration": "Calibration relative (Ratio)",
        "revisability": "Révisabilité (R)",
        "cognitive_closure": "Clôture cognitive",
        "interpretation": "Interprétation",
        "llm_metrics": "Métriques LLM",
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
        "ai_claim_explanations": "Explication IA des affirmations",
        "ai_explain_claim": "Expliquer cette affirmation",
        "ai_explanation": "Explication",
        "method": "Méthode",
        "original_formula": "Formule originelle",
        "articulated_knowledge_density": "G : densité de savoir articulé — sources, chiffres, noms, références, traces vérifiables.",
        "integration": "N : intégration — contexte, nuances, réserves, cohérence argumentative.",
        "assertive_rigidity": "D : rigidité assertive — certitudes non soutenues, emballement rhétorique.",
        "disclaimer": "Cette app ne remplace ni un journaliste, ni un chercheur, ni un greffier du réel. Mais elle retire déjà quelques masques au texte qui parade.",
    },
    "English": {
        "title": "🧠 Mecroyance Lab — Credibility Analyzer",
        "intro": "Evaluate the solidity of a text, identify its weaknesses, and examine the robustness of its claims.",
        "intro_2": "Mecroyance Lab is neither a verification gadget nor a mere automatic score. It is a critical reading lab: less eager to bless a text than to understand how it stands, where it wavers, and how far it resists reality.",
        "language": "Language",
        "settings": "Settings",
        "load_example": "Load example",
        "show_method": "Show method",
        "hard_fact_score_scale": "Hard Fact Score Scale",
        "scale_0_5": "very fragile",
        "scale_6_9": "doubtful",
        "scale_10_14": "plausible but needs cross-checking",
        "scale_15_20": "structurally robust",
        "topic_section": "Analyze multiple articles by topic",
        "topic": "Topic to analyze",
        "topic_placeholder": "e.g. artificial intelligence",
        "analyze_topic": "📰 Analyze 10 articles on this topic",
        "searching": "Searching and analyzing articles...",
        "articles_analyzed": "articles analyzed.",
        "analyzed_articles": "Analyzed articles",
        "avg_hard_fact": "Avg Hard Fact",
        "avg_classic_score": "Avg Classic Score",
        "topic_doxa_index": "Topic Doxa Index",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "credibility_score_dispersion": "Credibility score dispersion",
        "article_label": "Article",
        "no_exploitable_articles_found": "No exploitable articles found for this topic.",
        "enter_keyword_first": "Enter a keyword or topic first.",
        "url": "Analyze article from URL",
        "load_url": "🌐 Load article from URL",
        "article_loaded_from_url": "Article loaded from URL.",
        "unable_to_retrieve_text": "Unable to retrieve text from this URL.",
        "paste_url_first": "Paste a URL first.",
        "paste": "Paste an article or text here",
        "analyze": "🔍 Analyze article",
        "manual_paste": "manual paste",
        "loaded_url_source": "article loaded from URL",
        "text_source": "Text source",
        "paste_text_or_load_url": "Paste a text or load a URL, then click “🔍 Analyze article”.",
        "classic_score": "Classic Score",
        "improved_score": "Improved Score",
        "hard_fact_score": "Hard Fact Score",
        "help_classic_score": "M = (G + N) − D",
        "help_improved_score": "Addition of V and R penalty",
        "help_hard_fact_score": "Stricter control of claims and sources",
        "credibility_gauge": "Credibility Gauge",
        "fragile": "Fragile",
        "fragile_message": "The text shows major structural or factual weaknesses.",
        "doubtful": "Doubtful",
        "doubtful_message": "The text contains some credible elements, but remains highly uncertain.",
        "plausible": "Plausible",
        "plausible_message": "The text appears broadly plausible, but still needs verification.",
        "robust": "Robust",
        "robust_message": "The text presents a fairly solid structural and factual basis.",
        "score": "Score",
        "verdict": "Verdict",
        "summary": "Analysis summary",
        "strengths_detected": "Strengths detected",
        "few_strong_signals": "Few strong signals identified.",
        "weaknesses_detected": "Weaknesses detected",
        "no_major_weakness": "No major weakness identified by heuristics.",
        "presence_of_source_markers": "Presence of source or data markers",
        "verifiability_clues": "Verifiability clues found: links, figures, dates or percentages",
        "text_contains_nuances": "The text contains nuances, limits or counterpoints",
        "text_evokes_robust_sources": "The text evokes potentially robust or institutional sources",
        "some_claims_verifiable": "Some claims are anchored enough to be checked properly",
        "overly_assertive_language": "Overly assertive or absolutist language",
        "notable_emotional_sensational_charge": "Notable emotional or sensational charge",
        "almost_total_absence_of_verifiable_elements": "Almost total absence of verifiable elements",
        "text_too_short": "Text too short to seriously support a strong claim",
        "multiple_claims_very_fragile": "Several central claims are very fragile given the available clues",
        "hard_fact_checking_by_claim": "Hard fact-checking by claim",
        "claim": "Claim",
        "status": "Status",
        "verifiability": "Verifiability",
        "risk": "Risk",
        "number": "Number",
        "date": "Date",
        "named_entity": "Named Entity",
        "attributed_source": "Attributed Source",
        "yes": "Yes",
        "no": "No",
        "to_verify": "To verify",
        "rather_verifiable": "Rather verifiable",
        "very_fragile": "Very fragile",
        "low_credibility": "Low credibility",
        "prudent_credibility": "Prudent credibility",
        "rather_credible": "Rather credible",
        "strong_credibility": "Strong credibility",
        "paste_longer_text": "Paste a slightly longer text to obtain a finer mapping of claims.",
        "llm_analysis": "Mecroyance Analysis for LLM",
        "llm_intro": "This section applies derived models from the treatise to evaluate the cognitive posture of a system (AI or human).",
        "overconfidence": "Overconfidence (Asymmetry)",
        "calibration": "Relative Calibration (Ratio)",
        "revisability": "Revisability (R)",
        "cognitive_closure": "Cognitive Closure",
        "interpretation": "Interpretation",
        "llm_metrics": "LLM Metrics",
        "zone_closure": "Cognitive closure zone: certainty exceeds cognitive anchoring.",
        "zone_stability": "Revisable stability zone: mecroyance accompanies without dominating.",
        "zone_lucidity": "Increasing lucidity zone: doubt structures cognition.",
        "zone_rare": "Rare zone: highly integrated and reflexive cognition.",
        "zone_pansapience": "Hypothetical pan-sapience: limit horizon of an almost totally revisable cognition.",
        "zone_asymptote": "Ideal asymptote: totality of knowledge and integration, without rigidification.",
        "out_of_spectrum": "Value outside the theoretical spectrum.",
        "external_corroboration_module": "🔎 External corroboration module",
        "external_corroboration_caption": "This module looks for external sources likely to confirm, nuance, or contradict the central claims of the pasted text.",
        "corroboration_in_progress": "Searching for corroborations...",
        "generated_query": "Generated query",
        "no_strong_sources_found": "No sufficiently strong source found for this claim.",
        "no_corroboration_found": "No exploitable corroboration found.",
        "corroborated": "Corroborated",
        "mixed": "Mixed",
        "not_corroborated": "Not corroborated",
        "insufficiently_documented": "Insufficiently documented",
        "corroboration_verdict": "Corroboration verdict",
        "match_score": "Match score",
        "contradiction_signal": "Contradiction signal",
        "detected": "Detected",
        "not_detected": "Not detected",
        "ai_module": "AI module",
        "ai_module_caption": "The AI rereads the heuristic analysis and provides a more synthetic critical reading.",
        "generate_ai_analysis": "✨ Generate AI analysis",
        "ai_unavailable": "AI module unavailable: missing OpenAI key or library not installed.",
        "ai_analysis_result": "AI analysis",
        "ai_claim_explanations": "AI explanation of claims",
        "ai_explain_claim": "Explain this claim",
        "ai_explanation": "Explanation",
        "method": "Method",
        "original_formula": "Original Formula",
        "articulated_knowledge_density": "G: articulated knowledge density — sources, figures, names, references, verifiable traces.",
        "integration": "N: integration — context, nuances, reservations, argumentative coherence.",
        "assertive_rigidity": "D: assertive rigidity — unsupported certainties, rhetorical inflation.",
        "disclaimer": "This app does not replace a journalist, a researcher, or a clerk of reality. But it already removes a few masks from the text that parades.",
    },
    "Español": {
        "title": "🧠 Mecroyance Lab — Analizador de Credibilidad",
        "intro": "Evalúe la solidez de un texto, identifique sus fragilidades y examine la robustez de sus afirmaciones.",
        "intro_2": "Mecroyance Lab no es ni un juguete de verificación ni una simple puntuación automática. Es un laboratorio de lectura crítica: busca menos bendecir un texto que entender cómo se sostiene, dónde vacila y hasta qué punto resiste a lo real.",
        "language": "Idioma / Language",
        "settings": "Ajustes",
        "load_example": "Cargar ejemplo",
        "show_method": "Mostrar método",
        "hard_fact_score_scale": "Escala del Hard Fact Score",
        "scale_0_5": "muy frágil",
        "scale_6_9": "dudoso",
        "scale_10_14": "plausible pero necesita contraste",
        "scale_15_20": "estructuralmente robusto",
        "topic_section": "Analizar múltiples artículos por tema",
        "topic": "Tema a analizar",
        "topic_placeholder": "ej.: inteligencia artificial",
        "analyze_topic": "📰 Analizar 10 artículos sobre este tema",
        "searching": "Buscando y analizando artículos...",
        "articles_analyzed": "artículos analizados.",
        "analyzed_articles": "Artículos analizados",
        "avg_hard_fact": "Promedio Hard Fact",
        "avg_classic_score": "Promedio Score Clásico",
        "topic_doxa_index": "Índice de doxa del tema",
        "high": "Alto",
        "medium": "Medio",
        "low": "Bajo",
        "credibility_score_dispersion": "Dispersión de puntuaciones de credibilidad",
        "article_label": "Artículo",
        "no_exploitable_articles_found": "No se encontraron artículos explotables para este tema.",
        "enter_keyword_first": "Introduzca primero una palabra clave o tema.",
        "url": "Analizar artículo por URL",
        "load_url": "🌐 Cargar artículo desde URL",
        "article_loaded_from_url": "Artículo cargado desde URL.",
        "unable_to_retrieve_text": "No se pudo recuperar el texto de esta URL.",
        "paste_url_first": "Pegue primero una URL.",
        "paste": "Pegue aquí un artículo o texto",
        "analyze": "🔍 Analizar artículo",
        "manual_paste": "pegado manual",
        "loaded_url_source": "artículo cargado desde URL",
        "text_source": "Fuente del texto",
        "paste_text_or_load_url": "Pegue un texto o cargue una URL, luego haga clic en “🔍 Analizar artículo”.",
        "classic_score": "Score Clásico",
        "improved_score": "Score Mejorado",
        "hard_fact_score": "Hard Fact Score",
        "help_classic_score": "M = (G + N) − D",
        "help_improved_score": "Adición de V y penalización R",
        "help_hard_fact_score": "Control más estricto de afirmaciones y fuentes",
        "credibility_gauge": "Indicador de credibilidad",
        "fragile": "Frágil",
        "fragile_message": "El texto presenta grandes debilidades estructurales o fácticas.",
        "doubtful": "Dudoso",
        "doubtful_message": "El texto contiene algunos elementos creíbles, pero sigue siendo muy incierto.",
        "plausible": "Plausible",
        "plausible_message": "El texto parece plausible en general, pero aún requiere verificación.",
        "robust": "Robusto",
        "robust_message": "El texto presenta una base estructural y fáctica bastante sólida.",
        "score": "Puntuación",
        "verdict": "Veredicto",
        "summary": "Resumen del análisis",
        "strengths_detected": "Fortalezas detectadas",
        "few_strong_signals": "Pocas señales fuertes detectadas.",
        "weaknesses_detected": "Fragilidades detectadas",
        "no_major_weakness": "No se detectó ninguna gran fragilidad mediante heurísticas.",
        "presence_of_source_markers": "Presencia de marcadores de fuentes o datos",
        "verifiability_clues": "Indicios de verificabilidad detectados: enlaces, cifras, fechas o porcentajes",
        "text_contains_nuances": "El texto contiene matices, límites o contrapuntos",
        "text_evokes_robust_sources": "El texto evoca fuentes potencialmente robustas o institucionales",
        "some_claims_verifiable": "Algunas afirmaciones están lo bastante ancladas para verificarse bien",
        "overly_assertive_language": "Lenguaje demasiado seguro o absolutista",
        "notable_emotional_sensational_charge": "Carga emocional o sensacionalista notable",
        "almost_total_absence_of_verifiable_elements": "Ausencia casi total de elementos verificables",
        "text_too_short": "Texto demasiado corto para sostener seriamente una afirmación fuerte",
        "multiple_claims_very_fragile": "Varias afirmaciones centrales son muy frágiles a la luz de los indicios presentes",
        "hard_fact_checking_by_claim": "Fact-checking duro por afirmación",
        "claim": "Afirmación",
        "status": "Estado",
        "verifiability": "Verificabilidad",
        "risk": "Riesgo",
        "number": "Número",
        "date": "Fecha",
        "named_entity": "Entidad nombrada",
        "attributed_source": "Fuente atribuida",
        "yes": "Sí",
        "no": "No",
        "to_verify": "Por verificar",
        "rather_verifiable": "Bastante verificable",
        "very_fragile": "Muy frágil",
        "low_credibility": "Credibilidad baja",
        "prudent_credibility": "Credibilidad prudente",
        "rather_credible": "Bastante creíble",
        "strong_credibility": "Credibilidad fuerte",
        "paste_longer_text": "Pegue un texto un poco más largo para obtener un mapa más fino de las afirmaciones.",
        "llm_analysis": "Análisis de Mecroyance para LLM",
        "llm_intro": "Esta sección aplica modelos derivados del tratado para evaluar la postura cognitiva de un sistema (IA o humano).",
        "overconfidence": "Sobreconfianza (Asimetría)",
        "calibration": "Calibración relativa (Ratio)",
        "revisability": "Revisabilidad (R)",
        "cognitive_closure": "Cierre cognitivo",
        "interpretation": "Interpretación",
        "llm_metrics": "Métricas LLM",
        "zone_closure": "Zona de cierre cognitivo: la certeza excede el anclaje cognitivo.",
        "zone_stability": "Zona de estabilidad revisable: la mecroyance acompaña sin dominar.",
        "zone_lucidity": "Zona de lucidez creciente: la duda estructura la cognición.",
        "zone_rare": "Zona rara: cognición altamente integrada y reflexiva.",
        "zone_pansapience": "Pan-sapiencia hipotética: horizonte límite de una cognición casi totalmente revisable.",
        "zone_asymptote": "Asíntota ideal: totalidad del saber y de la integración, sin rigidificación.",
        "out_of_spectrum": "Valor fuera del espectro teórico.",
        "external_corroboration_module": "🔎 Módulo de corroboración externa",
        "external_corroboration_caption": "Este módulo busca fuentes externas susceptibles de confirmar, matizar o contradecir las afirmaciones centrales del texto pegado.",
        "corroboration_in_progress": "Buscando corroboraciones...",
        "generated_query": "Consulta generada",
        "no_strong_sources_found": "No se encontró una fuente suficientemente sólida para esta afirmación.",
        "no_corroboration_found": "No se encontró corroboración explotable.",
        "corroborated": "Corroborada",
        "mixed": "Matizada",
        "not_corroborated": "No corroborada",
        "insufficiently_documented": "Insuficientemente documentada",
        "corroboration_verdict": "Veredicto de corroboración",
        "match_score": "Puntuación de coincidencia",
        "contradiction_signal": "Señal de contradicción",
        "detected": "Detectado",
        "not_detected": "No detectado",
        "ai_module": "Módulo de IA",
        "ai_module_caption": "La IA relee el análisis heurístico y formula una lectura crítica más sintética.",
        "generate_ai_analysis": "✨ Generar análisis IA",
        "ai_unavailable": "Módulo IA no disponible: falta la clave OpenAI o la biblioteca no está instalada.",
        "ai_analysis_result": "Análisis IA",
        "ai_claim_explanations": "Explicación IA de las afirmaciones",
        "ai_explain_claim": "Explicar esta afirmación",
        "ai_explanation": "Explicación",
        "method": "Método",
        "original_formula": "Fórmula original",
        "articulated_knowledge_density": "G: densidad de conocimiento articulado — fuentes, cifras, nombres, referencias, huellas verificables.",
        "integration": "N: integración — contexto, matices, reservas, coherencia argumentativa.",
        "assertive_rigidity": "D: rigidez asertiva — certezas no sustentadas, inflación retórica.",
        "disclaimer": "Esta app no reemplaza ni a un periodista, ni a un investigador, ni a un escribano de la realidad. Pero ya arranca algunas máscaras al texto que desfila.",
    },
    "Filipino": {
        "title": "🧠 Mecroyance Lab — Credibility Analyzer",
        "intro": "Suriin ang tibay ng isang teksto, tukuyin ang mga kahinaan nito, at siyasatin ang katatagan ng mga pahayag nito.",
        "intro_2": "Ang Mecroyance Lab ay hindi laruan sa beripikasyon at hindi rin simpleng awtomatikong score. Isa itong laboratoryo ng mapanuring pagbasa: mas mahalaga rito kung paano tumitindig ang teksto, saan ito umuuga, at hanggang saan ito lumalaban sa realidad.",
        "language": "Wika / Language",
        "settings": "Mga Setting",
        "load_example": "I-load ang halimbawa",
        "show_method": "Ipakita ang pamamaraan",
        "hard_fact_score_scale": "Scale ng Hard Fact Score",
        "scale_0_5": "napakarupok",
        "scale_6_9": "kahina-hinala",
        "scale_10_14": "kapani-paniwala ngunit kailangang i-cross-check",
        "scale_15_20": "matibay ang istruktura",
        "topic_section": "Suriin ang maraming artikulo ayon sa paksa",
        "topic": "Paksang susuriin",
        "topic_placeholder": "hal.: artificial intelligence",
        "analyze_topic": "📰 Suriin ang 10 artikulo sa paksang ito",
        "searching": "Hinahanap at sinusuri ang mga artikulo...",
        "articles_analyzed": "mga artikulong nasuri.",
        "analyzed_articles": "Mga nasuring artikulo",
        "avg_hard_fact": "Avg Hard Fact",
        "avg_classic_score": "Avg Classic Score",
        "topic_doxa_index": "Topic Doxa Index",
        "high": "Mataas",
        "medium": "Katamtaman",
        "low": "Mababa",
        "credibility_score_dispersion": "Pagkakaiba-iba ng credibility score",
        "article_label": "Artikulo",
        "no_exploitable_articles_found": "Walang nahanap na magagamit na artikulo para sa paksang ito.",
        "enter_keyword_first": "Maglagay muna ng keyword o paksa.",
        "url": "Suriin ang artikulo mula sa URL",
        "load_url": "🌐 I-load ang artikulo mula sa URL",
        "article_loaded_from_url": "Na-load na ang artikulo mula sa URL.",
        "unable_to_retrieve_text": "Hindi makuha ang teksto mula sa URL na ito.",
        "paste_url_first": "I-paste muna ang URL.",
        "paste": "I-paste ang artikulo o teksto rito",
        "analyze": "🔍 Suriin ang artikulo",
        "manual_paste": "mano-manong paste",
        "loaded_url_source": "artikulong na-load mula sa URL",
        "text_source": "Pinagmulan ng teksto",
        "paste_text_or_load_url": "I-paste ang teksto o i-load ang URL, pagkatapos ay i-click ang “🔍 Suriin ang artikulo”.",
        "classic_score": "Classic Score",
        "improved_score": "Improved Score",
        "hard_fact_score": "Hard Fact Score",
        "help_classic_score": "M = (G + N) − D",
        "help_improved_score": "Pagdaragdag ng V at parusa sa R",
        "help_hard_fact_score": "Mas mahigpit na kontrol sa mga claim at source",
        "credibility_gauge": "Credibility Gauge",
        "fragile": "Marupok",
        "fragile_message": "Ipinapakita ng teksto ang malalaking kahinaang istruktural o paktwal.",
        "doubtful": "Kahina-hinala",
        "doubtful_message": "May ilang kapani-paniwalang elemento ang teksto, ngunit nananatiling lubhang hindi tiyak.",
        "plausible": "Kapani-paniwala",
        "plausible_message": "Mukhang kapani-paniwala ang teksto sa kabuuan, ngunit kailangan pa ring tiyakin.",
        "robust": "Matibay",
        "robust_message": "May medyo matibay na batayang istruktural at paktwal ang teksto.",
        "score": "Iskor",
        "verdict": "Hatol",
        "summary": "Buod ng pagsusuri",
        "strengths_detected": "Mga natukoy na lakas",
        "few_strong_signals": "Kaunti ang malalakas na signal na natukoy.",
        "weaknesses_detected": "Mga natukoy na kahinaan",
        "no_major_weakness": "Walang malaking kahinaang natukoy ng heuristic.",
        "presence_of_source_markers": "May mga marker ng source o datos",
        "verifiability_clues": "May mga palatandaan ng verifiability: link, numero, petsa o porsiyento",
        "text_contains_nuances": "May mga nuance, limitasyon o kontra-punto ang teksto",
        "text_evokes_robust_sources": "Tumutukoy ang teksto sa mga source na maaaring matibay o institusyonal",
        "some_claims_verifiable": "May ilang claim na sapat ang pagkakaangkla upang ma-verify nang maayos",
        "overly_assertive_language": "Masyadong tiyak o absolutistang wika",
        "notable_emotional_sensational_charge": "May kapansin-pansing emosyonal o sensasyonal na bigat",
        "almost_total_absence_of_verifiable_elements": "Halos walang elementong maaaring ma-verify",
        "text_too_short": "Masyadong maikli ang teksto para seryosong suportahan ang malakas na claim",
        "multiple_claims_very_fragile": "Maraming sentral na claim ang napakarupok batay sa mga nakikitang palatandaan",
        "hard_fact_checking_by_claim": "Hard fact-checking ayon sa claim",
        "claim": "Claim",
        "status": "Katayuan",
        "verifiability": "Verifiability",
        "risk": "Panganib",
        "number": "Numero",
        "date": "Petsa",
        "named_entity": "Named Entity",
        "attributed_source": "Attributed Source",
        "yes": "Oo",
        "no": "Hindi",
        "to_verify": "Susuriin",
        "rather_verifiable": "Medyo mabe-verify",
        "very_fragile": "Napakarupok",
        "low_credibility": "Mababang kredibilidad",
        "prudent_credibility": "Maingat na kredibilidad",
        "rather_credible": "Medyo kapani-paniwala",
        "strong_credibility": "Malakas na kredibilidad",
        "paste_longer_text": "Mag-paste ng mas mahabang teksto para sa mas pinong mapa ng mga claim.",
        "llm_analysis": "Mecroyance Analysis para sa LLM",
        "llm_intro": "Inilalapat ng seksyong ito ang mga modelong hango sa treatise upang suriin ang cognitive posture ng isang sistema (AI o tao).",
        "overconfidence": "Overconfidence (Asymmetry)",
        "calibration": "Relative Calibration (Ratio)",
        "revisability": "Revisability (R)",
        "cognitive_closure": "Cognitive Closure",
        "interpretation": "Interpretasyon",
        "llm_metrics": "Mga Metriko ng LLM",
        "zone_closure": "Zone ng cognitive closure: ang katiyakan ay lumalampas sa cognitive anchoring.",
        "zone_stability": "Zone ng revisable stability: ang mecroyance ay sumasama nang hindi nangingibabaw.",
        "zone_lucidity": "Zone ng tumataas na lucidity: ang duda ang naghuhubog sa cognition.",
        "zone_rare": "Rare zone: mataas na integrated at reflexive na cognition.",
        "zone_pansapience": "Hypothetical pan-sapience: limit horizon ng halos ganap na revisable na cognition.",
        "zone_asymptote": "Ideal asymptote: kabuuan ng kaalaman at integrasyon nang walang rigidification.",
        "out_of_spectrum": "Halaga sa labas ng theoretical spectrum.",
        "external_corroboration_module": "🔎 Panlabas na corroboration module",
        "external_corroboration_caption": "Naghahanap ang module na ito ng mga panlabas na source na maaaring magkumpirma, magbigay-linaw, o sumalungat sa mga sentral na claim ng pasted text.",
        "corroboration_in_progress": "Naghahanap ng corroborations...",
        "generated_query": "Nabuo na query",
        "no_strong_sources_found": "Walang sapat na matibay na source para sa claim na ito.",
        "no_corroboration_found": "Walang mapapakinabangang corroboration na nahanap.",
        "corroborated": "Nakoroborahan",
        "mixed": "May halong pagtutugma",
        "not_corroborated": "Hindi nakoroborahan",
        "insufficiently_documented": "Hindi sapat ang dokumentasyon",
        "corroboration_verdict": "Hatol ng corroboration",
        "match_score": "Match score",
        "contradiction_signal": "Signal ng contradiction",
        "detected": "Nakita",
        "not_detected": "Hindi nakita",
        "ai_module": "AI module",
        "ai_module_caption": "Muling binabasa ng AI ang heuristic analysis at gumagawa ng mas buod na kritikal na pagbasa.",
        "generate_ai_analysis": "✨ Gumawa ng AI analysis",
        "ai_unavailable": "Hindi available ang AI module: walang OpenAI key o hindi naka-install ang library.",
        "ai_analysis_result": "AI analysis",
        "ai_claim_explanations": "AI paliwanag ng mga claim",
        "ai_explain_claim": "Ipaliwanag ang claim na ito",
        "ai_explanation": "Paliwanag",
        "method": "Pamamaraan",
        "original_formula": "Orihinal na Formula",
        "articulated_knowledge_density": "G: articulated knowledge density — mga source, numero, pangalan, sanggunian, at mga bakas na mabe-verify.",
        "integration": "N: integration — konteksto, mga nuance, reserbasyon, at argumentative coherence.",
        "assertive_rigidity": "D: assertive rigidity — mga katiyakang walang sapat na batayan, retorikal na paglobo.",
        "disclaimer": "Hindi kapalit ng mamamahayag, mananaliksik, o tagapag-ingat ng realidad ang app na ito. Ngunit nakakatanggal na ito ng ilang maskara sa tekstong nagmamartsa.",
    },
}


# -----------------------------
# Language
# -----------------------------
lang = st.selectbox(translations["Français"]["language"], list(translations.keys()))
T = translations[lang]


# -----------------------------
# Header
# -----------------------------
st.title(T["title"])
st.info(T["intro"])
st.caption(T["intro_2"])


# -----------------------------
# Cognition model
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
# Example data
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

    # -----------------------------
    # 1) NewsAPI d'abord
    # -----------------------------
    api_key = st.secrets.get("NEWS_API_KEY")

    if api_key:
        url = "https://newsapi.org/v2/everything"

        params = {
            "q": keyword,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": max_results * 2,
            "apiKey": api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                for art in data.get("articles", []):
                    article_url = art.get("url")
                    title = art.get("title")
                    source = art.get("source", {}).get("name", "Unknown")

                    if not article_url or article_url in seen_urls:
                        continue

                    seen_urls.add(article_url)

                    articles.append(
                        {
                            "title": title,
                            "url": article_url,
                            "source": source,
                        }
                    )

                    if len(articles) >= max_results:
                        return articles

            else:
                st.warning(f"NewsAPI HTTP error {response.status_code}")

        except Exception as e:
            st.warning(f"NewsAPI error: {e}")

    # -----------------------------
    # 2) Fallback DuckDuckGo
    # -----------------------------
    trusted_domains = [
        "lemonde.fr", "lefigaro.fr", "liberation.fr", "francetvinfo.fr",
        "lexpress.fr", "lepoint.fr", "nouvelobs.com", "la-croix.com",
        "lesechos.fr", "latribune.fr", "mediapart.fr", "arte.tv",
        "bbc.com", "reuters.com", "apnews.com", "nytimes.com",
        "theguardian.com", "bloomberg.com", "dw.com", "aljazeera.com",
        "nature.com", "science.org", "who.int", "un.org", "worldbank.org",
        "elpais.com", "elmundo.es", "corriere.it", "spiegel.de", "zeit.de",
        "france24.com", "20minutes.fr", "ouest-france.fr", "tf1info.fr",
        "cnbc.com", "npr.org", "abcnews.go.com", "cbsnews.com",
    ]

    try:
        with DDGS() as ddgs:
            query = f"{keyword} news article analysis report"
            ddg_results = list(ddgs.text(query, max_results=max_results * 8))

            for r in ddg_results:
                article_url = r.get("href", "")
                title = r.get("title", "Untitled")

                if not article_url or article_url in seen_urls:
                    continue

                if not any(domain in article_url for domain in trusted_domains):
                    continue

                seen_urls.add(article_url)

                articles.append(
                    {
                        "title": title,
                        "url": article_url,
                        "source": article_url.split("/")[2] if "/" in article_url else article_url,
                    }
                )

                if len(articles) >= max_results:
                    break

    except Exception as e:
        st.warning(f"DuckDuckGo fallback error: {e}")

    return articles[:max_results]


    trusted_domains = [
        "lemonde.fr", "lefigaro.fr", "liberation.fr", "francetvinfo.fr",
        "lexpress.fr", "lepoint.fr", "nouvelobs.com", "la-croix.com",
        "lesechos.fr", "latribune.fr", "mediapart.fr", "arte.tv",
        "bbc.com", "reuters.com", "apnews.com", "nytimes.com",
        "theguardian.com", "bloomberg.com", "dw.com", "aljazeera.com",
        "nature.com", "science.org", "who.int", "un.org", "worldbank.org",
        "elpais.com", "elmundo.es", "corriere.it", "spiegel.de", "zeit.de",
    ]
    results: List[Dict] = []
    try:
        with DDGS() as ddgs:
            query = f"{keyword} news article analysis study report"
            ddg_results = list(ddgs.text(query, max_results=max_results * 5))
            for r in ddg_results:
                url = r.get("href", "")
                if any(domain in url for domain in trusted_domains):
                    results.append(
                        {
                            "title": r.get("title", "Untitled"),
                            "url": url,
                            "source": url.split("/")[2] if "/" in url else url,
                        }
                    )
                    if len(results) >= max_results:
                        break
    except Exception as e:
        st.warning(f"Search error: {e}")
    return results


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
    "selon", "affirme", "déclare", "rapport", "étude", "expert", "source", "dit", "écrit", "publié",
    "according to", "claims", "states", "report", "study", "expert", "source", "says", "writes", "published",
    "según", "informe", "estudio", "experto", "fuente", "publicado",
]
ABSOLUTIST_WORDS = [
    "toujours", "jamais", "absolument", "certain", "prouvé", "incontestable", "tous", "aucun",
    "always", "never", "absolutely", "certain", "proven", "unquestionable", "all", "none",
    "siempre", "nunca", "absolutamente", "cierto", "probado", "incuestionable", "todos", "ninguno",
]
EMOTIONAL_WORDS = [
    "choc", "incroyable", "terrible", "peur", "menace", "scandale", "révolution", "urgent",
    "shock", "incredible", "terrible", "fear", "threat", "scandal", "revolution", "urgent",
    "choque", "increíble", "miedo", "amenaza", "escándalo", "revolución", "urgente",
]
NUANCE_MARKERS = [
    "cependant", "pourtant", "néanmoins", "toutefois", "mais", "nuancer", "prudence", "possible", "peut-être",
    "however", "yet", "nevertheless", "nonetheless", "but", "nuance", "caution", "possible", "maybe",
    "sin embargo", "no obstante", "pero", "matizar", "prudencia", "posible", "quizá",
]


def analyze_claim(sentence: str) -> Claim:
    has_number = bool(re.search(r"\d+", sentence))
    has_date = bool(
        re.search(
            r"\d{4}|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre",
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

    certainty = len(re.findall(r"certain|absolument|prouvé|évident|incontestable|certainly|absolutely|proven|obvious|unquestionable|cierto|absolutamente|probado", text.lower()))
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

    M = (G + N) - D
    ME = (2 * D) - (G + N)

    return {
        "words": len(words),
        "sentences": len(sentences),
        "G": G,
        "N": N,
        "D": D,
        "M": M,
        "ME": ME,
        "V": V,
        "R": R,
        "improved": improved,
        "source_quality": source_quality,
        "avg_claim_risk": avg_claim_risk,
        "avg_claim_verifiability": avg_claim_verifiability,
        "hard_fact_score": hard_fact_score,
        "verdict": verdict,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "claims": claims,
        "red_flags": red_flags,
    }


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
                        "Title": art["title"],
                        "Classic Score": analysis["M"],
                        "Hard Fact Score": analysis["hard_fact_score"],
                        "Verdict": analysis["verdict"],
                        "URL": art["url"],
                    }
                )
        except Exception:
            continue
    return results


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
        if any(word in s.lower() for word in ["selon", "affirme", "déclare", "rapport", "étude", "expert", "source", "publié", "annonce", "confirme", "révèle", "according to", "report", "study", "expert"]):
            score += 1
        if any(word in s.lower() for word in ["absolument", "certain", "jamais", "toujours", "incontestable", "choc", "scandale", "révolution", "urgent", "absolutely", "certain", "never", "always", "urgent"]):
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
        "les", "des", "une", "dans", "avec", "pour", "that", "this", "from", "have",
        "will", "être", "sont", "mais", "plus", "comme", "nous", "vous", "they",
        "their", "about", "into", "sur", "par", "est", "ont", "aux", "the", "and",
        "du", "de", "la", "le", "un", "et", "ou", "en", "à", "au", "ce",
        "ces", "ses", "son", "sa", "qui", "que", "quoi", "dont", "ainsi", "alors",
        "los", "las", "del", "para", "con", "como", "pero", "sobre", "este", "esta",
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
        "false", "faux", "misleading", "trompeur", "incorrect", "inexact",
        "debunked", "démenti", "refuted", "réfuté", "no evidence", "aucune preuve",
        "falso", "engañoso", "desmentido", "refutado", "sin pruebas",
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
        st.warning(f"Corroboration error: {e}")

    return corroboration_results


# -----------------------------
# AI helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def generate_ai_summary(lang: str, article_text: str, result: Dict, max_chars: int = 7000) -> str:
    if client is None:
        return ""

    short_text = article_text[:max_chars]
    claims_preview = []
    for c in result.get("claims", [])[:8]:
        claims_preview.append(
            {
                "claim": c.text,
                "status": c.status,
                "verifiability": c.verifiability,
                "risk": c.risk,
                "has_number": c.has_number,
                "has_date": c.has_date,
                "has_named_entity": c.has_named_entity,
                "has_source_cue": c.has_source_cue,
            }
        )

    prompt = f"""
You are a rigorous critical-reading assistant.
Write in the selected language: {lang}

Your task:
1. Summarize the overall credibility profile of the text.
2. Explain the difference between structural plausibility and factual robustness.
3. Point out the 3 main strengths.
4. Point out the 3 main weaknesses.
5. End with a prudent verdict.

Constraints:
- Be clear, concise, and concrete.
- Do not invent facts.
- Do not say the text is true or false with certainty unless the evidence clearly justifies it.
- Base yourself on the heuristic metrics below.

Heuristic analysis:
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

Text to analyze:
{short_text}
"""

    try:
        response = client.responses.create(model="gpt-4o", input=prompt)
        return response.output_text.strip()
    except Exception as e:
        return f"AI error: {e}"


@st.cache_data(show_spinner=False)
def explain_claim_with_ai(lang: str, claim_text: str, claim_data: Dict) -> str:
    if client is None:
        return ""

    prompt = f"""
You are a critical fact-checking assistant.
Write in the selected language: {lang}

Explain why this sentence received its score.
Be concrete and structured in 4 short parts:
1. What makes it verifiable
2. What makes it fragile
3. What would be needed to verify it properly
4. Final caution level

Sentence:
{claim_text}

Claim data:
{json.dumps(claim_data, ensure_ascii=False, indent=2)}
"""

    try:
        response = client.responses.create(model="gpt-4o-mini", input=prompt)
        return response.output_text.strip()
    except Exception as e:
        return f"AI error: {e}"


# -----------------------------
# Settings panel
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
if use_sample:
    st.session_state.article = SAMPLE_ARTICLE
    st.session_state.article_source = "paste"


# -----------------------------
# Multi-article section
# -----------------------------
st.subheader(T["topic_section"])
keyword = st.text_input(T["topic"], placeholder=T["topic_placeholder"])

if st.button(T["analyze_topic"], key="analyze_topic"):
    if keyword.strip():
        st.info(T["searching"])
        multiple_results = analyze_multiple_articles(keyword.strip(), max_results=10)
        if multiple_results:
            df_multi = pd.DataFrame(multiple_results).sort_values("Hard Fact Score", ascending=False)
            st.success(f"{len(df_multi)} {T['articles_analyzed']}")

            c1, c2 = st.columns(2)
            c1.metric(T["analyzed_articles"], len(df_multi))
            c2.metric(T["avg_hard_fact"], round(df_multi["Hard Fact Score"].mean(), 1))
            st.metric(T["avg_classic_score"], round(df_multi["Classic Score"].mean(), 1))

            ecart_type_hf = df_multi["Hard Fact Score"].std()
            indice_doxa = "high" if ecart_type_hf < 1.5 else ("medium" if ecart_type_hf < 3 else "low")
            st.metric(T["topic_doxa_index"], T[indice_doxa])

            st.subheader(T["credibility_score_dispersion"])
            df_plot = df_multi.copy()
            df_plot["Article"] = [f"{T['article_label']} {i+1}" for i in range(len(df_plot))]
            st.bar_chart(df_plot.set_index("Article")["Hard Fact Score"])
            st.dataframe(df_multi, use_container_width=True, hide_index=True)
        else:
            st.warning(T["no_exploitable_articles_found"])
    else:
        st.warning(T["enter_keyword_first"])


# -----------------------------
# URL form
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
            st.success(T["article_loaded_from_url"])
        else:
            st.error(T["unable_to_retrieve_text"])
    else:
        st.warning(T["paste_url_first"])


# -----------------------------
# Main article form
# -----------------------------
# -----------------------------
# Zone de saisie + micro visuellement collé au texte
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
            value=st.session_state.article,
            height=220,
            label_visibility="collapsed",
            placeholder=T["paste"]
        )
        analyze_submitted = st.form_submit_button(
            T["analyze"],
            use_container_width=True
        )

if article.strip() != previous_article.strip():
    st.session_state.article_source = "paste"

st.session_state.article = article

st.caption(
    f"{T['text_source']} : "
    f"{T['manual_paste'] if st.session_state.get('article_source') == 'paste' else T['loaded_url_source']}"
)

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_article" not in st.session_state:
    st.session_state.last_article = ""

if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = ""

# -----------------------------
# Main analysis
# -----------------------------
# -----------------------------
# Main analysis
# -----------------------------
if analyze_submitted:
    st.session_state.last_result = analyze_article(article)
    st.session_state.last_article = article
    st.session_state.ai_summary = ""

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

    fig_triangle = plot_cognitive_triangle_3d(
        result["G"],
        result["N"],
        result["D"]
    )
    st.pyplot(fig_triangle, use_container_width=True)

    st.subheader("Cognitive Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Mécroyance Index (M)", round(result["M"], 2))

    with col2:
        st.metric("Mendacity Index (ME)", round(result["ME"], 2))
    delta_mm = round(result["M"] - result["ME"], 2)
    st.caption(f"Cognitive gap (M − ME) : {delta_mm}")
    if result["M"] > result["ME"] + 1:
        dominant_pattern = "Dominant pattern: mécroyance"
    elif result["ME"] > result["M"] + 1:
        dominant_pattern = "Dominant pattern: strategic lying"
    else:
        dominant_pattern = "Dominant pattern: mixed or ambiguous"

    st.subheader("Dominant cognitive pattern")
    st.write(dominant_pattern)

    if result["ME"] > result["M"] and result["ME"] > 0:
        cognitive_type = "Possible strategic lying"
    elif result["M"] < 0:
        cognitive_type = "High mécroyance / cognitive closure"
    else:
        cognitive_type = "Likely sincere but misaligned cognition"

    st.subheader("Cognitive Interpretation")
    st.write(cognitive_type)

    if result["M"] - result["ME"] > 3:
        diagnosis = "Strong mécroyance structure"
    elif result["M"] > result["ME"]:
        diagnosis = "Moderate mécroyance structure"
    elif abs(result["M"] - result["ME"]) <= 1:
        diagnosis = "Ambiguous cognitive structure"
    else:
        diagnosis = "Possible strategic deception"

    st.subheader("Cognitive diagnosis")
    st.write(diagnosis)
    conflict = abs(result["M"] - result["ME"])
    conflict_bar = min(conflict / 10, 1)

    st.write("Cognitive tension (mécroyance vs mendacity)")
    st.progress(conflict_bar)

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
    st.subheader(T["llm_analysis"])
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
            with st.spinner("AI is analyzing..."):
                ai_summary = generate_ai_summary(lang, article, result)
            st.subheader(T["ai_analysis_result"])
            st.markdown(ai_summary)
    if st.session_state.get("article_source") == "paste":
        st.divider()
        st.subheader(T["external_corroboration_module"])
        st.caption(T["external_corroboration_caption"])
        with st.spinner(T["corroboration_in_progress"]):
            corroboration = corroborate_claims(article, max_claims=5, max_results_per_claim=3)
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
# Method section
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
# Laboratoire interactif de la mécroyance
# -----------------------------
st.divider()
st.subheader("Laboratoire interactif de la mécroyance")
st.caption(
    "Expérimentez la formule cognitive : M = (G + N) − D. "
    "Modifiez les paramètres pour observer l’évolution des stades cognitifs."
)

# Curseurs
g_game = st.slider("G — gnōsis (savoir articulé)", 0.0, 10.0, 5.0, 0.5)
n_game = st.slider("N — nous (intégration vécue)", 0.0, 10.0, 5.0, 0.5)
d_game = st.slider("D — doxa (certitude / saturation)", 0.0, 10.0, 5.0, 0.5)

# Calcul
m_game = round((g_game + n_game) - d_game, 1)

# Affichage formule
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

# Détermination du stade
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

# Affichage stable du stade
st.markdown(f"**Stade actuel : {stage}**")
st.progress(percent / 100)
st.caption(f"M = {m_game} — {explanation}")

# Frise cognitive
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

st.caption(
    "Lorsque G et N augmentent sans inflation de D, la cognition gagne en revisabilité."
)
