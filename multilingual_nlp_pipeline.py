"""
multilingual_nlp_pipeline.py
==============================
Multilingual NLP pipeline for Ooredoo complaint analysis.
Supports Arabic, French, and English without translation.

Each language has its own lexicon — no translation needed.
Language detection is automatic (rule-based, no external API).

Usage:
    from src.nlp.multilingual_nlp_pipeline import MultilingualNLPPipeline
    pipe   = MultilingualNLPPipeline()
    result = pipe.analyze("شبكتي مقطوعة في تونس منذ 3 أيام")
    result = pipe.analyze("Mon réseau coupe à Sfax depuis 3 jours")
    result = pipe.analyze("My 4G network keeps dropping in Tunis")
"""

from __future__ import annotations
import re
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

ARABIC_RANGE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
FRENCH_WORDS = {
    "mon","ma","le","la","les","de","du","je","ne","pas","un","une",
    "réseau","internet","appel","connexion","coupure","lent","problème",
    "depuis","jours","impossible","mauvais","bonjour","merci",
    "débit","facture","client","service","panne","signal","appels",
}
ENGLISH_WORDS = {
    "my","the","is","not","network","internet","call","connection",
    "slow","problem","since","days","impossible","bad","hello","thanks",
    "signal","drop","weak","no","service","issue","complaint","data",
    "billing","customer","support","coverage","speed","working",
}

def detect_language(text: str) -> str:
    """Detect language: 'ar', 'fr', or 'en'."""
    if not text or not isinstance(text, str):
        return "fr"
    
    text_lower = text.lower()
    arabic_chars = len(ARABIC_RANGE.findall(text))
    
    # Arabic detection (strong signal)
    if arabic_chars > len(text) * 0.2:  # At least 20% Arabic characters
        return "ar"
    
    words = set(re.findall(r'\b\w+\b', text_lower))
    fr_score = len(words & FRENCH_WORDS)
    en_score = len(words & ENGLISH_WORDS)
    
    # If no matches, default to French
    if fr_score == 0 and en_score == 0:
        return "fr"
    
    # Return the language with higher score
    return "fr" if fr_score >= en_score else "en"


# ══════════════════════════════════════════════════════════════════════════════
# LEXICONS — one per language
# ══════════════════════════════════════════════════════════════════════════════

# ── FRENCH ────────────────────────────────────────────────────────────────────
FR_CATEGORIES = {
    "Réseau / Couverture": [
        "pas de réseau","aucun signal","pas de couverture","zone blanche",
        "hors réseau","réseau indisponible","signal faible","coupure réseau",
        "réseau mobile","antenne","4g","5g","3g","lte","perte réseau",
        "pas de 4g","réseau coupe","pas de signal","sans réseau",
    ],
    "Débit / Internet": [
        "débit faible","internet lent","connexion lente","lenteur","slow",
        "téléchargement lent","mbps","bande passante","accès internet",
        "pas d'internet","internet ne marche pas","coupure internet",
        "déconnexion","connexion instable","page ne charge pas",
        "vitesse","lags","débit","lent",
    ],
    "Appels / Voix": [
        "appel coupé","coupure appel","appel impossible","voix haché","echo",
        "bruit","qualité voix","appel échoué","ne peut pas appeler",
        "appel ne passe pas","tonalité","volte","appels coupent",
        "ligne coupe","communication coupée","grésillement",
    ],
    "SMS": [
        "sms non reçu","sms non envoyé","message non délivré","texto",
        "sms bloqué","message échoué","ne reçois pas sms",
    ],
    "Facturation": [
        "facture","surfacturation","débit abusif","recharge","solde incorrect",
        "forfait","crédit","tarification","erreur facturation","prélèvement",
        "souscription","option","facturation","trop payé",
    ],
    "Support Client": [
        "service client","attente","conseiller","réclamation non traitée",
        "aucune réponse","problème non résolu","rappel","hotline",
        "pas de retour","support","assistance","réclamation",
    ],
}

FR_SENTIMENT = {
    "critique": [
        "inacceptable","inadmissible","scandaleux","honteux","catastrophique",
        "nul","terrible","arnaque","résiliation","porter plainte","aberrant",
        "dégoutant","intolérable","exaspérant","insupportable",
    ],
    "négatif": [
        "problème","panne","coupure","mauvais","lent","difficile","impossible",
        "ne marche pas","ne fonctionne pas","déçu","insatisfait","gêné",
        "énervé","fatigant","perturbation","dysfonctionnement",
    ],
    "positif": [
        "merci","bien","bon","excellent","parfait","satisfait","rapide","bravo",
        "génial","top","super","agréable","content",
    ],
}

FR_URGENCY = [
    "urgence","immédiatement","tout de suite","bloqué","impossible",
    "hôpital","danger","critique","sos","grave","priorité",
    "urgent","vite","maintenant",
]

FR_STOPS = {
    "le","la","les","un","une","des","de","du","en","et","est","au","aux",
    "ce","se","je","tu","il","nous","vous","ils","mon","ma","mes","ton",
    "que","qui","pour","par","avec","sur","dans","ne","pas","plus","très",
    "bien","tout","même","si","mais","depuis","lors","jai","cest","na",
    "faire","être","avoir","aller","venir","sans","comme",
}

# ── ARABIC ────────────────────────────────────────────────────────────────────
AR_CATEGORIES = {
    "الشبكة / التغطية": [
        "لا شبكة","انقطاع الشبكة","ضعف التغطية","لا إشارة","شبكة مقطوعة",
        "لا تغطية","إشارة ضعيفة","الشبكة واقعة","لا 4g","لا 3g",
        "تغطية سيئة","الشبكة لا تعمل","انقطاع متكرر","فقدان الشبكة",
        "منقطع","مشكلة شبكة","التغطية","الأنتنة",
    ],
    "الصبيب / الأنترنت": [
        "أنترنت بطيء","صبيب ضعيف","اتصال بطيء","تحميل بطيء",
        "أنترنت منقطع","لا أنترنت","اتصال غير مستقر","سرعة ضعيفة",
        "أنترنت لا يعمل","انقطاع الأنترنت","ميغابت","بطء",
        "النت","الأنترنت","السرعة","تقطيع",
    ],
    "المكالمات / الصوت": [
        "المكالمة منقطعة","انقطاع المكالمات","لا يمكن الاتصال",
        "جودة صوت سيئة","صدى","ضجيج","المكالمة لا تمر","فشل المكالمة",
        "مكالمة مقطوعة","لا أسمع","اتصال صوتي","صعوبة في المكالمة",
        "مشكلة صوت","مكالمة","اتصال",
    ],
    "الرسائل القصيرة": [
        "sms","رسالة لم تصل","رسالة لم ترسل","لم استلم الرسالة",
        "الرسالة فشلت","مشكلة رسائل","واتساب","ماسنجر",
    ],
    "الفاتورة": [
        "فاتورة","خصم مجحف","رصيد","شحن","خطأ في الفاتورة",
        "تسعير","اشتراك","خصم","رصيد غير صحيح","سحب فلوس",
        "فلوس","حساب","فوترة","تخفيض",
    ],
    "دعم العملاء": [
        "خدمة العملاء","انتظار","لا رد","شكوى غير محلولة",
        "لا استجابة","خط ساخن","موظف","شكوى","تذمر",
        "المشكلة","مساعدة","دعم","مركز الاتصال",
    ],
}

AR_SENTIMENT = {
    "حرج": [
        "لا يقبل","فضيحة","كارثي","مرفوض","سيء جدا","مشكلة كبيرة",
        "غير مقبول","رديء","فاضح","مروع","فظيع","مقزز",
    ],
    "سلبي": [
        "مشكلة","عطل","انقطاع","بطيء","صعب","مستحيل","لا يعمل",
        "تعطل","غير راضي","متضايق","غاضب","زعلان","متعب",
        "خلل","عيوب","تعب","ملل",
    ],
    "إيجابي": [
        "شكرا","ممتاز","جيد","رائع","راضي","سريع","أحسنتم",
        "ممتازة","جميل","حسن","سرور","استحسان",
    ],
}

AR_URGENCY = [
    "عاجل","فورا","الآن","خطر","حرج","طوارئ","ضروري","مستعجل",
    "مهم جدا","خطير","حياة","مستشفى","سريع","حالا",
]

AR_CITIES = {
    "تونس","صفاقس","سوسة","القيروان","بنزرت","قابس","أريانة",
    "قفصة","المنستير","بن عروس","نابل","القصرين","سيدي بوزيد",
    "المهدية","منوبة","جندوبة","سليانة","زغوان","باجة","الكاف",
    "قبلي","توزر","تطاوين","مدنين","الحمامات","جرجيس","جربة",
    "المرسى","الكرم","قرطاج",
}

AR_STOPS = {
    "في","من","إلى","على","عن","مع","هذا","هذه","التي","الذي",
    "و","أو","لكن","لأن","كان","كانت","هو","هي","نحن","أنا",
    "أن","إن","قد","لقد","لا","ما","كل","بعض","ذلك","تلك",
}

# ── ENGLISH ───────────────────────────────────────────────────────────────────
EN_CATEGORIES = {
    "Network / Coverage": [
        "no network","no signal","no coverage","dead zone","network down",
        "weak signal","network drop","no 4g","no 3g","network unavailable",
        "signal lost","poor coverage","network cut","out of service",
        "no service","can't connect","signal issue","coverage issue",
    ],
    "Data / Internet": [
        "slow internet","slow connection","low speed","buffering","lag",
        "no internet","internet down","connection drop","disconnecting",
        "unstable connection","download slow","mbps","bandwidth",
        "speed issue","data not working","wifi problem","can't browse",
    ],
    "Calls / Voice": [
        "call drops","call failed","cannot call","voice quality","echo",
        "noise","call cut","call disconnected","no dial tone",
        "can't make calls","volte","call breaking up","can't hear",
        "call quality","dropped call","failing calls",
    ],
    "SMS": [
        "sms not received","sms not delivered","message failed",
        "text not sent","sms blocked","can't send sms","no sms",
    ],
    "Billing": [
        "invoice","overcharged","billing error","recharge","balance wrong",
        "plan","credit","wrong charge","deducted","overcharge",
        "bill","payment","subscription","charging",
    ],
    "Customer Support": [
        "customer service","waiting","no response","complaint not resolved",
        "hotline","callback","agent","support","no help","bad service",
        "unresolved issue","no solution","poor support",
    ],
}

EN_SENTIMENT = {
    "critical": [
        "unacceptable","terrible","awful","outrageous","disgusting",
        "worst","horrible","cancel","lawsuit","scam","fraud",
        "intolerable","unbearable","insufferable","atrocious",
    ],
    "negative": [
        "problem","issue","broken","slow","bad","impossible","doesn't work",
        "not working","disappointed","unhappy","frustrated","annoyed",
        "upset","angry","poor","mediocre","useless",
    ],
    "positive": [
        "thank","good","great","excellent","perfect","satisfied","fast",
        "well done","awesome","amazing","wonderful","fantastic",
        "happy","pleased","impressed",
    ],
}

EN_URGENCY = [
    "urgent","immediately","right now","blocked","critical","emergency",
    "asap","danger","sos","severe","priority","quick","fast",
    "hours","minutes","now",
]

EN_STOPS = {
    "the","a","an","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","might","i",
    "my","your","his","her","our","their","it","this","that","these",
    "those","and","or","but","for","so","yet","nor","in","on","at",
    "to","of","with","by","from","up","about","into","through","during",
    "without","between","after","before","above","below","between",
}

TN_CITIES_EN = {
    "tunis","sfax","sousse","kairouan","bizerte","gabes","ariana",
    "gafsa","monastir","ben arous","nabeul","kasserine","sidi bouzid",
    "mahdia","manouba","jendouba","siliana","zaghouan","beja","kef",
    "kebili","tozeur","tataouine","medenine","hammamet","zarzis","djerba",
    "marsa","kram","carthage","sakiet","sidi","raoued",
}

TN_CITIES_FR = {
    "tunis","sfax","sousse","kairouan","bizerte","gabès","gabes","ariana",
    "gafsa","monastir","ben arous","nabeul","kasserine","sidi bouzid",
    "mahdia","manouba","jendouba","siliana","zaghouan","béja","beja",
    "kef","kebili","tozeur","tataouine","medenine","mednine","hammamet",
    "zarzis","djerba","la marsa","el kram","carthage","sakiet","raoued",
}

NETWORK_TYPES_ALL = {"4g","5g","3g","2g","volte","lte","wifi","fibre",
                     "4G","5G","3G","2G","LTE","VoLTE","4g+","5g+"}


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class MultilingualNLPPipeline:
    """
    Analyze complaints in Arabic, French, or English.
    No translation — each language uses its own lexicon.
    
    Features:
        - Automatic language detection
        - Category classification
        - Sentiment analysis
        - Urgency scoring
        - Entity extraction (city, network type)
        - Keyword extraction
        - Batch processing for DataFrames
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the pipeline.
        
        Args:
            verbose: If True, print progress information during batch processing
        """
        self.verbose = verbose
        self.stats = {
            "total_processed": 0,
            "languages": {"ar": 0, "fr": 0, "en": 0},
            "categories": {},
        }

    def analyze(self, text: str) -> dict:
        """
        Analyze a single complaint text.
        
        Args:
            text: Complaint text in Arabic, French, or English
            
        Returns:
            Dictionary with analysis results
        """
        if not text or not isinstance(text, str) or len(text.strip()) < 3:
            return self._empty(text)

        lang = detect_language(text)
        text_clean = self._preprocess(text, lang)

        category = self._category(text_clean, lang)
        sentiment = self._sentiment(text_clean, lang)
        urgency = self._urgency(text_clean, lang, sentiment)
        entities = self._entities(text_clean, lang)
        keywords = self._keywords(text_clean, lang)

        # Map sentiment labels to unified French for storage
        sentiment_fr = self._map_sentiment_to_french(sentiment, lang)

        # Update stats
        self.stats["total_processed"] += 1
        self.stats["languages"][lang] = self.stats["languages"].get(lang, 0) + 1
        self.stats["categories"][category] = self.stats["categories"].get(category, 0) + 1

        return {
            "text": text,
            "language": lang,
            "category": category,
            "sentiment": sentiment_fr,
            "sentiment_raw": sentiment,  # Original language sentiment
            "urgency_score": urgency["score"],
            "urgency_level": urgency["level"],
            "city": entities.get("city"),
            "network_type": entities.get("network_type"),
            "keywords": keywords,
            "processed_at": datetime.now().isoformat(),
        }
    
    def _map_sentiment_to_french(self, sentiment: str, lang: str) -> str:
        """Map sentiment to unified French labels."""
        if lang == "ar":
            mapping = {"حرج": "critique", "سلبي": "négatif", "إيجابي": "positif", "محايد": "neutre"}
            return mapping.get(sentiment, "neutre")
        elif lang == "en":
            mapping = {"critical": "critique", "negative": "négatif", 
                      "positive": "positif", "neutral": "neutre"}
            return mapping.get(sentiment, "neutre")
        else:
            # French
            mapping = {"critique": "critique", "négatif": "négatif", 
                      "positif": "positif", "neutre": "neutre"}
            return mapping.get(sentiment, "neutre")
    
    def analyze_batch(self, texts: List[str], show_progress: bool = True) -> List[dict]:
        """
        Analyze multiple complaint texts.
        
        Args:
            texts: List of complaint texts
            show_progress: If True, display progress bar
            
        Returns:
            List of analysis results
        """
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress and self.verbose and i % 100 == 0:
                print(f"Processing {i+1}/{total}...")
            results.append(self.analyze(text))
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = "complaint_text",
                         add_columns: bool = True) -> Union[pd.DataFrame, List[dict]]:
        """
        Analyze complaints from a DataFrame.
        
        Args:
            df: DataFrame containing complaint texts
            text_column: Name of column containing text
            add_columns: If True, add analysis columns to DataFrame
            
        Returns:
            If add_columns=True: DataFrame with added columns
            If add_columns=False: List of analysis results
        """
        texts = df[text_column].fillna("").astype(str).tolist()
        results = self.analyze_batch(texts, show_progress=True)
        
        if add_columns:
            result_df = df.copy()
            result_df["nlp_language"] = [r["language"] for r in results]
            result_df["nlp_category"] = [r["category"] for r in results]
            result_df["nlp_sentiment"] = [r["sentiment"] for r in results]
            result_df["nlp_urgency_score"] = [r["urgency_score"] for r in results]
            result_df["nlp_urgency_level"] = [r["urgency_level"] for r in results]
            result_df["nlp_city"] = [r["city"] for r in results]
            result_df["nlp_network_type"] = [r["network_type"] for r in results]
            result_df["nlp_keywords"] = [", ".join(r["keywords"]) for r in results]
            return result_df
        
        return results
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self.stats = {
            "total_processed": 0,
            "languages": {"ar": 0, "fr": 0, "en": 0},
            "categories": {},
        }

    # ── Preprocessing ──────────────────────────────────────────────────────
    def _preprocess(self, text: str, lang: str) -> str:
        """Clean and normalize text based on language."""
        if lang == "ar":
            # Normalize Arabic
            text = re.sub(r'[إأآا]', 'ا', text)
            text = re.sub(r'ى', 'ي', text)
            text = re.sub(r'ة', 'ه', text)
            text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        else:
            text = text.lower()
            text = re.sub(r'[^\w\s\-\']', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    # ── Category ───────────────────────────────────────────────────────────
    def _category(self, text: str, lang: str) -> str:
        """Classify complaint category."""
        lexicon = (AR_CATEGORIES if lang == "ar"
                   else EN_CATEGORIES if lang == "en"
                   else FR_CATEGORIES)
        
        scores = {}
        for cat, keywords in lexicon.items():
            score = 0
            for kw in keywords:
                if kw in text:
                    score += 1
                    # Give more weight to exact matches for short keywords like "4g"
                    if len(kw) <= 3 and re.search(rf'\b{re.escape(kw)}\b', text):
                        score += 0.5
            scores[cat] = score
        
        best = max(scores, key=scores.get)
        
        if scores[best] == 0:
            # Default "Other" category
            defaults = {"ar": "أخرى", "en": "Other", "fr": "Autre"}
            return defaults[lang]
        
        return best

    # ── Sentiment ──────────────────────────────────────────────────────────
    def _sentiment(self, text: str, lang: str) -> str:
        """Analyze sentiment."""
        lexicon = (AR_SENTIMENT if lang == "ar"
                   else EN_SENTIMENT if lang == "en"
                   else FR_SENTIMENT)
        
        scores = {}
        for sent, keywords in lexicon.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[sent] = score
        
        # Get keys in order of severity (first is most severe)
        keys = list(lexicon.keys())
        
        # Priority: return the most severe sentiment with positive score
        for k in keys:
            if scores.get(k, 0) > 0:
                return k
        
        # Default neutral
        defaults = {"ar": "محايد", "en": "neutral", "fr": "neutre"}
        return defaults[lang]

    # ── Urgency ────────────────────────────────────────────────────────────
    def _urgency(self, text: str, lang: str, sentiment: str) -> dict:
        """Calculate urgency score and level."""
        # Base scores by sentiment
        base_scores = {
            # French
            "critique": 0.7, "négatif": 0.4, "neutre": 0.2, "positif": 0.1,
            # Arabic
            "حرج": 0.7, "سلبي": 0.4, "محايد": 0.2, "إيجابي": 0.1,
            # English
            "critical": 0.7, "negative": 0.4, "neutral": 0.2, "positive": 0.1,
        }
        score = base_scores.get(sentiment, 0.2)

        # Urgency keywords
        urgency_kws = (AR_URGENCY if lang == "ar"
                       else EN_URGENCY if lang == "en"
                       else FR_URGENCY)
        
        for kw in urgency_kws:
            if kw in text:
                score = min(score + 0.25, 1.0)
                break  # One match is enough for boost

        # Duration boost (problem duration in days)
        if lang == "ar":
            days = re.findall(r'(\d+)\s*(?:أيام|يوم|ايام)', text)
        elif lang == "en":
            days = re.findall(r'(\d+)\s*(?:days?|day)', text)
        else:
            days = re.findall(r'(\d+)\s*(?:jours?|jour)', text)
        
        if days:
            days_count = int(days[0])
            score = min(score + (days_count * 0.05), 1.0)

        # Capitalization boost (for French/English)
        if lang != "ar":
            uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            if uppercase_ratio > 0.3:  # Lots of uppercase
                score = min(score + 0.15, 1.0)

        # Exclamation boost
        if "!" in text:
            score = min(score + 0.1, 1.0)

        score = round(score, 2)
        
        # Determine urgency level
        if score >= 0.8:
            level = "très urgent"
        elif score >= 0.5:
            level = "urgent"
        else:
            level = "normal"
        
        return {"score": score, "level": level}

    # ── Entity extraction ──────────────────────────────────────────────────
    def _entities(self, text: str, lang: str) -> dict:
        """Extract entities: city, network type."""
        entities = {"city": None, "network_type": None}

        # Network type (universal)
        text_lower = text.lower()
        for net_type in ["5g", "4g", "3g", "2g", "volte", "lte", "wifi", "fibre"]:
            if net_type in text_lower:
                entities["network_type"] = net_type.upper()
                break

        # City extraction
        if lang == "ar":
            for city in AR_CITIES:
                if city in text:
                    entities["city"] = city
                    break
        elif lang == "en":
            for city in TN_CITIES_EN:
                if city in text_lower:
                    # Capitalize properly
                    entities["city"] = city.title()
                    break
        else:  # French
            for city in TN_CITIES_FR:
                if city in text_lower:
                    entities["city"] = city.title()
                    break

        return entities

    # ── Keywords ───────────────────────────────────────────────────────────
    def _keywords(self, text: str, lang: str, top_n: int = 5) -> List[str]:
        """Extract top keywords."""
        stops = (AR_STOPS if lang == "ar"
                 else EN_STOPS if lang == "en"
                 else FR_STOPS)
        
        # For Arabic, keep words with length >= 3
        # For others, keep words with length >= 4
        min_len = 3 if lang == "ar" else 4
        
        words = re.findall(r'\b\w+\b', text)
        words = [w for w in words if len(w) >= min_len and w not in stops]
        
        # Count frequencies
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        
        # Return top N
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_n]]

    def _empty(self, text) -> dict:
        """Return empty/default analysis for invalid text."""
        return {
            "text": text,
            "language": "fr",
            "category": "Autre",
            "sentiment": "neutre",
            "sentiment_raw": "neutral",
            "urgency_score": 0.0,
            "urgency_level": "normal",
            "city": None,
            "network_type": None,
            "keywords": [],
            "processed_at": datetime.now().isoformat(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def quick_analyze(text: str) -> dict:
    """
    Quick analysis function for single texts.
    
    Args:
        text: Complaint text
        
    Returns:
        Analysis dictionary
    """
    pipeline = MultilingualNLPPipeline()
    return pipeline.analyze(text)


def batch_analyze(texts: List[str], show_progress: bool = True) -> List[dict]:
    """
    Batch analysis for multiple texts.
    
    Args:
        texts: List of complaint texts
        show_progress: Show progress information
        
    Returns:
        List of analysis dictionaries
    """
    pipeline = MultilingualNLPPipeline(verbose=show_progress)
    return pipeline.analyze_batch(texts, show_progress=show_progress)


def enrich_dataframe(df: pd.DataFrame, text_column: str = "complaint_text") -> pd.DataFrame:
    """
    Enrich a DataFrame with NLP analysis columns.
    
    Args:
        df: DataFrame with complaints
        text_column: Name of the column containing text
        
    Returns:
        DataFrame with added NLP columns
    """
    pipeline = MultilingualNLPPipeline()
    return pipeline.analyze_dataframe(df, text_column=text_column, add_columns=True)


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test examples
    pipe = MultilingualNLPPipeline(verbose=True)
    
    test_texts = [
        "شبكتي مقطوعة في تونس منذ 3 أيام، لا أستطيع الاتصال، هذا غير مقبول!",
        "Mon réseau 4G coupe à Sfax depuis 5 jours, c'est catastrophique!",
        "My network keeps dropping in Tunis for 2 days, this is unacceptable!",
        "Merci pour votre service, tout fonctionne bien.",
        "Test",
        "",
    ]
    
    print("=" * 60)
    print("Multilingual NLP Pipeline Test")
    print("=" * 60)
    
    for text in test_texts:
        result = pipe.analyze(text)
        print(f"\n📝 Text: {text[:50]}...")
        print(f"   Language: {result['language']}")
        print(f"   Category: {result['category']}")
        print(f"   Sentiment: {result['sentiment']}")
        print(f"   Urgency: {result['urgency_level']} ({result['urgency_score']})")
        print(f"   City: {result['city']}")
        print(f"   Network: {result['network_type']}")
        print(f"   Keywords: {', '.join(result['keywords'])}")
    
    print("\n" + "=" * 60)
    print(f"Pipeline Statistics: {pipe.get_stats()}")
    print("=" * 60)
