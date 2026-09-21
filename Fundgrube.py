# ============================================================
# FUNDGRUBE – Katharineum zu Lübeck
# Interaktive Streamlit-App
# ============================================================

import base64
import io
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


# ============================================================
# 1. GRUNDEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundgrube – Katharineum zu Lübeck",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


APP_TITLE = "Fundgrube"

MODEL_PATH = Path("keras_model.h5")
LABELS_PATH = Path("labels.txt")

ASSETS_DIR = Path("assets")
SILOUETTE1_PATH = ASSETS_DIR / "Silouette1.jpg"
SILOUETTE2_PATH = ASSETS_DIR / "Silouette2.jpg"

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "items.json"

IMAGE_SIZE = (224, 224)


# ============================================================
# 2. ORDNER ANLEGEN
# ============================================================

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "search"

if "menu_open" not in st.session_state:
    st.session_state.menu_open = False

if "search_text" not in st.session_state:
    st.session_state.search_text = ""

if "search_performed" not in st.session_state:
    st.session_state.search_performed = False


# ============================================================
# 4. HILFSFUNKTIONEN
# ============================================================

def load_labels():
    """
    Liest die Klassen aus labels.txt.
    Funktioniert sowohl mit:
        0 Hose
        1 Sporthose
    als auch mit:
        Hose
        Sporthose
    """

    if not LABELS_PATH.exists():
        return []

    labels = []

    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                # Format: "0 Hose"
                match = re.match(r"^\s*\d+\s+(.+)$", line)

                if match:
                    label = match.group(1).strip()
                else:
                    label = line.strip()

                labels.append(label)

    except Exception:
        return []

    return labels


LABELS = load_labels()


# ============================================================
# 5. MODELL LADEN
# ============================================================

@st.cache_resource
def load_model():
    """
    Lädt das Teachable-Machine-Keras-Modell.
    """

    if not MODEL_PATH.exists():
        return None

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model

    except Exception:
        return None


MODEL = load_model()


# ============================================================
# 6. BILDERKLASSIFIZIERUNG
# ============================================================

def predict_image(image):
    """
    Klassifiziert ein Bild mit dem vorhandenen
    Teachable-Machine-Modell.
    """

    if MODEL is None:
        return "Unbekannt", 0.0

    try:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

        array = np.asarray(image).astype(np.float32)

        # Teachable Machine verwendet normalerweise
        # eine Skalierung auf -1 bis +1.
        array = (array / 127.5) - 1.0

        array = np.expand_dims(array, axis=0)

        prediction = MODEL.predict(array, verbose=0)

        prediction = np.asarray(prediction).flatten()

        if len(prediction) == 0:
            return "Unbekannt", 0.0

        index = int(np.argmax(prediction))
        confidence = float(prediction[index])

        if index < len(LABELS):
            label = LABELS[index]
        else:
            label = f"Klasse {index + 1}"

        return label, confidence

    except Exception:
        return "Unbekannt", 0.0


# ============================================================
# 7. METADATEN LADEN / SPEICHERN
# ============================================================

def load_items():
    if not METADATA_PATH.exists():
        return []

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_items(items):
    try:
        with open(
            METADATA_PATH,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                items,
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception:
        pass


# ============================================================
# 8. SILHOUETTEN
# ============================================================

@st.cache_data
def create_silhouette_png(path_string):
    """
    Nimmt Silouette1.jpg bzw. Silouette2.jpg und entfernt
    technisch den hellen/weißen Hintergrund.

    Dadurch wird NICHT das komplette JPG als Hintergrund
    angezeigt.

    Nur die dunkle Schulsilhouette bleibt sichtbar.
    """

    path = Path(path_string)

    if not path.exists():
        return None

    try:
        image = Image.open(path).convert("RGBA")

        # Auf eine brauchbare Größe bringen
        max_width = 1800

        if image.width > max_width:
            ratio = max_width / image.width

            image = image.resize(
                (
                    int(image.width * ratio),
                    int(image.height * ratio)
                ),
                Image.Resampling.LANCZOS
            )

        pixels = image.load()

        for y in range(image.height):
            for x in range(image.width):

                r, g, b, a = pixels[x, y]

                # Helle Pixel werden transparent.
                brightness = (r + g + b) / 3

                if brightness > 220:
                    pixels[x, y] = (255, 255, 255, 0)

                elif brightness > 170:
                    alpha = int(
                        255 *
                        (220 - brightness) /
                        50
                    )

                    pixels[x, y] = (
                        0,
                        0,
                        0,
                        max(0, min(255, alpha))
                    )

                else:
                    pixels[x, y] = (
                        0,
                        0,
                        0,
                        255
                    )

        output = io.BytesIO()
        image.save(output, format="PNG")

        return output.getvalue()

    except Exception:
        return None


def image_to_base64(image_bytes):
    if not image_bytes:
        return ""

    return base64.b64encode(image_bytes).decode("utf-8")


SILOUETTE1_BYTES = create_silhouette_png(
    str(SILOUETTE1_PATH)
)

SILOUETTE2_BYTES = create_silhouette_png(
    str(SILOUETTE2_PATH)
)

SILOUETTE1_BASE64 = image_to_base64(
    SILOUETTE1_BYTES
)

SILOUETTE2_BASE64 = image_to_base64(
    SILOUETTE2_BYTES
)


# ============================================================
# 9. CSS
# ============================================================

st.markdown(
    """
<style>

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

[data-testid="stAppViewContainer"] {
    background: #ffffff;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 0rem !important;
    padding-bottom: 2rem !important;
    max-width: 1500px;
}


/* ----------------------------------------------------------
   ALLGEMEINE BUTTONS
---------------------------------------------------------- */

.stButton > button {
    border: 3px solid #111111 !important;
    border-radius: 30px !important;
    background: white !important;
    color: #111111 !important;
    font-size: 18px !important;
    font-weight: 500 !important;
    min-height: 48px !important;
}

.stButton > button:hover {
    background: #f0f0f0 !important;
    border-color: #111111 !important;
}


/* ----------------------------------------------------------
   HAMBURGER
---------------------------------------------------------- */

.hamburger-button {
    position: absolute;
    top: 20px;
    left: 20px;
    z-index: 999;
}


/* ----------------------------------------------------------
   MENÜ
---------------------------------------------------------- */

.menu-box {
    position: fixed;
    top: 80px;
    left: 25px;
    width: 280px;
    padding: 25px;
    background: rgba(255,255,255,0.97);
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.20);
    z-index: 998;
}


/* ----------------------------------------------------------
   TITEL
---------------------------------------------------------- */

.main-title {
    text-align: center;
    font-size: clamp(50px, 7vw, 100px);
    font-weight: 800;
    color: #000000;
    margin-top: 15px;
    margin-bottom: 5px;
}


/* ----------------------------------------------------------
   SEITE 1
---------------------------------------------------------- */

.page-one {
    position: relative;
    min-height: 850px;
    overflow: hidden;
    background: linear-gradient(
        to bottom,
        #f28a91 0%,
        #f28a91 43%,
        #ffffff 43%,
        #ffffff 100%
    );
}


/* ----------------------------------------------------------
   SEITE 2
---------------------------------------------------------- */

.page-two {
    position: relative;
    min-height: 850px;
    overflow: hidden;
    background: linear-gradient(
        to bottom,
        #a9ddf4 0%,
        #a9ddf4 38%,
        #ffffff 38%,
        #ffffff 100%
    );
}


/* ----------------------------------------------------------
   SILHOUETTE
---------------------------------------------------------- */

.silhouette {
    position: absolute;
    left: 0;
    width: 100%;
    height: auto;
    z-index: 1;
    pointer-events: none;
}

.silhouette-one {
    top: 250px;
}

.silhouette-two {
    top: 210px;
}


/* ----------------------------------------------------------
   INHALT
---------------------------------------------------------- */

.content-layer {
    position: relative;
    z-index: 10;
}


/* ----------------------------------------------------------
   SUCHSYMBOL
---------------------------------------------------------- */

.search-icon {
    text-align: center;
    font-size: 130px;
    line-height: 1;
    margin-top: 50px;
}


/* ----------------------------------------------------------
   EINGABEFELD
---------------------------------------------------------- */

.search-container {
    max-width: 900px;
    margin: 25px auto;
}


/* ----------------------------------------------------------
   UPLOAD BOX
---------------------------------------------------------- */

.upload-box {
    max-width: 900px;
    margin: 80px auto 0 auto;
    padding: 60px 40px;
    border: 4px dashed #111111;
    border-radius: 30px;
    background: rgba(255,255,255,0.78);
    text-align: center;
}


/* ----------------------------------------------------------
   SEITE 3
---------------------------------------------------------- */

.oldest-page {
    min-height: 850px;
    background: white;
    position: relative;
}

.oldest-title {
    text-align: center;
    font-size: clamp(45px, 6vw, 80px);
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 50px;
}


/* ----------------------------------------------------------
   BILD-KARTEN
---------------------------------------------------------- */

.item-card {
    border: 3px solid #111111;
    border-radius: 18px;
    overflow: hidden;
    background: white;
    margin-bottom: 25px;
}

.item-card img {
    width: 100%;
    height: 220px;
    object-fit: cover;
}

.item-label {
    font-size: 18px;
    font-weight: 600;
    padding: 10px;
    text-align: center;
}


/* ----------------------------------------------------------
   LOGO / FUSS
---------------------------------------------------------- */

.school-footer {
    text-align: center;
    margin-top: 70px;
    font-size: 17px;
    font-weight: 600;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# 10. NAVIGATION
# ============================================================

def hamburger():
    """
    Öffnet bzw. schließt das Menü.
    """

    if st.button(
        "☰",
        key="hamburger",
        help="Menü öffnen",
        use_container_width=False
    ):
        st.session_state.menu_open = (
            not st.session_state.menu_open
        )


hamburger()


# ============================================================
# 11. MENÜ
# ============================================================

if st.session_state.menu_open:

    st.markdown(
        """
        <div class="menu-box">
            <h2>Fundgrube</h2>
            <p>Katharineum zu Lübeck</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "🔎 Suche",
            key="menu_search"
        ):
            st.session_state.page = "search"
            st.session_state.menu_open = False
            st.rerun()

    with col2:
        if st.button(
            "📷 Fundstück einstellen",
            key="menu_upload"
        ):
            st.session_state.page = "upload"
            st.session_state.menu_open = False
            st.rerun()

    with col3:
        if st.button(
            "🕘 Älteste Fundstücke",
            key="menu_oldest"
        ):
            st.session_state.page = "oldest"
            st.session_state.menu_open = False
            st.rerun()


# ============================================================
# 12. SEITE 1 – SUCHE
# ============================================================

if st.session_state.page == "search":

    st.markdown(
        '<div class="page-one">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="content-layer">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-title">Fundgrube</div>',
        unsafe_allow_html=True
    )

    # Silhouette
    if SILOUETTE1_BASE64:

        st.markdown(
            f"""
            <img
                class="silhouette silhouette-one"
                src="data:image/png;base64,{SILOUETTE1_BASE64}"
            >
            """,
            unsafe_allow_html=True
        )

    # Lupe
    st.markdown(
        '<div class="search-icon">⌕</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="search-container">',
        unsafe_allow_html=True
    )

    search_text = st.text_input(
        "Beschreibe dein verlorenes Kleidungsstück",
        placeholder="z. B. schwarze Hose, Sporthose, Hoodie ...",
        key="search_input"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "Suchen",
        key="search_button"
    ):

        st.session_state.search_text = search_text
        st.session_state.search_performed = True

    st.markdown(
        '</div></div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # SUCHERGEBNISSE
    # --------------------------------------------------------

    if (
        st.session_state.search_performed
        and st.session_state.search_text.strip()
    ):

        items = load_items()

        query = st.session_state.search_text.lower().strip()

        results = []

        for item in items:

            category = str(
                item.get("category", "")
            ).lower()

            description = str(
                item.get("description", "")
            ).lower()

            filename = str(
                item.get("filename", "")
            ).lower()

            text = (
                category
                + " "
                + description
                + " "
                + filename
            )

            if (
                query in text
                or any(
                    word in text
                    for word in query.split()
                    if len(word) > 2
                )
            ):
                results.append(item)

        st.markdown(
            "### Gefundene Fundstücke"
        )

        if not results:

            st.info(
                "Leider wurde kein passendes Fundstück gefunden."
            )

        else:

            cols = st.columns(3)

            for index, item in enumerate(results):

                with cols[index % 3]:

                    image_path = Path(
                        item.get("path", "")
                    )

                    if image_path.exists():

                        st.image(
                            str(image_path),
                            use_container_width=True
                        )

                    st.write(
                        "**{}**".format(
                            item.get(
                                "category",
                                "Fundstück"
                            )
                        )
                    )


# ============================================================
# 13. SEITE 2 – FUNDSTÜCK HOCHLADEN
# ============================================================

elif st.session_state.page == "upload":

    st.markdown(
        '<div class="page-two">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="content-layer">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-title">Lade ein Bild hoch</div>',
        unsafe_allow_html=True
    )

    # Silhouette
    if SILOUETTE2_BASE64:

        st.markdown(
            f"""
            <img
                class="silhouette silhouette-two"
                src="data:image/png;base64,{SILOUETTE2_BASE64}"
            >
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="upload-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        "### Lade ein Foto aus deiner Mediathek hoch"
    )

    uploaded_file = st.file_uploader(
        "Foto auswählen",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.markdown(
                "### Vorschau"
            )

            st.image(
                image,
                width=400
            )

            if st.button(
                "Fundstück einstellen",
                key="submit_item"
            ):

                category, confidence = predict_image(
                    image
                )

                timestamp = datetime.now().isoformat(
                    timespec="seconds"
                )

                safe_name = re.sub(
                    r"[^a-zA-Z0-9._-]",
                    "_",
                    uploaded_file.name
                )

                filename = (
                    datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    + "_"
                    + safe_name
                )

                save_path = IMAGE_DIR / filename

                image.save(
                    save_path,
                    format="JPEG",
                    quality=92
                )

                items = load_items()

                new_item = {
                    "filename": filename,
                    "path": str(save_path),
                    "category": category,
                    "confidence": confidence,
                    "created_at": timestamp,
                    "description": ""
                }

                items.append(new_item)

                save_items(items)

                st.success(
                    "Das Fundstück wurde erfolgreich eingestellt."
                )

                if confidence > 0:

                    st.write(
                        f"Erkannte Kategorie: **{category}**"
                    )

                    st.write(
                        f"Erkennungswahrscheinlichkeit: "
                        f"**{confidence * 100:.1f} %**"
                    )

        except Exception as error:

            st.error(
                "Das Bild konnte nicht verarbeitet werden."
            )

    st.markdown(
        '</div></div>',
        unsafe_allow_html=True
    )


# ============================================================
# 14. SEITE 3 – ÄLTESTE FUNDSTÜCKE
# ============================================================

elif st.session_state.page == "oldest":

    st.markdown(
        '<div class="oldest-page">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="oldest-title">'
        'Bald nicht mehr verfügbar'
        '</div>',
        unsafe_allow_html=True
    )

    items = load_items()

    # --------------------------------------------------------
    # WICHTIG:
    # ÄLTESTE Fundstücke, nicht die neuesten.
    # --------------------------------------------------------

    def get_timestamp(item):

        value = item.get(
            "created_at",
            ""
        )

        try:
            return datetime.fromisoformat(
                value
            )
        except Exception:
            return datetime.max


    oldest_items = sorted(
        items,
        key=get_timestamp
    )[:9]


    if not oldest_items:

        st.info(
            "Es wurden noch keine Fundstücke eingestellt."
        )

    else:

        cols = st.columns(3)

        for index, item in enumerate(
            oldest_items
        ):

            with cols[index % 3]:

                st.markdown(
                    '<div class="item-card">',
                    unsafe_allow_html=True
                )

                image_path = Path(
                    item.get(
                        "path",
                        ""
                    )
                )

                if image_path.exists():

                    st.image(
                        str(image_path),
                        use_container_width=True
                    )

                category = item.get(
                    "category",
                    "Fundstück"
                )

                st.markdown(
                    f"""
                    <div class="item-label">
                        {category}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

    st.markdown(
        '<div class="school-footer">'
        'Katharineum zu Lübeck'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# 15. FALLBACK
# ============================================================

else:

    st.session_state.page = "search"
    st.rerun()
