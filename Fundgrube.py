import streamlit as st
import tensorflow as tf
import numpy as np

from pathlib import Path
from PIL import Image
import json
import base64
import io
from datetime import datetime


# ============================================================
# FUN DGRUBE – Katharineum zu Lübeck
# ============================================================

st.set_page_config(
    page_title="Fundgrube – Katharineum zu Lübeck",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# DATEIPFADE
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"

ASSETS_DIR = BASE_DIR / "assets"

SILHOUETTE1_PATH = ASSETS_DIR / "Silouette1.jpg"
SILHOUETTE2_PATH = ASSETS_DIR / "Silouette2.jpg"

DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "items.json"

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)


# ============================================================
# SEITEN
# ============================================================

PAGE_SEARCH = "search"
PAGE_UPLOAD = "upload"
PAGE_OLDEST = "oldest"


if "page" not in st.session_state:
    st.session_state.page = PAGE_SEARCH

if "menu_open" not in st.session_state:
    st.session_state.menu_open = False


# ============================================================
# FARBEN
# ============================================================

PINK = "#F1848B"
BLUE = "#A8DFF7"
BLACK = "#000000"
WHITE = "#FFFFFF"


# ============================================================
# HILFSFUNKTION:
# SILHOUETTE AUS JPG FREISTELLEN
# ============================================================

def silhouette_to_data_uri(image_path):
    """
    Lädt die JPG-Silhouette und macht den weißen Hintergrund
    transparent.

    Dadurch wird NICHT das komplette JPG als Hintergrund
    verwendet. Es bleibt nur die dunkle Silhouette übrig.
    """

    if not image_path.exists():
        return None

    try:
        image = Image.open(image_path).convert("RGBA")

        pixels = image.load()

        for y in range(image.height):
            for x in range(image.width):
                r, g, b, a = pixels[x, y]

                # Helligkeit des Pixels
                brightness = (r + g + b) / 3

                # Weiß wird transparent.
                # Graue Übergänge bleiben leicht sichtbar.
                new_alpha = int(max(0, min(255, 255 - brightness)))

                pixels[x, y] = (
                    0,
                    0,
                    0,
                    new_alpha
                )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return f"data:image/png;base64,{encoded}"

    except Exception:
        return None


# ============================================================
# SILHOUETTEN LADEN
# ============================================================

silhouette1 = silhouette_to_data_uri(SILHOUETTE1_PATH)
silhouette2 = silhouette_to_data_uri(SILHOUETTE2_PATH)


# ============================================================
# MODEL LADEN
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        return model
    except Exception:
        return None


model = load_model()


# ============================================================
# LABELS LADEN
# ============================================================

def load_labels():

    if not LABELS_PATH.exists():
        return [
            "Hose",
            "Sporthose",
            "Hoodies",
            "Jacke"
        ]

    labels = []

    try:
        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:
                line = line.strip()

                if not line:
                    continue

                # Teachable Machine labels.txt kann z.B.
                # "0 Hose" oder "Hose" enthalten.
                parts = line.split(" ", 1)

                if len(parts) == 2 and parts[0].isdigit():
                    labels.append(parts[1].strip())
                else:
                    labels.append(line)

    except Exception:
        labels = []

    return labels


LABELS = load_labels()


# ============================================================
# METADATEN
# ============================================================

def load_items():

    if not METADATA_PATH.exists():
        return []

    try:
        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):
                return data

            return []

    except Exception:
        return []


def save_items(items):

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


# ============================================================
# BILD KLASSIFIZIEREN
# ============================================================

def classify_image(image):

    if model is None:
        return "Unbekannt", 0.0

    try:

        image = image.convert("RGB")

        image = image.resize(
            (224, 224)
        )

        image_array = np.asarray(
            image
        ).astype(np.float32)

        # Teachable Machine erwartet normalerweise
        # Werte zwischen -1 und 1.
        image_array = (
            image_array / 127.5
        ) - 1

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        prediction = model.predict(
            image_array,
            verbose=0
        )

        probabilities = prediction[0]

        index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[index]
        )

        if index < len(LABELS):
            label = LABELS[index]
        else:
            label = f"Klasse {index + 1}"

        return label, confidence

    except Exception:
        return "Unbekannt", 0.0


# ============================================================
# CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* --------------------------------------------------------
       STREAMLIT GRUNDLAYOUT
       -------------------------------------------------------- */

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    header {{
        visibility: hidden;
    }}

    .block-container {{
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }}

    .stApp {{
        background: white;
    }}


    /* --------------------------------------------------------
       ALLGEMEINE SCHRIFT
       -------------------------------------------------------- */

    html, body, [class*="css"] {{
        font-family: Arial, Helvetica, sans-serif;
    }}


    /* --------------------------------------------------------
       TITEL
       -------------------------------------------------------- */

    .fundgrube-title {{
        color: black;
        font-size: clamp(45px, 6vw, 88px);
        font-weight: 800;
        text-align: center;
        line-height: 1;
        margin-top: 20px;
        margin-bottom: 15px;
    }}


    /* --------------------------------------------------------
       HAMBURGER
       -------------------------------------------------------- */

    .hamburger-line {{
        width: 58px;
        height: 5px;
        background: black;
        margin: 10px 0;
        border-radius: 4px;
    }}


    /* --------------------------------------------------------
       SILHOUETTE
       -------------------------------------------------------- */

    .silhouette-container {{
        width: 100%;
        height: 260px;
        display: flex;
        align-items: flex-end;
        justify-content: center;
        overflow: hidden;
        margin-top: -5px;
        margin-bottom: 20px;
    }}

    .silhouette-container img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
        object-position: center bottom;
    }}


    /* --------------------------------------------------------
       SUCHSEITE
       -------------------------------------------------------- */

    .search-area {{
        text-align: center;
        margin-top: 20px;
    }}

    .search-icon {{
        font-size: 125px;
        line-height: 1;
        color: black;
        margin-top: 5px;
        margin-bottom: 15px;
    }}

    .search-description {{
        font-size: clamp(22px, 3vw, 40px);
        color: black;
        border: 3px solid black;
        border-radius: 45px;
        padding: 8px 25px;
        display: inline-block;
        margin-bottom: 20px;
    }}


    /* --------------------------------------------------------
       UPLOAD-SEITE
       -------------------------------------------------------- */

    .upload-title {{
        color: black;
        font-size: clamp(40px, 5vw, 75px);
        font-weight: 800;
        text-align: center;
        margin-top: 20px;
    }}

    .upload-box {{
        border: 4px dashed black;
        border-radius: 30px;
        padding: 35px;
        margin: 30px auto;
        max-width: 850px;
        text-align: center;
        background: rgba(255,255,255,0.75);
    }}

    .upload-text {{
        font-size: 27px;
        color: black;
        margin-bottom: 20px;
    }}


    /* --------------------------------------------------------
       ÄLTESTE FUNDSTÜCKE
       -------------------------------------------------------- */

    .oldest-title {{
        color: black;
        font-size: clamp(40px, 5vw, 75px);
        font-weight: 800;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 45px;
    }}

    .item-card {{
        border: 3px solid black;
        border-radius: 18px;
        padding: 10px;
        background: white;
        margin-bottom: 20px;
    }}

    .item-label {{
        color: black;
        font-size: 20px;
        font-weight: 600;
        text-align: center;
        margin-top: 8px;
    }}


    /* --------------------------------------------------------
       MENÜ
       -------------------------------------------------------- */

    .menu-title {{
        color: black;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 20px;
    }}


    /* --------------------------------------------------------
       BUTTONS
       -------------------------------------------------------- */

    div.stButton > button {{
        border: 2px solid black;
        border-radius: 30px;
        background: white;
        color: black;
        font-size: 18px;
        font-weight: 500;
        min-height: 48px;
    }}

    div.stButton > button:hover {{
        border-color: black;
        background: #eeeeee;
        color: black;
    }}


    /* --------------------------------------------------------
       INPUT
       -------------------------------------------------------- */

    div[data-baseweb="input"] {{
        border: 3px solid black !important;
        border-radius: 35px !important;
        background: white !important;
    }}

    input {{
        color: black !important;
        font-size: 20px !important;
    }}


    /* --------------------------------------------------------
       MOBILE
       -------------------------------------------------------- */

    @media (max-width: 700px) {{

        .silhouette-container {{
            height: 160px;
        }}

        .search-icon {{
            font-size: 85px;
        }}

        .upload-box {{
            padding: 20px;
        }}

    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HAMBURGER-MENÜ
# ============================================================

top_left, top_middle, top_right = st.columns(
    [1, 8, 1]
)

with top_left:

    if st.button(
        "☰",
        key="hamburger",
        help="Menü öffnen"
    ):
        st.session_state.menu_open = not st.session_state.menu_open
        st.rerun()


# ============================================================
# MENÜ ANZEIGEN
# ============================================================

if st.session_state.menu_open:

    st.markdown(
        '<div class="menu-title">Fundgrube</div>',
        unsafe_allow_html=True
    )

    menu1, menu2, menu3 = st.columns(3)

    with menu1:

        if st.button(
            "Suche",
            key="menu_search",
            use_container_width=True
        ):
            st.session_state.page = PAGE_SEARCH
            st.session_state.menu_open = False
            st.rerun()

    with menu2:

        if st.button(
            "Fundstück einstellen",
            key="menu_upload",
            use_container_width=True
        ):
            st.session_state.page = PAGE_UPLOAD
            st.session_state.menu_open = False
            st.rerun()

    with menu3:

        if st.button(
            "Älteste Fundstücke",
            key="menu_oldest",
            use_container_width=True
        ):
            st.session_state.page = PAGE_OLDEST
            st.session_state.menu_open = False
            st.rerun()


st.divider()


# ============================================================
# SEITE 1 – SUCHE
# ============================================================

if st.session_state.page == PAGE_SEARCH:

    # Pinker Bereich
    st.markdown(
        f"""
        <div style="
            background:{PINK};
            padding:20px 20px 0 20px;
            border-radius:0;
        ">

            <div class="fundgrube-title">
                Fundgrube
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Silhouette 1
    if silhouette1:

        st.markdown(
            f"""
            <div class="silhouette-container"
                 style="background:{PINK};">

                <img src="{silhouette1}">

            </div>
            """,
            unsafe_allow_html=True
        )


    # Lupe
    st.markdown(
        """
        <div class="search-area">

            <div class="search-icon">
                🔍
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Suchfeld
    st.markdown(
        """
        <div style="text-align:center;">
        """,
        unsafe_allow_html=True
    )

    search_text = st.text_input(
        " ",
        placeholder="Beschreibe dein verlorenes Kleidungsstück",
        label_visibility="collapsed",
        key="search_input"
    )

    st.markdown("</div>", unsafe_allow_html=True)


    # Suche ausführen
    if search_text.strip():

        items = load_items()

        search_lower = search_text.lower().strip()

        matching_items = []

        for item in items:

            label = str(
                item.get("label", "")
            ).lower()

            filename = str(
                item.get("filename", "")
            ).lower()

            description = str(
                item.get("description", "")
            ).lower()

            if (
                search_lower in label
                or search_lower in filename
                or search_lower in description
            ):
                matching_items.append(item)


        st.markdown(
            "### Gefundene Fundstücke"
        )

        if not matching_items:

            st.info(
                "Leider wurde noch kein passendes Fundstück gefunden."
            )

        else:

            columns = st.columns(3)

            for index, item in enumerate(
                matching_items
            ):

                image_path = IMAGE_DIR / item.get(
                    "filename",
                    ""
                )

                if image_path.exists():

                    with columns[index % 3]:

                        st.image(
                            str(image_path),
                            use_container_width=True
                        )

                        st.caption(
                            item.get(
                                "label",
                                "Fundstück"
                            )
                        )


# ============================================================
# SEITE 2 – FUNDSTÜCK EINSTELLEN
# ============================================================

elif st.session_state.page == PAGE_UPLOAD:

    # Blauer Kopfbereich
    st.markdown(
        f"""
        <div style="
            background:{BLUE};
            padding:20px;
        ">

            <div class="upload-title">
                Lade ein Bild hoch
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Silhouette 2
    if silhouette2:

        st.markdown(
            f"""
            <div class="silhouette-container"
                 style="background:{BLUE};">

                <img src="{silhouette2}">

            </div>
            """,
            unsafe_allow_html=True
        )


    # Upload-Bereich
    st.markdown(
        """
        <div class="upload-box">

            <div class="upload-text">
                Lade ein Foto aus deiner Mediathek hoch
            </div>

        </div>
        """,
        unsafe_allow_html=True
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


    if uploaded_file is not None:

        try:

            uploaded_image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                uploaded_image,
                caption="Ausgewähltes Fundstück",
                use_container_width=True
            )


            if st.button(
                "Fundstück einstellen",
                use_container_width=True
            ):

                with st.spinner(
                    "Fundstück wird analysiert..."
                ):

                    label, confidence = classify_image(
                        uploaded_image
                    )


                # eindeutiger Dateiname
                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                extension = (
                    Path(uploaded_file.name)
                    .suffix
                    .lower()
                )

                if extension not in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]:
                    extension = ".jpg"


                filename = (
                    f"fundstueck_"
                    f"{timestamp}"
                    f"{extension}"
                )


                save_path = (
                    IMAGE_DIR / filename
                )


                uploaded_image.save(
                    save_path,
                    quality=95
                )


                # Metadaten speichern
                items = load_items()

                new_item = {
                    "filename": filename,
                    "label": label,
                    "confidence": confidence,
                    "description": "",
                    "created_at": datetime.now().isoformat()
                }

                items.append(
                    new_item
                )

                save_items(
                    items
                )


                st.success(
                    f"Fundstück gespeichert. "
                    f"Erkannte Kategorie: {label}"
                )

                st.info(
                    f"KI-Sicherheit: "
                    f"{confidence * 100:.1f}%"
                )


        except Exception as error:

            st.error(
                "Das Bild konnte nicht verarbeitet werden."
            )

            st.caption(
                str(error)
            )


# ============================================================
# SEITE 3 – ÄLTESTE FUNDSTÜCKE
# ============================================================

elif st.session_state.page == PAGE_OLDEST:

    # KEINE SILHOUETTE AUF DIESER SEITE

    st.markdown(
        """
        <div class="oldest-title">
            Bald nicht mehr verfügbar
        </div>
        """,
        unsafe_allow_html=True
    )


    items = load_items()


    # Nach Erstellungsdatum sortieren:
    # älteste zuerst
    items_sorted = sorted(
        items,
        key=lambda item: item.get(
            "created_at",
            ""
        )
    )


    # Nur die 9 ältesten
    oldest_items = items_sorted[:9]


    if not oldest_items:

        st.info(
            "Es wurden noch keine Fundstücke eingestellt."
        )

    else:

        # Drei Spalten
        columns = st.columns(3)


        for index, item in enumerate(
            oldest_items
        ):

            image_path = (
                IMAGE_DIR /
                item.get(
                    "filename",
                    ""
                )
            )


            if image_path.exists():

                with columns[index % 3]:

                    st.markdown(
                        '<div class="item-card">',
                        unsafe_allow_html=True
                    )

                    st.image(
                        str(image_path),
                        use_container_width=True
                    )

                    label = item.get(
                        "label",
                        "Fundstück"
                    )

                    st.markdown(
                        f"""
                        <div class="item-label">
                            {label}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )
