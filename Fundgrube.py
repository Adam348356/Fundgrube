import base64
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image


# ============================================================
# FUNDGRUBE
# Katharineum zu Lübeck
# ============================================================

APP_TITLE = "Fundgrube"

MODEL_PATH = Path("keras_model.h5")
LABELS_PATH = Path("labels.txt")

METADATA_PATH = "data/items.json"
IMAGE_DIR = "data/images"

IMAGE_SIZE = (224, 224)


# ============================================================
# STREAMLIT EINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap'
    );

    :root {
        --pink: #f28a91;
        --blue: #a6def4;
        --black: #050505;
        --white: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: Inter, Arial, sans-serif;
    }

    .stApp {
        background: white;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* --------------------------------------------------------
       HEADER
       -------------------------------------------------------- */

    .topbar {
        position: relative;
        min-height: 145px;

        margin-top: -20px;
        margin-left: -5vw;
        margin-right: -5vw;

        padding: 25px 5vw 15px 5vw;

        display: flex;
        align-items: flex-start;
        justify-content: center;

        overflow: hidden;
    }

    .topbar.pink {
        background: var(--pink);
    }

    .topbar.blue {
        background: var(--blue);
    }

    .title {
        font-size: clamp(48px, 6vw, 92px);
        font-weight: 800;
        color: var(--black);

        line-height: 0.95;

        margin: 0;

        z-index: 3;

        letter-spacing: -4px;
    }

    /* --------------------------------------------------------
       HAMBURGER
       -------------------------------------------------------- */

    .hamburger {
        position: absolute;

        left: 25px;
        top: 20px;

        z-index: 10;

        width: 58px;
    }

    .hamburger span {
        display: block;

        height: 5px;

        margin-bottom: 18px;

        border-radius: 4px;

        background: black;
    }

    /* --------------------------------------------------------
       SILHOUETTE
       -------------------------------------------------------- */

    .silhouette {
        position: absolute;

        left: 0;
        right: 0;
        bottom: -5px;

        width: 100%;
        height: 115px;

        object-fit: cover;
        object-position: center;

        opacity: 0.98;

        z-index: 1;
    }

    /* --------------------------------------------------------
       SEARCH
       -------------------------------------------------------- */

    .search-area {
        max-width: 920px;

        margin: 50px auto 0 auto;

        text-align: center;
    }

    .hint {
        text-align: center;

        font-size: 17px;

        margin-top: 15px;
    }

    /* --------------------------------------------------------
       RESULTS
       -------------------------------------------------------- */

    .result-card {
        border: 2px solid black;

        border-radius: 20px;

        padding: 12px;

        background: white;

        margin-bottom: 18px;
    }

    .category {
        font-size: 20px;

        font-weight: 800;

        margin-top: 8px;
    }

    .small {
        color: #555;

        font-size: 13px;
    }

    /* --------------------------------------------------------
       UPLOAD
       -------------------------------------------------------- */

    .upload-box {
        max-width: 820px;

        min-height: 300px;

        margin: 40px auto;

        border: 3px dashed black;

        border-radius: 30px;

        display: flex;

        align-items: center;

        justify-content: center;

        text-align: center;

        padding: 35px;
    }

    .upload-title {
        font-size: 25px;

        font-weight: 600;
    }

    /* --------------------------------------------------------
       BUTTONS
       -------------------------------------------------------- */

    div.stButton > button {
        border: 2px solid black;

        border-radius: 30px;

        background: white;

        color: black;

        font-size: 18px;

        font-weight: 700;

        min-height: 48px;
    }

    div.stButton > button:hover {
        border-color: black;

        color: black;

        background: #f5f5f5;
    }

    /* --------------------------------------------------------
       SIDEBAR
       -------------------------------------------------------- */

    [data-testid="stSidebar"] {
        border-left: 2px solid black;

        background: white;
    }

    .menu-title {
        font-size: 28px;

        font-weight: 800;

        margin-bottom: 15px;
    }

    .side-note {
        border-top: 2px solid black;

        margin-top: 25px;

        padding-top: 15px;

        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GITHUB EINSTELLUNGEN
# ============================================================

def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


GITHUB_TOKEN = get_secret("GITHUB_TOKEN")
GITHUB_REPO = get_secret("GITHUB_REPO")
GITHUB_BRANCH = get_secret("GITHUB_BRANCH", "main")

GITHUB_API = "https://api.github.com"


def github_enabled():
    return bool(GITHUB_REPO)


def github_headers():
    headers = {
        "Accept": "application/vnd.github+json"
    }

    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    return headers


# ============================================================
# GITHUB DATEI LESEN
# ============================================================

def github_read_file(path):

    if not github_enabled():
        return None

    url = (
        f"{GITHUB_API}/repos/"
        f"{GITHUB_REPO}/contents/{path}"
    )

    response = requests.get(
        url,
        headers=github_headers(),
        params={
            "ref": GITHUB_BRANCH
        },
        timeout=20,
    )

    if response.status_code != 200:
        return None

    data = response.json()

    if data.get("encoding") == "base64":
        return base64.b64decode(
            data["content"]
        )

    return None


# ============================================================
# GITHUB DATEI SCHREIBEN
# ============================================================

def github_write_file(
    path,
    content_bytes,
    commit_message
):

    if not github_enabled():
        raise RuntimeError(
            "GITHUB_REPO wurde nicht eingerichtet."
        )

    if not GITHUB_TOKEN:
        raise RuntimeError(
            "GITHUB_TOKEN wurde nicht eingerichtet."
        )

    url = (
        f"{GITHUB_API}/repos/"
        f"{GITHUB_REPO}/contents/{path}"
    )

    existing = requests.get(
        url,
        headers=github_headers(),
        params={
            "ref": GITHUB_BRANCH
        },
        timeout=20,
    )

    payload = {
        "message": commit_message,

        "content": base64.b64encode(
            content_bytes
        ).decode("ascii"),

        "branch": GITHUB_BRANCH,
    }

    if existing.status_code == 200:

        payload["sha"] = (
            existing.json()["sha"]
        )

    response = requests.put(
        url,

        headers=github_headers(),

        json=payload,

        timeout=30,
    )

    if response.status_code not in (200, 201):

        raise RuntimeError(
            "GitHub-Fehler: "
            f"{response.status_code}\n"
            f"{response.text}"
        )

    return response.json()


# ============================================================
# KI-MODELL LADEN
# ============================================================

@st.cache_resource
def load_model():

    try:
        import tensorflow as tf

    except ImportError:

        raise RuntimeError(
            "TensorFlow ist nicht installiert."
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "keras_model.h5 wurde nicht gefunden."
        )

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    return model


# ============================================================
# LABELS LADEN
# ============================================================

@st.cache_data
def load_labels():

    if not LABELS_PATH.exists():

        raise FileNotFoundError(
            "labels.txt wurde nicht gefunden."
        )

    labels = []

    lines = LABELS_PATH.read_text(
        encoding="utf-8"
    ).splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Teachable Machine:
        # "0 Hosen"
        # "1 Sporthosen"
        # usw.

        match = re.match(
            r"^\s*\d+\s+(.+)$",
            line
        )

        if match:
            label = match.group(1).strip()
        else:
            label = line

        labels.append(label)

    return labels


# ============================================================
# FOTO KLASSIFIZIEREN
# ============================================================

def classify_image(image):

    model = load_model()

    labels = load_labels()

    # Bild in RGB umwandeln
    image = image.convert("RGB")

    # Auf die vom Modell erwartete Größe bringen
    image = image.resize(IMAGE_SIZE)

    # Numpy Array
    image_array = np.asarray(
        image
    ).astype(np.float32)

    # Teachable-Machine-Normalisierung
    image_array = (
        image_array / 127.0
    ) - 1.0

    # Batch-Dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Vorhersage
    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    # Höchste Wahrscheinlichkeit
    class_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[class_index]
    )

    if class_index < len(labels):

        category = labels[class_index]

    else:

        category = (
            f"Klasse {class_index}"
        )

    return category, confidence


# ============================================================
# DATEN LADEN
# ============================================================

def read_items():

    # Zuerst GitHub
    if github_enabled():

        raw = github_read_file(
            METADATA_PATH
        )

        if raw:

            try:

                return json.loads(
                    raw.decode("utf-8")
                )

            except Exception:

                return []

    # Lokaler Fallback
    local_file = Path(
        METADATA_PATH
    )

    if local_file.exists():

        try:

            return json.loads(
                local_file.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:

            return []

    return []


# ============================================================
# DATEN SPEICHERN
# ============================================================

def write_items(items):

    content = json.dumps(
        items,

        ensure_ascii=False,

        indent=2
    ).encode("utf-8")

    if github_enabled():

        github_write_file(
            METADATA_PATH,

            content,

            "Fundgrube: Metadaten aktualisieren"
        )

    else:

        Path(
            METADATA_PATH
        ).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        Path(
            METADATA_PATH
        ).write_bytes(content)


# ============================================================
# BILD-URL
# ============================================================

def image_url(path):

    if not github_enabled():
        return None

    return (
        "https://raw.githubusercontent.com/"
        f"{GITHUB_REPO}/"
        f"{GITHUB_BRANCH}/"
        f"{path}"
    )


# ============================================================
# FUNDSTÜCK HINZUFÜGEN
# ============================================================

def add_item(
    image_bytes,
    category,
    confidence
):

    now = datetime.now(
        timezone.utc
    )

    timestamp = now.isoformat()

    filename = (
        now.strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + ".jpg"
    )

    image_path = (
        f"{IMAGE_DIR}/{filename}"
    )

    # Bild öffnen
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # JPEG erstellen
    buffer = io.BytesIO()

    image.save(
        buffer,

        format="JPEG",

        quality=88,

        optimize=True
    )

    jpeg_bytes = buffer.getvalue()

    # --------------------------------------------------------
    # BILD SPEICHERN
    # --------------------------------------------------------

    if github_enabled():

        github_write_file(
            image_path,

            jpeg_bytes,

            f"Fundgrube: {category} hinzufügen"
        )

    else:

        Path(
            image_path
        ).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        Path(
            image_path
        ).write_bytes(
            jpeg_bytes
        )

    # --------------------------------------------------------
    # METADATEN
    # --------------------------------------------------------

    items = read_items()

    items.append(
        {
            "id": filename,

            "timestamp": timestamp,

            "category": category,

            "confidence": round(
                confidence,
                4
            ),

            "image_path": image_path,
        }
    )

    # Nach Alter sortieren
    items.sort(
        key=lambda x:
        x.get(
            "timestamp",
            ""
        )
    )

    write_items(items)


# ============================================================
# NAVIGATION
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "Suche"


with st.sidebar:

    st.markdown(
        '<div class="menu-title">'
        'Fundgrube'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Katharineum zu Lübeck"
    )

    # Seite 1
    if st.button(
        "🔎 Suche",
        use_container_width=True
    ):

        st.session_state.page = "Suche"

        st.rerun()

    # Seite 2
    if st.button(
        "📷 Fundstück einstellen",
        use_container_width=True
    ):

        st.session_state.page = (
            "Fundstück einstellen"
        )

        st.rerun()

    # Seite 3
    if st.button(
        "🕘 Älteste Fundstücke",
        use_container_width=True
    ):

        st.session_state.page = (
            "Älteste Fundstücke"
        )

        st.rerun()

    st.markdown(
        '<div class="side-note">'
        'Menü'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

def create_header(
    title,
    color
):

    silhouette_path = (
        "assets/school_silhouette.png"
    )

    st.markdown(
        f"""
        <div class="topbar {color}">

            <div class="hamburger">

                <span></span>
                <span></span>
                <span></span>

            </div>

            <div class="title">
                {title}
            </div>

            <img
                class="silhouette"
                src="{silhouette_path}"
            >

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SEITE 1
# SUCHE
# ============================================================

if st.session_state.page == "Suche":

    create_header(
        "Fundgrube",
        "pink"
    )

    st.markdown(
        """
        <div class="search-area">

            <div
                style="
                    font-size:22px;
                    font-weight:600;
                "
            >
                Beschreibe dein verlorenes
                Kleidungsstück
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # Texteingabe
    query = st.text_input(
        "Suche",

        placeholder=(
            "z. B. Ich suche einen schwarzen Hoodie"
        ),

        label_visibility="collapsed"
    )

    # Alle Fundstücke laden
    items = read_items()

    # --------------------------------------------------------
    # TEXT AUSWERTEN
    # --------------------------------------------------------

    query_lower = query.lower()

    categories = {

        "Hosen": [
            "hose",
            "hosen",
            "jeans",
            "stoffhose"
        ],

        "Sporthosen": [
            "sporthose",
            "sporthosen",
            "trainingshose",
            "jogginghose",
            "sporthose"
        ],

        "Hoodies": [
            "hoodie",
            "hoodies",
            "kapuzenpullover",
            "kapuzenpulli"
        ],

        "Polohemden": [
            "polo",
            "polohemd",
            "polohemden",
            "poloshirt"
        ]
    }

    selected_category = None

    for category, words in categories.items():

        for word in words:

            if word in query_lower:

                selected_category = category

                break

        if selected_category:
            break

    # --------------------------------------------------------
    # ERGEBNISSE
    # --------------------------------------------------------

    if query:

        if selected_category:

            results = [

                item

                for item in items

                if item.get(
                    "category"
                ) == selected_category

            ]

            st.markdown(
                f"### Gefundene {selected_category}"
            )

        else:

            results = []

            st.info(
                "Ich konnte keine der vier "
                "Kategorien erkennen. "
                "Versuche beispielsweise "
                "Hose, Sporthose, Hoodie "
                "oder Polohemd."
            )

        if results:

            # Bei der Suche:
            # neuere passende Fundstücke zuerst

            results = sorted(
                results,

                key=lambda x:
                x.get(
                    "timestamp",
                    ""
                ),

                reverse=True
            )

            columns = st.columns(3)

            for index, item in enumerate(
                results
            ):

                with columns[
                    index % 3
                ]:

                    st.markdown(
                        '<div class="result-card">',
                        unsafe_allow_html=True
                    )

                    path = item.get(
                        "image_path"
                    )

                    url = image_url(
                        path
                    )

                    if url:

                        st.image(
                            url,
                            use_container_width=True
                        )

                    elif Path(
                        path
                    ).exists():

                        st.image(
                            path,
                            use_container_width=True
                        )

                    st.markdown(
                        f"""
                        <div class="category">
                            {item.get("category", "")}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    confidence = (
                        item.get(
                            "confidence",
                            0
                        )
                        * 100
                    )

                    st.markdown(
                        f"""
                        <div class="small">
                            KI-Sicherheit:
                            {confidence:.1f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

        elif selected_category:

            st.info(
                "Für diese Kategorie "
                "wurden noch keine Fundstücke "
                "eingestellt."
            )

    else:

        st.markdown(
            """
            <div class="hint">

                Zum Beispiel:
                „Ich suche einen schwarzen Hoodie“

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SEITE 2
# FUNDSTÜCK EINSTELLEN
# ============================================================

elif (
    st.session_state.page
    == "Fundstück einstellen"
):

    create_header(
        "Lade ein Bild hoch",
        "blue"
    )

    st.markdown(
        """
        <div class="upload-box">

            <div>

                <div class="upload-title">

                    Lade ein Foto aus
                    deiner Mediathek hoch

                </div>

                <div
                    style="
                        margin-top:10px;
                    "
                >

                    oder nimm direkt
                    ein Foto auf

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # Bild aus Galerie
    uploaded_file = st.file_uploader(
        "Lade ein Bild hoch",

        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],

        label_visibility="collapsed"
    )

    # Kamera
    camera_file = st.camera_input(
        "Foto aufnehmen"
    )

    # Kamera hat Vorrang
    selected_file = (
        camera_file
        if camera_file is not None
        else uploaded_file
    )

    # --------------------------------------------------------
    # FOTO VORHANDEN
    # --------------------------------------------------------

    if selected_file is not None:

        image_bytes = (
            selected_file.getvalue()
        )

        preview = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert("RGB")

        st.image(
            preview,

            caption="Dein Fundstück",

            use_container_width=True
        )

        # ----------------------------------------------------
        # KI
        # ----------------------------------------------------

        with st.spinner(
            "Die KI analysiert das Foto ..."
        ):

            try:

                category, confidence = (
                    classify_image(
                        preview
                    )
                )

            except Exception as error:

                st.error(
                    "Die KI konnte das "
                    "Bild nicht analysieren:"
                )

                st.code(
                    str(error)
                )

                st.stop()

        # ----------------------------------------------------
        # KI ERGEBNIS
        # ----------------------------------------------------

        st.success(
            f"Die KI erkennt: "
            f"**{category}** "
            f"({confidence * 100:.1f}% Sicherheit)"
        )

        st.write(
            "Das Foto wird erst gespeichert, "
            "wenn du auf den Button klickst."
        )

        # ----------------------------------------------------
        # SPEICHERN
        # ----------------------------------------------------

        if st.button(
            "Fundstück speichern",
            use_container_width=True
        ):

            try:

                add_item(
                    image_bytes,

                    category,

                    confidence
                )

                st.success(
                    "Fundstück erfolgreich "
                    "gespeichert!"
                )

            except Exception as error:

                st.error(
                    "Das Fundstück konnte "
                    "nicht gespeichert werden."
                )

                st.code(
                    str(error)
                )


# ============================================================
# SEITE 3
# ÄLTESTE 9 FUNDSTÜCKE
# ============================================================

else:

    create_header(
        "Älteste Fundstücke",
        "pink"
    )

    st.markdown(
        "### Die 9 ältesten eingestellten Fundstücke"
    )

    items = read_items()

    # --------------------------------------------------------
    # WICHTIG:
    #
    # Auf dieser Seite werden die ÄLTESTEN
    # 9 Fundstücke angezeigt.
    #
    # Nicht die neuesten!
    # --------------------------------------------------------

    oldest_items = sorted(
        items,

        key=lambda x:
        x.get(
            "timestamp",
            ""
        )
    )[:9]

    if not oldest_items:

        st.info(
            "Es wurden noch keine "
            "Fundstücke eingestellt."
        )

    else:

        columns = st.columns(3)

        for index, item in enumerate(
            oldest_items
        ):

            with columns[
                index % 3
            ]:

                path = item.get(
                    "image_path"
                )

                url = image_url(
                    path
                )

                if url:

                    st.image(
                        url,
                        use_container_width=True
                    )

                elif Path(
                    path
                ).exists():

                    st.image(
                        path,
                        use_container_width=True
                    )

                st.markdown(
                    f"**{item.get('category', 'Unbekannt')}**"
                )

                timestamp = item.get(
                    "timestamp",
                    ""
                )

                st.caption(
                    f"Eingestellt: {timestamp}"
                )
