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
# FUNDGRUBE – Katharineum zu Lübeck
# ============================================================

APP_TITLE = "Fundgrube"

MODEL_PATH = Path("keras_model.h5")
LABELS_PATH = Path("labels.txt")

# DEINE EIGENEN DESIGN-BILDER
FRONT_IMAGE = Path("assets/Frontseite.jpg")
UPLOAD_IMAGE = Path("assets/Seite2.jpg")
OLDEST_IMAGE = Path("assets/letzteSeite.jpg")

METADATA_PATH = "data/items.json"
IMAGE_DIR = "data/images"

IMAGE_SIZE = (224, 224)


# ============================================================
# BEKANNTE KLASSEN
# ============================================================

KNOWN_CATEGORIES = {
    "Hosen": [
        "hose",
        "hosen",
        "jeans",
        "stoffhose",
    ],

    "Sporthosen": [
        "sporthose",
        "sporthosen",
        "trainingshose",
        "jogginghose",
    ],

    "Hoodies": [
        "hoodie",
        "hoodies",
        "kapuzenpullover",
        "kapuzenpulli",
    ],

    "Polohemden": [
        "polo",
        "polohemd",
        "polohemden",
        "poloshirt",
    ],
}


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Fundgrube – Katharineum zu Lübeck",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# BILDER ALS BASE64 LADEN
# ============================================================

def image_to_data_uri(path):
    """
    Wandelt ein Bild aus assets/ in eine Data-URI um.
    Dadurch kann es direkt als Hintergrundbild verwendet werden.
    """

    if not path.exists():
        return None

    try:
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")

        suffix = path.suffix.lower()

        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        return f"data:{mime};base64,{encoded}"

    except Exception:
        return None


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    html, body, [class*="css"] {
        font-family: Arial, Helvetica, sans-serif;
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

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 0;
        padding-left: 0;
        padding-right: 0;
        padding-bottom: 0;
    }


    /* ========================================================
       DESIGN-HINTERGRUND
       ======================================================== */

    .design-page {
        position: relative;
        width: 100%;
        min-height: 850px;

        background-repeat: no-repeat;
        background-position: top center;
        background-size: 100% auto;

        overflow: hidden;
    }


    /* ========================================================
       MENÜ
       ======================================================== */

    .menu-area {
        position: fixed;
        top: 12px;
        left: 15px;
        z-index: 9999;
    }


    /* ========================================================
       STREAMLIT BUTTONS
       ======================================================== */

    div.stButton > button {
        border: 2px solid #111;
        border-radius: 999px;
        background: white;
        color: black;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background: #111;
        color: white;
    }


    /* ========================================================
       SUCHFELD
       ======================================================== */

    div[data-baseweb="input"] {
        border: 2px solid #111;
        border-radius: 999px;
        background: white;
    }

    div[data-baseweb="input"]:focus-within {
        border: 2px solid #111;
        box-shadow: none;
    }


    /* ========================================================
       UPLOAD
       ======================================================== */

    [data-testid="stFileUploader"] {
        background: transparent;
        border: none;
    }


    /* ========================================================
       KARTEN
       ======================================================== */

    .item-card {
        background: white;
        border: 2px solid #111;
        border-radius: 20px;
        padding: 10px;
        margin-bottom: 20px;
    }


    /* ========================================================
       UNSICHTBARE/TRANSPARENTE BEREICHE
       ======================================================== */

    .overlay-space {
        height: 420px;
    }

    .small-space {
        height: 50px;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 800px) {

        .design-page {
            min-height: 700px;
            background-size: cover;
        }

        .block-container {
            padding-left: 8px;
            padding-right: 8px;
        }

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

    try:

        response = requests.get(
            url,
            headers=github_headers(),
            params={"ref": GITHUB_BRANCH},
            timeout=20,
        )

    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    data = response.json()

    if data.get("encoding") == "base64":

        try:

            return base64.b64decode(
                data.get("content", "")
            )

        except Exception:
            return None

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

    try:

        existing = requests.get(
            url,
            headers=github_headers(),
            params={"ref": GITHUB_BRANCH},
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

    except requests.RequestException as error:

        raise RuntimeError(
            f"Verbindung zu GitHub fehlgeschlagen: {error}"
        ) from error

    if response.status_code not in (200, 201):

        raise RuntimeError(
            "GitHub-Fehler: "
            f"{response.status_code}\n"
            f"{response.text}"
        )

    return response.json()


# ============================================================
# KI-MODELL
# ============================================================

@st.cache_resource
def load_model():

    try:

        import tensorflow as tf

    except ImportError as error:

        raise RuntimeError(
            "TensorFlow ist nicht installiert."
        ) from error

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "keras_model.h5 wurde nicht gefunden."
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )


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

    text = LABELS_PATH.read_text(
        encoding="utf-8"
    )

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        match = re.match(
            r"^\s*\d+\s+(.+)$",
            line
        )

        if match:

            labels.append(
                match.group(1).strip()
            )

        else:

            labels.append(line)

    return labels


# ============================================================
# BILD KLASSIFIZIEREN
# ============================================================

def classify_image(image):

    model = load_model()
    labels = load_labels()

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.asarray(
        image
    ).astype(np.float32)

    image_array = (
        image_array / 127.0
    ) - 1.0

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    prediction = model.predict(
        image_array,
        verbose=0,
    )[0]

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
# DATEN LESEN
# ============================================================

def read_items():

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
                pass

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
            pass

    return []


# ============================================================
# DATEN SCHREIBEN
# ============================================================

def write_items(items):

    content = json.dumps(
        items,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    if github_enabled():

        github_write_file(
            METADATA_PATH,
            content,
            "Fundgrube: Metadaten aktualisieren",
        )

        return

    local_file = Path(
        METADATA_PATH
    )

    local_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_file.write_bytes(
        content
    )


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

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=88,
        optimize=True,
    )

    jpeg_bytes = buffer.getvalue()

    if github_enabled():

        github_write_file(
            image_path,
            jpeg_bytes,
            f"Fundgrube: {category} hinzufügen",
        )

    else:

        local_path = Path(
            image_path
        )

        local_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        local_path.write_bytes(
            jpeg_bytes
        )

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

    items.sort(
        key=lambda item: item.get(
            "timestamp",
            ""
        )
    )

    write_items(items)


# ============================================================
# FUNDSTÜCK-BILD ANZEIGEN
# ============================================================

def show_item_image(item):

    path = item.get(
        "image_path"
    )

    if not path:

        st.info(
            "Kein Foto vorhanden."
        )

        return

    url = image_url(path)

    if url:

        st.image(
            url,
            use_container_width=True
        )

        return

    local_path = Path(path)

    if local_path.exists():

        st.image(
            str(local_path),
            use_container_width=True
        )

    else:

        st.info(
            "Foto konnte nicht geladen werden."
        )


# ============================================================
# SEITENWECHSEL
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "suche"


# ============================================================
# MENÜ
# ============================================================

def menu():

    st.markdown(
        """
        <div class="menu-area">
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.popover(
        "☰",
        use_container_width=False
    ):

        st.markdown(
            "### Fundgrube"
        )

        st.caption(
            "Katharineum zu Lübeck"
        )

        if st.button(
            "Suche",
            use_container_width=True,
            key="nav_suche"
        ):

            st.session_state.page = (
                "suche"
            )

            st.rerun()

        if st.button(
            "Fundstück einstellen",
            use_container_width=True,
            key="nav_upload"
        ):

            st.session_state.page = (
                "upload"
            )

            st.rerun()

        if st.button(
            "Älteste Fundstücke",
            use_container_width=True,
            key="nav_oldest"
        ):

            st.session_state.page = (
                "oldest"
            )

            st.rerun()


# ============================================================
# SEITE 1 – FRONTSEITE
# ============================================================

def page_search():

    background = image_to_data_uri(
        FRONT_IMAGE
    )

    if background:

        st.markdown(
            f"""
            <div
                class="design-page"
                style="
                    background-image:
                    url('{background}');
                "
            >
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.error(
            "assets/Frontseite.jpg wurde nicht gefunden."
        )

    menu()

    # Suchfeld
    st.markdown(
        "<div style='height:0px'></div>",
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Suche",
        placeholder=(
            "Beschreibe dein verlorenes Kleidungsstück"
        ),
        label_visibility="collapsed",
        key="search_input",
    )

    items = read_items()

    if not query:

        return

    query_lower = query.lower()

    selected_category = None

    # Kategorien aus dem Code
    for category, words in KNOWN_CATEGORIES.items():

        for word in words:

            if word in query_lower:

                selected_category = category

                break

        if selected_category:
            break

    # Zusätzlich labels.txt
    if selected_category is None:

        try:

            labels = load_labels()

            for label in labels:

                if label.lower() in query_lower:

                    selected_category = label

                    break

        except Exception:
            pass

    if selected_category is None:

        st.warning(
            "Ich konnte keine passende Klasse erkennen. "
            "Versuche zum Beispiel: Hose, Sporthose, "
            "Hoodie oder Polohemd."
        )

        return

    results = [
        item
        for item in items
        if item.get("category")
        == selected_category
    ]

    results = sorted(
        results,
        key=lambda item: item.get(
            "timestamp",
            ""
        ),
        reverse=True,
    )

    st.markdown(
        f"### {len(results)} passende Fundstücke"
    )

    if not results:

        st.info(
            "Für diese Kategorie wurden noch keine "
            "Fundstücke eingestellt."
        )

        return

    columns = st.columns(3)

    for index, item in enumerate(results):

        with columns[
            index % 3
        ]:

            with st.container(
                border=True
            ):

                show_item_image(
                    item
                )

                st.markdown(
                    f"**{item.get('category', 'Unbekannt')}**"
                )

                confidence = (
                    float(
                        item.get(
                            "confidence",
                            0
                        )
                    )
                    * 100
                )

                st.caption(
                    f"KI-Sicherheit: "
                    f"{confidence:.1f}%"
                )


# ============================================================
# SEITE 2 – UPLOAD
# ============================================================

def page_upload():

    background = image_to_data_uri(
        UPLOAD_IMAGE
    )

    if background:

        st.markdown(
            f"""
            <div
                class="design-page"
                style="
                    background-image:
                    url('{background}');
                "
            >
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.error(
            "assets/Seite2.jpg wurde nicht gefunden."
        )

    menu()

    uploaded_file = st.file_uploader(
        "Bild hochladen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        label_visibility="collapsed",
        key="upload_file",
    )

    if uploaded_file is None:

        return

    image_bytes = (
        uploaded_file.getvalue()
    )

    try:

        preview = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert("RGB")

    except Exception as error:

        st.error(
            f"Bild konnte nicht geöffnet werden: "
            f"{error}"
        )

        return

    st.image(
        preview,
        width=450
    )

    with st.spinner(
        "Die KI analysiert das Bild ..."
    ):

        try:

            category, confidence = (
                classify_image(
                    preview
                )
            )

        except Exception as error:

            st.error(
                "Die KI konnte das Bild "
                "nicht analysieren."
            )

            st.code(
                str(error)
            )

            return

    st.markdown(
        f"### Erkannte Klasse: {category}"
    )

    st.write(
        f"KI-Sicherheit: "
        f"{confidence * 100:.1f}%"
    )

    if st.button(
        "Fundstück speichern",
        type="primary",
        use_container_width=True,
        key="save_item"
    ):

        try:

            add_item(
                image_bytes,
                category,
                confidence
            )

            st.success(
                "Fundstück wurde erfolgreich "
                "gespeichert."
            )

        except Exception as error:

            st.error(
                "Fundstück konnte nicht gespeichert werden."
            )

            st.code(
                str(error)
            )


# ============================================================
# SEITE 3 – DIE 9 ÄLTESTEN FUNDSTÜCKE
# ============================================================

def page_oldest():

    background = image_to_data_uri(
        OLDEST_IMAGE
    )

    if background:

        st.markdown(
            f"""
            <div
                class="design-page"
                style="
                    background-image:
                    url('{background}');
                "
            >
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.error(
            "assets/letzteSeite.jpg wurde nicht gefunden."
        )

    menu()

    items = read_items()

    # ========================================================
    # WICHTIG:
    # Die ältesten neun werden angezeigt.
    # NICHT die neuesten neun.
    # ========================================================

    oldest_items = sorted(
        items,
        key=lambda item: item.get(
            "timestamp",
            ""
        )
    )[:9]

    if not oldest_items:

        st.info(
            "Es wurden noch keine Fundstücke eingestellt."
        )

        return

    columns = st.columns(3)

    for index, item in enumerate(
        oldest_items
    ):

        with columns[
            index % 3
        ]:

            with st.container(
                border=True
            ):

                show_item_image(
                    item
                )

                st.markdown(
                    f"**{item.get('category', 'Unbekannt')}**"
                )

                confidence = (
                    float(
                        item.get(
                            "confidence",
                            0
                        )
                    )
                    * 100
                )

                st.caption(
                    f"KI-Sicherheit: "
                    f"{confidence:.1f}%"
                )

                timestamp = item.get(
                    "timestamp",
                    ""
                )

                try:

                    dt = datetime.fromisoformat(
                        timestamp.replace(
                            "Z",
                            "+00:00"
                        )
                    )

                    readable = (
                        dt.astimezone().strftime(
                            "%d.%m.%Y – %H:%M"
                        )
                    )

                except Exception:

                    readable = timestamp

                st.caption(
                    f"Eingestellt: {readable}"
                )


# ============================================================
# APP STARTEN
# ============================================================

if st.session_state.page == "suche":

    page_search()

elif st.session_state.page == "upload":

    page_upload()

elif st.session_state.page == "oldest":

    page_oldest()

else:

    st.session_state.page = "suche"

    st.rerun()
