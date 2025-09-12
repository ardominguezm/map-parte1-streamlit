# === Branding en barra lateral, anclado abajo sin overlays ===
import base64

LOGO_CANDIDATES = ["assets/logo_3is.png"]

def _logo_base64():
    for p in LOGO_CANDIDATES:
        if Path(p).exists():
            with open(p, "rb") as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    return None

def _sidebar_css():
    st.sidebar.markdown(
        """
        <style>
        /* Convierte el contenedor del sidebar en un flex-column a toda altura */
        section[data-testid="stSidebar"] > div {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        /* Empuja el bloque de marca al fondo */
        .sidebar-spacer { flex: 1 1 auto; }
        .sidebar-brand {
            padding: 10px 12px;
            border-top: 1px solid #e5e7eb;
            font-size: 13px;
            color: #374151;
            background: transparent;
        }
        .sidebar-brand img {
            width: 120px;            /* ajusta tamaño del logo aquí (100–140) */
            display: block;
            margin: 0 auto 6px;
        }
        .sidebar-brand .center { text-align: center; line-height: 1.2; }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_sidebar_brand(author_name: str, author_email: str):
    _sidebar_css()
    # espacio flexible arriba (empuja hacia el fondo)
    st.sidebar.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)

    src = _logo_base64()
    img_tag = f'<img src="{src}" alt="logo"/>' if src else ""
    html = f"""
    <div class="sidebar-brand">
        {img_tag}
        <div class="center">
            <strong>{author_name}</strong><br/>
            <a href="mailto:{author_email}">{author_email}</a>
        </div>
    </div>
    """
    st.sidebar.markdown(html, unsafe_allow_html=True)
