import streamlit as st


# ── Brand colours ──────────────────────────────────────────────────────
MONGO_GREEN   = "#00ED64"
MONGO_FOREST  = "#00C853"     # brighter green for dark backgrounds
MONGO_SLATE   = "#001E2B"
PURPLE        = "#BB86FC"
PURPLE_DK     = "#9B59F0"

# ── Dark-theme surfaces ───────────────────────────────────────────────
BG_MAIN   = "#0E1117"   # app background (near-black)
BG_SEC    = "#141922"   # secondary background
SURFACE   = "#1A1F2E"   # cards / inputs / sidebar
SURFACE_H = "#242B3D"   # hover / raised surface
BORDER    = "#2D3548"   # subtle borders
TEXT_MAIN = "#E8ECF1"   # primary text (off-white)
TEXT_MUTE = "#8B95A5"   # secondary text
ACCENT    = MONGO_GREEN  # primary accent


def apply_theme():
    """Inject global CSS to skin the app in a dark, high-contrast palette."""
    st.markdown(
        f"""
        <style>
        /* ---- Global font ---- */
        @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman');
        html, body, .stApp, .stApp *,
        section[data-testid="stSidebar"] *,
        [data-testid="stChatMessage"] *,
        input, textarea, select, button, th, td, label {{
            font-family: 'Times New Roman', Times, Georgia, serif !important;
        }}
        /* Preserve Material Symbols icon font so icon ligatures render as
           icons (not literal text like "upload" / "keyboard_double_arrow_right") */
        span[data-testid="stIconMaterial"],
        [data-testid="stIconMaterial"],
        .material-symbols-rounded,
        .material-symbols-outlined,
        [class*="material-symbols"],
        [class*="material-icons"] {{
            font-family: 'Material Symbols Rounded',
                         'Material Symbols Outlined',
                         'Material Icons' !important;
            font-weight: normal !important;
            font-style: normal !important;
            text-transform: none !important;
            letter-spacing: normal !important;
            word-wrap: normal !important;
            white-space: nowrap !important;
            direction: ltr !important;
            -webkit-font-feature-settings: 'liga' !important;
            font-feature-settings: 'liga' !important;
            -webkit-font-smoothing: antialiased !important;
        }}

        /* ---- App background: dark ---- */
        .stApp {{
            background: {BG_MAIN};
            color: {TEXT_MAIN};
        }}

        # /* ---- Hide the top-right menu & toolbar (removes the Light/Dark/
        #        system theme switcher, which lives in Settings) ---- */
        # #MainMenu {{ visibility: hidden !important; display: none !important; }}
        # [data-testid="stMainMenu"] {{ display: none !important; }}
        # [data-testid="stToolbar"] {{ display: none !important; }}
        # [data-testid="stToolbarActions"] {{ display: none !important; }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {{
            background: {SURFACE};
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] * {{ color: {TEXT_MAIN}; }}

        /* ---- Headings ---- */
        h1, h2, h3 {{ color: #FFFFFF; font-weight: 700; }}

        /* ---- Body text ---- */
        .stApp, .stApp p, .stApp li, .stApp span, .stApp label,
        .stMarkdown, .stMarkdown p, .stMarkdown li {{
            color: {TEXT_MAIN};
        }}
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: {TEXT_MUTE} !important;
        }}

        /* ---- Primary buttons: blue bg, white text — high contrast ---- */
        .stButton > button {{
            background: #3B82F6 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px;
            font-weight: 700;
            transition: all 0.15s ease;
        }}
        .stButton > button:hover {{
            background: #1D4ED8 !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(59,130,246,0.45);
        }}
        .stButton > button:active {{
            background: #1E40AF !important;
            color: #FFFFFF !important;
        }}
        /* button icon / svg visibility */
        .stButton > button svg {{
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
        }}

        /* ---- Download buttons ---- */
        .stDownloadButton > button {{
            background: #3B82F6 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px;
            font-weight: 700;
        }}
        .stDownloadButton > button:hover {{
            background: #1D4ED8 !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 14px rgba(59,130,246,0.45);
        }}

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            border-bottom: 1px solid {BORDER};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {TEXT_MUTE};
            font-weight: 500;
        }}
        .stTabs [aria-selected="true"] {{
            color: {ACCENT} !important;
            border-bottom: 2px solid {ACCENT};
        }}

        /* ---- Metric cards ---- */
        div[data-testid="stMetric"] {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 14px 16px;
        }}
        div[data-testid="stMetricValue"] {{ color: {ACCENT} !important; }}
        div[data-testid="stMetricLabel"] {{ color: {TEXT_MUTE}; }}

        /* ---- Chat messages ---- */
        [data-testid="stChatMessage"] {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 6px 14px;
            margin-bottom: 8px;
        }}
        [data-testid="stChatMessage"] * {{
            color: {TEXT_MAIN} !important;
        }}
        [data-testid="stChatMessage"] code {{ color: {ACCENT} !important; }}
        [data-testid="stChatMessage"] a {{ color: {ACCENT} !important; }}

        /* ---- Chat input ---- */
        [data-testid="stChatInput"] textarea {{
            color: {TEXT_MAIN} !important;
            background: {SURFACE} !important;
            border: 1px solid {BORDER} !important;
        }}
        [data-testid="stChatInput"] textarea::placeholder {{
            color: {TEXT_MUTE} !important;
        }}

        /* ---- Text / number inputs ---- */
        .stTextInput input, .stNumberInput input, .stTextArea textarea {{
            background-color: {SURFACE} !important;
            color: {TEXT_MAIN} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 8px;
        }}
        .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
            color: {TEXT_MUTE} !important;
        }}

        /* ---- File uploader ---- */
        [data-testid="stFileUploader"] {{
            background-color: {SURFACE} !important;
            border: 2px dashed {BORDER} !important;
            border-radius: 10px;
            padding: 16px !important;
        }}
        [data-testid="stFileUploader"] * {{
            color: {TEXT_MAIN} !important;
        }}
        [data-testid="stFileUploader"] section {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }}
        [data-testid="stFileUploader"] section > button {{
            background: #3B82F6 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px;
            font-weight: 700;
            padding: 0.5rem 2rem !important;
            min-height: 42px;
            font-size: 15px !important;
            cursor: pointer;
        }}
        [data-testid="stFileUploader"] section > button:hover {{
            background: #1D4ED8 !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 14px rgba(59,130,246,0.45);
        }}
        /* drag-and-drop label text */
        [data-testid="stFileUploader"] small {{
            color: {TEXT_MUTE} !important;
            font-size: 13px !important;
        }}
        /* uploaded file name chip */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {{
            background-color: {SURFACE_H} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 6px;
        }}
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {{
            color: {TEXT_MAIN} !important;
        }}
        /* delete-file button inside the chip */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button {{
            background: transparent !important;
            color: {TEXT_MUTE} !important;
            border: none !important;
            min-height: auto;
            padding: 2px !important;
            font-size: 13px !important;
        }}
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button:hover {{
            color: #EF4444 !important;
            background: transparent !important;
            box-shadow: none;
        }}

        /* ---- Selectbox / multiselect ---- */
        div[data-baseweb="select"] > div {{
            background-color: {SURFACE} !important;
            border: 1px solid {BORDER} !important;
            color: {TEXT_MAIN} !important;
        }}
        div[data-baseweb="select"] * {{ color: {TEXT_MAIN} !important; }}

        /* dropdown popover list */
        ul[data-baseweb="menu"], div[data-baseweb="popover"] ul {{
            background-color: {SURFACE_H} !important;
        }}
        div[data-baseweb="popover"] {{
            background-color: {SURFACE_H} !important;
        }}
        ul[data-baseweb="menu"] li {{
            background-color: {SURFACE_H} !important;
            color: {TEXT_MAIN} !important;
        }}
        ul[data-baseweb="menu"] li:hover {{
            background-color: #1B3A2A !important;
            color: {ACCENT} !important;
        }}

        /* multiselect chips */
        span[data-baseweb="tag"] {{
            background-color: #1B3A2A !important;
            color: {ACCENT} !important;
            border: 1px solid {ACCENT} !important;
        }}
        span[data-baseweb="tag"] * {{ color: {ACCENT} !important; }}

        /* ---- Dataframes / tables ---- */
        [data-testid="stDataFrame"] {{
            background-color: {SURFACE} !important;
            border: 1px solid {BORDER};
            border-radius: 8px;
        }}
        [data-testid="stDataFrame"] * {{ color: {TEXT_MAIN} !important; }}
        [data-testid="stTable"] table {{ color: {TEXT_MAIN} !important; }}
        [data-testid="stTable"] th {{
            background-color: {SURFACE_H} !important;
            color: #FFFFFF !important;
        }}
        [data-testid="stTable"] td {{
            background-color: {SURFACE} !important;
            color: {TEXT_MAIN} !important;
            border-color: {BORDER} !important;
        }}

        /* ---- Expanders ---- */
        [data-testid="stExpander"] {{
            border: 1px solid {BORDER} !important;
            border-radius: 8px;
            background-color: {SURFACE} !important;
        }}
        [data-testid="stExpander"] summary {{
            color: {ACCENT} !important;
        }}
        [data-testid="stExpander"] summary:hover {{
            color: #33FF88 !important;
        }}
        details[data-testid="stExpander"] > div {{
            color: {TEXT_MAIN} !important;
        }}

        /* ---- Toggle / slider / radio / checkbox ---- */
        [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {{
            color: {TEXT_MAIN} !important;
        }}
        /* slider track and thumb */
        .stSlider [data-baseweb="slider"] div[role="slider"] {{
            background: {ACCENT} !important;
        }}

        /* ---- Status / alert boxes ---- */
        [data-testid="stAlert"] {{
            background-color: {SURFACE} !important;
            color: {TEXT_MAIN} !important;
            border: 1px solid {BORDER} !important;
        }}
        [data-testid="stAlert"] * {{
            color: {TEXT_MAIN} !important;
        }}

        /* ---- Progress bar ---- */
        .stProgress > div > div > div {{
            background-color: {ACCENT} !important;
        }}

        /* ---- Status widget ---- */
        [data-testid="stStatusWidget"] {{
            background-color: {SURFACE} !important;
            border: 1px solid {BORDER} !important;
        }}
        [data-testid="stStatusWidget"] * {{
            color: {TEXT_MAIN} !important;
        }}

        /* ---- Links / code ---- */
        a {{ color: {ACCENT}; }}
        a:hover {{ color: #33FF88; }}
        code {{
            color: {ACCENT};
            background-color: {SURFACE_H} !important;
        }}

        /* ---- Tooltips ---- */
        [data-testid="stTooltipIcon"] svg {{
            fill: {TEXT_MUTE} !important;
        }}

        /* ---- All SVG icons: ensure visibility ---- */
        .stApp svg {{
            fill: {TEXT_MAIN};
        }}
        section[data-testid="stSidebar"] svg {{
            fill: {TEXT_MAIN};
        }}

        /* ---- Scrollbar ---- */
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: {BG_MAIN}; }}
        ::-webkit-scrollbar-thumb {{
            background: {BORDER};
            border-radius: 6px;
        }}
        ::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}

        /* ---- Dividers ---- */
        hr {{ border-color: {BORDER} !important; }}

        /* ---- Form submit buttons ---- */
        .stForm [data-testid="stFormSubmitButton"] > button {{
            background: #3B82F6 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px;
            font-weight: 700;
        }}
        .stForm [data-testid="stFormSubmitButton"] > button:hover {{
            background: #1D4ED8 !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 14px rgba(59,130,246,0.45);
        }}

        /* ---- Column alignment ---- */
        [data-testid="stHorizontalBlock"] {{
            align-items: flex-start;
        }}
        [data-testid="stColumn"] > div {{
            display: flex;
            flex-direction: column;
            height: 100%;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header():
    st.markdown(
        f"""
        <div style="
            display:flex; align-items:center; justify-content:space-between;
            padding:14px 20px; margin:-8px 0 18px 0;
            background: linear-gradient(90deg, rgba(0,237,100,0.08) 0%, rgba(187,134,252,0.06) 100%);
            border:1px solid {BORDER};
            border-left:4px solid {ACCENT};
            border-radius:12px;">
          <div style="display:flex; align-items:center; gap:14px;">
            <span style="font-size:26px;">🔍</span>
            <div>
              <div style="font-size:22px; font-weight:800; color:#FFFFFF; letter-spacing:0.3px;">
                LegacyLens
              </div>
              <div style="font-size:12px; color:{TEXT_MUTE};">
                Talk to your legacy codebase
              </div>
            </div>
          </div>
          <div style="display:flex; align-items:center; gap:14px; font-size:13px;">
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )