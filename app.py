import streamlit as st
from story_generator import (
    extract_scene_from_images,
    extract_main_characters,
    generate_story_from_images,
    generate_story_with_critic,
    narrate_story,
    token_monitor,
    build_emotional_arc,
    score_story_coherence,
    parse_cited_story,
    generate_hallucination_report,
    story_memory,
    stream_story_token_level,
    stream_story_ui_level,
    StreamingMode,
    get_streaming_explanation,
    run_attention_analysis,
    format_analysis_report,
    format_critique_report,
    REGENERATION_THRESHOLD,
)
from PIL import Image
import time
import uuid


# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Story Generator",
    page_icon="📖",
    layout="wide"
)

# ---------------------------------------------------
# Session State Init
# ---------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "last_scene_data" not in st.session_state:
    st.session_state.last_scene_data = None
if "last_character_map" not in st.session_state:
    st.session_state.last_character_map = None
if "last_emotional_arc" not in st.session_state:
    st.session_state.last_emotional_arc = None
if "last_story" not in st.session_state:
    st.session_state.last_story = None
if "last_critique" not in st.session_state:
    st.session_state.last_critique = None

# ---------------------------------------------------
# UI Logger Bridge
# ---------------------------------------------------
class StreamlitLogger:
    @staticmethod
    def log(level, message):
        if level == "info":
            st.info(f"ℹ️ {message}")
        elif level == "warning":
            st.warning(f"⚠️ {message}")
        elif level == "error":
            st.error(f"❌ {message}")
        elif level == "success":
            st.success(f"😎 {message}")

ui_logger = StreamlitLogger()


# ---------------------------------------------------
# Advanced Mode Panel Helper Functions
# ---------------------------------------------------
def show_architecture_panel():
    """Clean architecture breakdown — neutral, professional."""
    with st.expander("🏗️ AI System Architecture", expanded=True):
        st.markdown("""
**Level 1 — Core Engineering**
- Context-aware character mapping (gender + age)
- Real token monitoring (tiktoken)
- Structured logging (file + console)
- Auto-retry failure recovery

**Level 2 — Narrative Intelligence**
- Scene-level emotional arc tracking
- Citation enforcement (hallucination reduction)
- Multimodal memory across sessions

**Level 3 — Research Extensions**
- Structured JSON schema output
- True token-level streaming
- Ablation study (3-strategy comparison)

**Level 4 — Self-Refinement**
- Story → Critic → Improved Story loop
        """)

        st.markdown("---")
        st.markdown("**Pipeline Flow**")
        st.code("""\
Images
 └─▶ Level 1: Vision Analysis (Gemini Vision)
       ├─ Scene JSON extraction
       ├─ Retry + logging
       └─ Token monitoring
           ↓
   Level 2: Character & Narrative Modeling
       ├─ Gender-aware character mapping
       ├─ Emotional arc modeling
       └─ Memory injection (optional)
           ↓
   Level 3: Story Generation (Gemini 2.5 Flash)
       ├─ Citation enforcement (optional)
       ├─ Structured output (optional)
       ├─ Self-healing validation
       └─ Token tracking
           ↓
   Level 4: Refinement & Output
       ├─ Critic scoring (optional)
       ├─ Regeneration if score too low
       ├─ Coherence scoring
       └─ Text-to-Speech (gTTS)""", language="text")


def show_streaming_explanation():
    """Token-level vs UI-level streaming comparison."""
    explanation = get_streaming_explanation()
    with st.expander("🌊 Streaming Architecture", expanded=True):
        col_tok, col_ui = st.columns(2)

        with col_tok:
            st.markdown("#### ⚡ Token-Level Streaming (Real)")
            st.success(
                f"**How it works:** {explanation['token_level']['description']}\n\n"
                f"**First token latency:** `{explanation['token_level']['latency']}`\n\n"
                f"**Use case:** {explanation['token_level']['use_case']}"
            )
            st.code(
                "for chunk in client.models\n"
                "    .generate_content_stream(prompt):\n"
                "    yield chunk.text   # live from API",
                language="python"
            )

        with col_ui:
            st.markdown("#### 🧪 UI-Level Streaming (Simulated)")
            st.warning(
                f"**How it works:** {explanation['ui_level']['description']}\n\n"
                f"**Latency:** `{explanation['ui_level']['latency']}`\n\n"
                f"**Use case:** {explanation['ui_level']['use_case']}"
            )
            st.code(
                "story = generate(prompt)   # waits\n"
                "for para in story.split():\n"
                "    display(para)\n"
                "    time.sleep(0.4)        # fake",
                language="python"
            )

        st.markdown(
            "> 💡 Token-level streaming reduces *perceived* latency because the user sees "
            "the first token in ~50–200 ms, before the full response is ready. "
            "UI-level simulation only hides the wait — the model has already finished generating."
        )


def show_ablation_controls():
    """
    Ablation study button — shown in sidebar only in Advanced Mode,
    only once scene data exists.
    Returns True if the button was pressed, False otherwise.
    """
    st.markdown("---")
    st.subheader("🔬 Advanced Tools")
    if st.session_state.last_scene_data:
        st.caption("Available after first story generation.")
        return st.button(
            "Run Ablation Study\n(3-strategy comparison)",
            use_container_width=True,
            help=(
                "Fires 3 Gemini calls with progressively richer prompts:\n"
                "A) Vision-only  B) Vision + Citations  C) Full Pipeline\n\n"
                "Compares coherence score, citation coverage %, "
                "hallucination risk, latency, and cost."
            )
        )
    else:
        st.caption("Ablation study unlocks after first story generation.")
        return False


# ---------------------------------------------------
# Helper: Emotional Arc Visualiser
# ---------------------------------------------------
def display_emotional_arc(emotional_arc: dict):
    """Render the emotional arc as a visual timeline in Streamlit."""
    st.subheader("Emotional Arc Analysis")

    arc = emotional_arc.get("arc", [])
    if not arc:
        st.info("No emotional data detected in scenes.")
        return

    valence_emoji = {
        "positive": "😊",
        "negative": "😨",
        "tense":    "😰",
        "neutral":  "😐"
    }

    cols = st.columns(len(arc))
    for i, entry in enumerate(arc):
        emoji = valence_emoji.get(entry["valence"], "❓")
        with cols[i]:
            st.metric(
                label=f"Scene {entry['scene']}",
                value=emoji,
                delta=entry["valence"]
            )
            st.caption(f"_{entry['emotion']}_")

    st.markdown(f"**Overall arc:** `{emotional_arc.get('arc_summary', 'N/A')}`")

    transitions = emotional_arc.get("transitions", [])
    if transitions:
        st.markdown("**Emotional transformations detected:**")
        for t in transitions:
            st.markdown(
                f"- Scene {t['from_scene']} → Scene {t['to_scene']}: "
                f"`{t['from_emotion']}` → `{t['to_emotion']}` ({t['shift']})"
            )
    else:
        st.info("Consistent emotional tone throughout — no major shifts detected.")


# ---------------------------------------------------
# Helper: Streaming Display
# ---------------------------------------------------
def display_with_streaming(prompt_or_story: str, mode: str, style: str,
                            is_prompt: bool = False) -> str:
    placeholder = st.empty()
    full_text   = ""

    if mode == StreamingMode.TOKEN_LEVEL:
        st.caption("🔴 **Token-level streaming active** — tokens arriving from Gemini API in real-time")
        for chunk in stream_story_token_level(prompt_or_story):
            full_text += chunk
            placeholder.markdown(f"**📖 {style} Story (streaming...)**\n\n{full_text}")
        placeholder.markdown(f"**📖 Your {style} Story**\n\n{full_text}")

    elif mode == StreamingMode.UI_LEVEL:
        st.caption("🟡 **UI-level streaming active** — paragraphs revealed progressively (simulated)")
        for chunk in stream_story_ui_level(prompt_or_story):
            full_text += chunk
            placeholder.markdown(f"**📖 {style} Story (loading...)**\n\n{full_text}")
            time.sleep(0.4)
        placeholder.markdown(f"**📖 Your {style} Story**\n\n{full_text}")

    else:
        full_text = prompt_or_story
        placeholder.markdown(f"**📖 Your {style} Story**\n\n{full_text}")

    return full_text.strip()


# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.title("📖 AI Story Generator")
st.markdown("Upload 1–10 images. The AI will analyse, remember, and narrate a story.")

# ---------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    uploaded_files = st.file_uploader(
        "Upload images (1–10)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    story_style = st.selectbox(
        "Story style",
        ("Comedy", "Thriller", "Fairy Tale", "Sci-Fi", "Mystery", "Adventure", "Morale")
    )
    story_tone = st.selectbox(
        "Story tone",
        ("Calm", "Dark", "Energetic", "Neutral")
    )
    narrative_structure = st.selectbox(
        "Narrative structure",
        ("Linear storytelling", "Hero's Journey", "Flashback-based", "Mystery reveal format")
    )

    st.markdown("---")
    st.subheader("⚙️ Advanced Features")

    advanced_mode = st.toggle(
        "Advanced Mode",
        value=False,
        help="Reveals architecture details, streaming explanation, and ablation study controls."
    )

    streaming_mode = st.radio(
        "Streaming mode",
        [StreamingMode.TOKEN_LEVEL, StreamingMode.UI_LEVEL, StreamingMode.DISABLED],
        format_func=lambda m: {
            StreamingMode.TOKEN_LEVEL: "⚡ Token-Level (Real)",
            StreamingMode.UI_LEVEL:    "🧪 UI-Level (Simulated)",
            StreamingMode.DISABLED:    "⏸️ Disabled"
        }[m],
        index=0
    )

    enable_citations    = st.checkbox("Hallucination reduction (cite scenes)", value=True)
    use_structured_out  = st.checkbox("Structured JSON output mode", value=False)
    enable_memory       = st.checkbox("Memory (continue previous story)", value=True)
    show_arc            = st.checkbox("Show emotional arc analysis", value=True)
    enable_critic_loop  = st.checkbox(
        "Story Critic Loop",
        value=False,
        help=(
            f"Runs a second Gemini call to score coherence, emotional depth, "
            f"and cultural accuracy (1–10 each). If the weighted overall score "
            f"is below {REGENERATION_THRESHOLD}/100, the story is automatically "
            f"regenerated using the critic's suggestions.\n\n"
            f"⚠️ Uses 2–3× more API calls. Best for final quality output."
        )
    )
    show_debug          = st.checkbox("Show debug / processing details", value=False)
    show_token_usage    = st.checkbox("Show token monitoring", value=True)
    enable_caching      = st.checkbox("Enable scene caching", value=True)

    st.markdown("---")

    # Session info + clear memory
    st.caption(f"🔑 Session ID: `{st.session_state.session_id}`")
    has_memory = story_memory.has_memory(st.session_state.session_id)
    if has_memory:
        st.success("🧠 Memory active — next story will continue the previous one")
        if st.button("🗑️ Clear Memory (Start Fresh)"):
            story_memory.clear(st.session_state.session_id)
            st.session_state.last_story = None
            st.rerun()
    else:
        st.info("🆕 No memory yet — first story of this session")

    # Token dashboard
    if show_token_usage:
        st.markdown("---")
        st.subheader("🤖 Token Dashboard")
        session_stats = token_monitor.get_session_stats()
        if session_stats["total_input_tokens"] > 0:
            st.metric("Input Tokens",  session_stats["total_input_tokens"])
            st.metric("Output Tokens", session_stats["total_output_tokens"])
            st.metric("Session Cost",  f"${session_stats['total_cost']:.4f}")
            st.caption(f"Method: `{session_stats['tokenization_method']}`")

    st.markdown("---")
    generate_button = st.button("🚀 Generate Story", type="primary", use_container_width=True)

    # Ablation study — only visible in Advanced Mode
    run_research = False
    if advanced_mode:
        run_research = show_ablation_controls()


# ---------------------------------------------------
# Advanced Mode Panels (main area)
# ---------------------------------------------------
if advanced_mode:
    col_arch, col_stream = st.columns(2)
    with col_arch:
        show_architecture_panel()
    with col_stream:
        show_streaming_explanation()
    st.markdown("---")


# ---------------------------------------------------
# MAIN LOGIC — Story Generation
# ---------------------------------------------------
if generate_button:
    if not uploaded_files:
        st.warning("Please upload at least 1 image.")
    elif len(uploaded_files) > 10:
        st.warning("Maximum 10 images allowed.")
    else:
        progress_bar = st.progress(0)
        status_text  = st.empty()

        with st.spinner("Running AI pipeline..."):
            try:
                # ── Level 1: Vision ──────────────────────────────────────
                status_text.text("Level 1: Analysing images with Gemini Vision...")
                progress_bar.progress(15)

                pil_images = [Image.open(f) for f in uploaded_files]

                st.subheader("📸 Uploaded Images")
                img_cols = st.columns(min(len(pil_images), 5))
                for i, img in enumerate(pil_images):
                    with img_cols[i % 5]:
                        st.image(img, use_container_width=True, caption=f"Scene {i+1}")

                scene_data = extract_scene_from_images(
                    pil_images,
                    logger_callback=ui_logger.log if show_debug else None,
                    use_cache=enable_caching
                )

                if isinstance(scene_data, dict) and "error" in scene_data:
                    st.error(f"❌ Scene extraction failed: {scene_data['error']}")
                    st.stop()

                st.session_state.last_scene_data = scene_data

                if show_debug:
                    with st.expander("🔍 Level 1 — Raw Scene JSON"):
                        st.json(scene_data)

                progress_bar.progress(30)

                # ── Level 2: Character Mapping ───────────────────────────
                status_text.text("Level 2: Building gender-aware character map...")
                character_map = extract_main_characters(scene_data)
                st.session_state.last_character_map = character_map
                progress_bar.progress(40)

                if show_debug:
                    with st.expander("👥 Level 2 — Character Map"):
                        st.json(character_map)

                # ── Emotional Arc ────────────────────────────────────────
                status_text.text("Computing emotional arc...")
                emotional_arc = build_emotional_arc(scene_data)
                st.session_state.last_emotional_arc = emotional_arc
                progress_bar.progress(50)

                if show_arc:
                    display_emotional_arc(emotional_arc)

                # ── Level 3: Story Generation ────────────────────────────
                status_text.text("Level 3: Generating story (this takes ~10–20s)...")
                progress_bar.progress(60)

                session_id_to_use = st.session_state.session_id if enable_memory else None

                # ── Critic Loop path ─────────────────────────────────────
                if enable_critic_loop and streaming_mode != StreamingMode.TOKEN_LEVEL:
                    status_text.text("Level 4: Story → Critic → Improved Story...")

                    critic_result = generate_story_with_critic(
                        scene_data,
                        story_style,
                        story_tone,
                        narrative_structure,
                        session_id=session_id_to_use,
                        enable_citations=enable_citations,
                    )

                    final_story = critic_result["final_story"]
                    st.session_state.last_critique = critic_result

                    st.subheader(f"📖 Your {story_style} Story")
                    if streaming_mode == StreamingMode.UI_LEVEL:
                        final_story = display_with_streaming(
                            final_story, StreamingMode.UI_LEVEL, story_style
                        )
                    else:
                        st.markdown(final_story)

                    # ── Critic Report ────────────────────────────────────
                    st.markdown("---")
                    critic_score = critic_result["critic_score"]
                    was_improved = critic_result["was_improved"]

                    score_color = (
                        "🟢" if critic_score >= 80
                        else ("🟡" if critic_score >= REGENERATION_THRESHOLD else "🔴")
                    )
                    badge = (
                        "✨ Story was auto-improved by the critic."
                        if was_improved
                        else "✅ Story passed quality threshold — no regeneration needed."
                    )

                    st.subheader("📋 Story Critic Report")
                    st.markdown(f"**Overall Score:** {score_color} `{critic_score:.1f}/100` — {badge}")

                    with st.expander("Full Critic Evaluation", expanded=was_improved):
                        st.markdown(critic_result["critique_report"])

                    if was_improved:
                        with st.expander("🔄 Before / After Comparison"):
                            col_before, col_after = st.columns(2)
                            with col_before:
                                st.markdown("**Original Story**")
                                st.markdown(critic_result["initial_story"])
                            with col_after:
                                st.markdown("**Critic-Improved Story**")
                                st.markdown(final_story)

                # ── Standard / Token-level path ──────────────────────────
                elif streaming_mode == StreamingMode.TOKEN_LEVEL and not use_structured_out:
                    from story_generator.narrative import build_contextual_prompt
                    from story_generator.memory import story_memory as _sm

                    mem_entry = _sm.load(session_id_to_use) if session_id_to_use else None
                    stream_prompt = build_contextual_prompt(
                        scene_data, character_map,
                        story_style, story_tone, narrative_structure,
                        emotional_arc=emotional_arc,
                        memory_entry=mem_entry,
                        enable_citations=enable_citations
                    )

                    st.subheader(f"📖 Your {story_style} Story")
                    final_story = display_with_streaming(
                        stream_prompt, StreamingMode.TOKEN_LEVEL,
                        story_style, is_prompt=True
                    )

                    if session_id_to_use:
                        _sm.save(session_id_to_use, scene_data,
                                 character_map, final_story, emotional_arc)

                else:
                    generated_story = generate_story_from_images(
                        scene_data,
                        story_style,
                        story_tone,
                        narrative_structure,
                        session_id=session_id_to_use,
                        enable_citations=enable_citations,
                        use_structured_output=use_structured_out
                    )

                    if "Error" in str(generated_story):
                        st.error(f"❌ {generated_story}")
                        st.stop()

                    st.subheader(f"📖 Your {story_style} Story")
                    final_story = display_with_streaming(
                        generated_story,
                        StreamingMode.UI_LEVEL if streaming_mode == StreamingMode.UI_LEVEL
                        else StreamingMode.DISABLED,
                        story_style
                    )

                st.session_state.last_story = final_story
                progress_bar.progress(75)

                # ── Coherence Score ──────────────────────────────────────
                coherence = score_story_coherence(final_story, emotional_arc)

                # ── Hallucination Report ─────────────────────────────────
                if enable_citations:
                    parsed_story = parse_cited_story(final_story, len(scene_data))
                    h_report     = generate_hallucination_report(parsed_story)
                else:
                    h_report = {"citation_coverage": 0, "risk_level": "N/A",
                                "total_flags": 0, "summary": "Citations disabled"}

                # ── Story Metrics ────────────────────────────────────────
                st.markdown("---")
                st.subheader("📊 Story Quality Metrics")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Words",              len(final_story.split()))
                m2.metric("Coherence Score",    f"{coherence['score']}/100")
                m3.metric("Citation Coverage",  f"{h_report['citation_coverage']}%")
                m4.metric("Hallucination Risk", h_report['risk_level'].upper())
                m5.metric("Style",              story_style)

                if enable_citations and h_report["total_flags"] > 0:
                    with st.expander(f"⚠️ {h_report['total_flags']} Hallucination Flag(s) Detected"):
                        for flag in h_report.get("flagged_items", []):
                            st.warning(f"**{flag['reason']}**\n> {flag['paragraph']}")

                # ── Memory Status ────────────────────────────────────────
                if enable_memory and story_memory.has_memory(st.session_state.session_id):
                    st.success("🧠 Story saved to memory — upload new images and generate to continue this story!")

                # ── Level 4: TTS ─────────────────────────────────────────
                status_text.text("Level 4: Generating audio narration...")
                progress_bar.progress(90)

                st.subheader("🎧 Audio Narration")
                audio_file = narrate_story(final_story)
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")
                else:
                    st.warning("⚠️ Audio generation failed — story is still available above.")

                progress_bar.progress(100)
                status_text.text("😇 Done!")

                # ── Token Summary ────────────────────────────────────────
                if show_token_usage:
                    stats = token_monitor.get_session_stats()
                    with st.sidebar:
                        st.markdown("---")
                        st.subheader("📊 Session Summary")
                        st.metric("Total Input",  stats["total_input_tokens"])
                        st.metric("Total Output", stats["total_output_tokens"])
                        st.metric("Total Cost",   f"${stats['total_cost']:.4f}")

            except Exception as e:
                st.error(f"❌ Pipeline Error: {e}")
                if show_debug:
                    st.exception(e)


# ---------------------------------------------------
# Ablation Study
# ---------------------------------------------------
if run_research and st.session_state.last_scene_data:
    st.markdown("---")
    st.header("📊 Ablation Study — 3-Strategy Comparison")
    st.markdown(
        "Running **3 prompt strategies** on the same scene data and comparing "
        "coherence scores, citation coverage, and hallucination risk."
    )

    with st.spinner("Running 3 API calls (takes ~30–60s)..."):
        try:
            analysis_result = run_attention_analysis(
                st.session_state.last_scene_data,
                st.session_state.last_character_map or {},
                st.session_state.last_emotional_arc or {},
                style=story_style
            )

            report_md = format_analysis_report(analysis_result)
            st.markdown(report_md)

            summary = analysis_result.get("summary", {})
            table   = summary.get("comparison_table", [])

            if table:
                st.subheader("📈 Visual Comparison")
                import pandas as pd
                df = pd.DataFrame(table)
                df.columns = ["Strategy", "Coherence", "Citations %",
                              "Hallucination Risk", "Latency ms", "Cost $"]
                st.dataframe(df, use_container_width=True)

                st.bar_chart(
                    data=df.set_index("Strategy")["Coherence"],
                    use_container_width=True
                )

            if analysis_result.get("best_story"):
                with st.expander("📖 Best Story (from winning strategy)"):
                    st.markdown(analysis_result["best_story"])

        except Exception as e:
            st.error(f"❌ Ablation study failed: {e}")
            if show_debug:
                st.exception(e)
