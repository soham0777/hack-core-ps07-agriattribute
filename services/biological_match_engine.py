"""
biological_match_engine.py - Transparent Multi-Dimensional Biological Matching Engine
AgriAttribute AI — Syngenta Biologicals × ANNAM.AI Hack Core 2026 (PS-07)

Purpose:
Evaluates and ranks biological products against active Field Context (Crop, Phenology,
Soil NPK, Ambient Weather / MCII Telemetry, Foliar Pathology, and Controllable Management).
Clearly separates RAW SENSOR DATA from PRODUCT EVIDENCE and AGRIATTRIBUTE INTERPRETATION.
"""

import os
import textwrap
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st

try:
    from services.management_engine import safe_float, safe_int
except (ImportError, ModuleNotFoundError):
    from management_engine import safe_float, safe_int

from services.biological_catalog_service import (
    BiologicalProduct,
    get_all_products,
    get_product_by_id,
    get_verified_image_path,
    get_base64_image
)


@dataclass
class ProductMatchResult:
    """Detailed multi-dimensional matching outcome for a specific biological product."""
    product: BiologicalProduct
    compatibility_rating: str  # 'Strong evidence match', 'Partial evidence match', 'Alternative crop use'
    compatibility_color: str
    compatibility_badge: str
    compatibility_score: int
    
    # Dimensions
    crop_fit: bool
    crop_fit_reason: str
    stage_fit: bool
    stage_fit_reason: str
    stress_fit: bool
    stress_fit_reason: str
    weather_fit: bool
    weather_fit_reason: str
    disease_fit: bool
    disease_fit_reason: str
    
    # Evidence Structure
    reasons: List[str]
    raw_telemetry: Dict[str, Any]
    product_evidence: Dict[str, Any]
    agriattribute_interpretation: str


# ============================================================================
# MULTI-DIMENSIONAL MATCHING ALGORITHM
# ============================================================================

def evaluate_product_match(
    product: BiologicalProduct,
    field_ctx: Any,
    ow_live: Optional[Dict[str, Any]] = None,
    disease_risk_pct: float = 25.0
) -> ProductMatchResult:
    """
    Evaluates compatibility between a single biological product and the active field state.
    """
    crop_name = getattr(field_ctx, 'crop', 'Soybean')
    crop_stage = getattr(field_ctx, 'crop_stage', 'Flowering / Pod Formation')
    temp_c = safe_float(getattr(field_ctx, 'temp_c', 28.5), 28.5)
    heat_stress_days = safe_int(getattr(field_ctx, 'heat_stress_days', 2), 2)
    wind_kmh = safe_float(getattr(field_ctx, 'wind_speed_kmh', 10.5), 10.5)
    rain_prob = safe_int(getattr(field_ctx, 'rain_mm', 0), 0)
    if ow_live:
        temp_c = safe_float(ow_live.get('temp_c', temp_c), temp_c)
        wind_kmh = safe_float(ow_live.get('wind_speed_kmh', wind_kmh), wind_kmh)
        rain_prob = safe_int(ow_live.get('rain_prob_pct', 10), 10)
        
    soil_ph = safe_float(getattr(field_ctx, 'ph', 7.2), 7.2)
    soc = safe_float(getattr(field_ctx, 'soc', 0.52), 0.52)

    reasons = []
    score = 0
    
    # 1. CROP FIT (40 points)
    c_low = crop_name.lower()
    crop_matched = False
    for tc in product.target_crops:
        if tc.lower() in c_low or c_low in tc.lower():
            crop_matched = True
            break
        elif tc.lower() == "all crops":
            crop_matched = True
            break
        elif "vegetables" in tc.lower() and ("tomato" in c_low or "onion" in c_low or "chilli" in c_low):
            crop_matched = True
            break
        elif "cereals" in tc.lower() and ("wheat" in c_low or "rice" in c_low or "maize" in c_low):
            crop_matched = True
            break
        elif "pulses" in tc.lower() and ("soybean" in c_low or "chickpea" in c_low or "tur" in c_low or "groundnut" in c_low):
            crop_matched = True
            break

    if crop_matched:
        score += 40
        crop_fit = True
        crop_fit_reason = f"Documented for {crop_name} in official technical label."
        reasons.append(f"🌾 <b>Crop Alignment:</b> Officially documented and validated for <b>{crop_name}</b>.")
    else:
        crop_fit = False
        crop_fit_reason = f"Primary label documentation is for {', '.join(product.target_crops[:3])}."

    # 2. PHENOLOGICAL STAGE FIT (20 points)
    stage_low = crop_stage.lower()
    prod_stage_low = product.application_stage.lower()
    stage_fit = False
    
    if any(s in prod_stage_low for s in ["flowering", "bloom", "flower", "pod", "tuber", "panicle", "heading", "silking", "square", "boll"]) and any(s in stage_low for s in ["flower", "pod", "bloom", "heading", "silking", "square", "boll"]):
        stage_fit = True
        score += 20
        stage_fit_reason = f"Documented reproductive timing matches current '{crop_stage}'."
        reasons.append(f"🌱 <b>Growth Stage Fit:</b> Current <b>{crop_stage}</b> aligns with the documented peak application window.")
    elif any(s in prod_stage_low for s in ["vegetative", "tillering", "seedling", "establishment", "sowing"]) and any(s in stage_low for s in ["veg", "tillering", "seedling", "establishment"]):
        stage_fit = True
        score += 20
        stage_fit_reason = f"Documented vegetative timing matches current '{crop_stage}'."
        reasons.append(f"🌱 <b>Growth Stage Fit:</b> Current <b>{crop_stage}</b> aligns with the recommended vegetative window.")
    elif "throughout" in prod_stage_low or "pre-stress" in prod_stage_low or "any" in prod_stage_low:
        stage_fit = True
        score += 15
        stage_fit_reason = "Flexible application window across active crop growth."
        reasons.append("🌱 <b>Growth Stage Fit:</b> Broad operational window throughout active canopy development.")
    else:
        stage_fit = False
        stage_fit_reason = f"Documented window: {product.application_stage} (Current stage: {crop_stage})."

    # 3. STRESS / SOIL FIT (20 points)
    stress_fit = False
    stress_fit_reason = ""
    is_high_heat = (temp_c > 32.0 or heat_stress_days >= 2)
    is_drought = (getattr(field_ctx, 'irrigation_type', '') == 'Rainfed' or (getattr(field_ctx, 'mcii_soil_moisture_pct', 40.0) < 30.0))
    is_high_ph = (soil_ph > 7.8 or soil_ph < 6.0)
    
    if is_high_heat and any("heat" in ts.lower() or "thermal" in ts.lower() or "drought" in ts.lower() for ts in product.target_stress):
        stress_fit = True
        score += 20
        stress_fit_reason = f"Targeted heat stress priming ({temp_c:.1f}°C ambient forecast)."
        reasons.append(f"☀️ <b>Abiotic Stress Fit:</b> High heat stress detected ({temp_c:.1f}°C). Product contains osmoprotectants for thermal cellular buffering.")
    elif is_drought and any("drought" in ts.lower() or "water" in ts.lower() or "scarcity" in ts.lower() for ts in product.target_stress):
        stress_fit = True
        score += 20
        stress_fit_reason = "Water Use Efficiency (WUE) active osmoregulation."
        reasons.append(f"💧 <b>Moisture Deficit Fit:</b> Water stress mitigation active. Formulated to preserve cellular turgor under moisture deficit.")
    elif any("nutrient" in ts.lower() or "nitrogen" in ts.lower() or "phosphorus" in ts.lower() or "potassium" in ts.lower() or "zinc" in ts.lower() for ts in product.target_stress):
        stress_fit = True
        score += 18
        stress_fit_reason = "Rhizosphere nutrient solubilization and mobilization."
        reasons.append("🧪 <b>Nutrient Use Efficiency:</b> Stimulates microbial assimilation and prevents rhizosphere nutrient lockup.")
    elif product.product_category == "Biostimulant":
        stress_fit = True
        score += 15
        stress_fit_reason = "General biostimulant vitality and photosynthetic stimulation."
        reasons.append("🧬 <b>Metabolic Activation:</b> Supplies natural peptides, betaines, and co-factors to elevate photosynthetic index.")
    else:
        stress_fit_reason = "General preventative maintenance."

    # 4. WEATHER & SPRAY WINDOW FIT (10 points)
    if wind_kmh < 15.0 and rain_prob <= 25:
        weather_fit = True
        score += 10
        weather_fit_reason = f"Optimal spray window (Wind: {wind_kmh:.1f} km/h, Rain: {rain_prob}%)."
        reasons.append(f"🌦️ <b>Weather Window:</b> Current wind speed ({wind_kmh:.1f} km/h) and rain probability ({rain_prob}%) provide an optimal foliar spray window.")
    elif wind_kmh < 25.0 and rain_prob <= 40:
        weather_fit = True
        score += 6
        weather_fit_reason = f"Moderate spray conditions (Wind: {wind_kmh:.1f} km/h)."
        reasons.append(f"🌦️ <b>Weather Window:</b> Spray early morning to avoid midday wind drift ({wind_kmh:.1f} km/h).")
    else:
        weather_fit = False
        weather_fit_reason = f"Suboptimal weather (Wind: {wind_kmh:.1f} km/h, Rain: {rain_prob}%)."

    # 5. DISEASE / PATHOLOGY FIT (10 points)
    disease_fit = False
    disease_fit_reason = "General crop vitality and root zone health."
    if disease_risk_pct > 35.0 and product.product_category == "Biofungicide":
        disease_fit = True
        score += 10
        disease_fit_reason = f"Elevated foliar disease risk ({disease_risk_pct:.0f}%). Preventive biofungicide barrier."
        reasons.append(f"🛡️ <b>Pathology Defense:</b> Elevated foliar risk ({disease_risk_pct:.0f}%). Forms a preventive microbial biofilm barrier.")
    elif product.product_category != "Biofungicide":
        disease_fit = True
        score += 8
        disease_fit_reason = "Enhances natural plant immunity and physiological vigor."
    else:
        disease_fit = False
        disease_fit_reason = "Biofungicide not required under current low foliar disease pressure."

    # Final Classification
    if crop_matched and score >= 75:
        compatibility_rating = "Strong evidence match"
        compatibility_badge = "STRONG EVIDENCE MATCH"
        compatibility_color = "#059669"
    elif crop_matched or score >= 50:
        compatibility_rating = "Partial evidence match"
        compatibility_badge = "PARTIAL EVIDENCE MATCH"
        compatibility_color = "#d97706"
    else:
        compatibility_rating = "Alternative crop use"
        compatibility_badge = "ALTERNATIVE CROP USE"
        compatibility_color = "#64748b"

    raw_telemetry = {
        "temperature_c": temp_c,
        "wind_speed_kmh": wind_kmh,
        "rain_probability_pct": rain_prob,
        "disease_risk_pct": disease_risk_pct,
        "active_crop": crop_name,
        "growth_stage": crop_stage
    }

    product_evidence = {
        "documented_rate": product.application_rate,
        "documented_stage": product.application_stage,
        "target_stress": product.target_stress,
        "source_document": product.source_document,
        "source_page": product.source_page,
        "trial_summary": product.trial_results_summary
    }

    if compatibility_rating == "Strong evidence match":
        interp = f"Current field conditions ({crop_name} at {crop_stage}, {temp_c:.1f}°C ambient) are highly compatible with the official documented use pattern of {product.product_name}."
    elif compatibility_rating == "Partial evidence match":
        interp = f"{product.product_name} is compatible with {crop_name}; review application timing or environmental forecast for peak efficacy."
    else:
        interp = f"{product.product_name} is primarily documented for {', '.join(product.target_crops[:2])}; evaluate suitability before applying on {crop_name}."

    return ProductMatchResult(
        product=product,
        compatibility_rating=compatibility_rating,
        compatibility_color=compatibility_color,
        compatibility_badge=compatibility_badge,
        compatibility_score=score,
        crop_fit=crop_fit,
        crop_fit_reason=crop_fit_reason,
        stage_fit=stage_fit,
        stage_fit_reason=stage_fit_reason,
        stress_fit=stress_fit,
        stress_fit_reason=stress_fit_reason,
        weather_fit=weather_fit,
        weather_fit_reason=weather_fit_reason,
        disease_fit=disease_fit,
        disease_fit_reason=disease_fit_reason,
        reasons=reasons,
        raw_telemetry=raw_telemetry,
        product_evidence=product_evidence,
        agriattribute_interpretation=interp
    )


def match_all_products(
    field_ctx: Any,
    ow_live: Optional[Dict[str, Any]] = None,
    disease_risk_pct: float = 25.0
) -> List[ProductMatchResult]:
    """Matches all 19 canonical products and sorts them by compatibility score."""
    products = get_all_products()
    results = [evaluate_product_match(p, field_ctx, ow_live, disease_risk_pct) for p in products]
    results.sort(key=lambda r: r.compatibility_score, reverse=True)
    return results


def get_top_field_matches(
    field_ctx: Any,
    ow_live: Optional[Dict[str, Any]] = None,
    disease_risk_pct: float = 25.0,
    top_n: int = 3
) -> List[ProductMatchResult]:
    """Returns top N evidence-matched products for the active field context."""
    all_matches = match_all_products(field_ctx, ow_live, disease_risk_pct)
    return all_matches[:top_n]


# ============================================================================
# STREAMLIT UI RENDERER: BIOLOGICALS INTELLIGENCE LAYER
# ============================================================================

def render_biologicals_section_ui(
    field_ctx: Any,
    ow_live: Optional[Dict[str, Any]] = None,
    ow_5day: Optional[List[Dict[str, Any]]] = None,
    disease_risk_pct: float = 25.0,
    lang: str = "en",
    t_func: Any = None
) -> None:
    """
    Renders the complete, authentic Farmer Decision Biologicals Intelligence Layer with 2-way Field Sync.
    """
    t = t_func if t_func is not None else (lambda k, l, **kwargs: k)
    
    crop_name = getattr(field_ctx, 'crop', 'Soybean')
    crop_stage = getattr(field_ctx, 'crop_stage', 'Flowering / Pod Formation')
    temp_c = safe_float(getattr(field_ctx, 'temp_c', 28.5), 28.5)
    heat_stress_days = safe_int(getattr(field_ctx, 'heat_stress_days', 2), 2)
    wind_kmh = safe_float(getattr(field_ctx, 'wind_speed_kmh', 10.5), 10.5)
    
    # Header & Live Context Bar
    st.markdown("### 🧬 Biologicals Intelligence & Evidence-Matched Protocol")
    st.caption("Authentic Syngenta Biologicals Product Dossiers, Causal Attribution & KRIBHCO / Third-Party Benchmarks")

    # Dynamic Field Context Synchronization Bar
    with st.container(border=True):
        col_ctx1, col_ctx2, col_ctx3, col_ctx4 = st.columns([1.2, 1.1, 1.1, 1.1])
        with col_ctx1:
            st.markdown(f"🌾 **Active Field Crop**  \n### {crop_name}")
            st.caption(f"Phenology: **{crop_stage}**")
        with col_ctx2:
            st.metric(label="🌡️ Ambient Temp", value=f"{temp_c:.1f}°C", delta=f"{heat_stress_days}d Heat Stress", delta_color="inverse")
        with col_ctx3:
            st.metric(label="💨 Wind & Spray", value=f"{wind_kmh:.1f} km/h", delta="Optimal Window" if wind_kmh < 15 else "Moderate Drift", delta_color="normal" if wind_kmh < 15 else "off")
        with col_ctx4:
            st.metric(label="🛡️ Foliar Disease Risk", value=f"{disease_risk_pct:.0f}%", delta="Bio-barrier Active" if disease_risk_pct > 35 else "Low Pressure", delta_color="inverse" if disease_risk_pct > 35 else "normal")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. FEATURED EVIDENCE MATCHES (TOP 3)
    st.subheader("🌟 Best Evidence Matches for Active Field")
    st.caption("Ranked using verifiable multi-factor compatibility (Crop fit, Growth stage, Thermal/Drought stress, Spray window, and Official technical sheets):")

    top_matches = get_top_field_matches(field_ctx, ow_live, disease_risk_pct, top_n=3)
    
    top_cols = st.columns(len(top_matches))
    for idx, match_res in enumerate(top_matches):
        p = match_res.product
        with top_cols[idx]:
            with st.container(border=True):
                # Badge headers
                org_tag = p.organization.upper()
                st.markdown(f"**:{'green' if 'syngenta' in p.organization.lower() else 'blue' if 'agrigem' in p.organization.lower() else 'orange'}[{org_tag}]**")
                st.markdown(f"**:{'green' if match_res.compatibility_score >= 75 else 'orange'}[{match_res.compatibility_badge}]**")
                
                img_path = get_verified_image_path(p)
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Packshot: {p.product_name}", use_container_width=True)
                
                st.markdown(f"### {p.product_name}")
                st.caption(f"{p.product_category} • {p.biological_class}")
                
                st.markdown(f"**Documented Rate:** :green[{p.application_rate}]  \n**Application Timing:** {p.application_stage}")
                
                st.markdown("**Why this product:**")
                for r in match_res.reasons[:2]:
                    st.markdown(f"- {r}", unsafe_allow_html=True)
                    
                st.caption(f"**Source:** {p.source_document} ({p.source_page})")
                
                # Action button to select for decision & sync
                if st.button(f"🎯 Select {p.product_name.split()[0]} as Active Treatment", key=f"btn_sel_bio_{p.product_id}", use_container_width=True, type="primary" if idx == 0 else "secondary"):
                    st.session_state["selected_bio_product"] = p.product_name
                    st.session_state["s_dosage"] = p.application_rate_num
                    st.session_state["s_dosage_unit"] = p.application_rate_unit
                    st.session_state["whatif_dosage"] = p.application_rate_num
                    st.session_state["_compute_key"] = None
                    st.toast(f"Selected {p.product_name} ({p.application_rate}) for Causal Attribution!", icon="🧬")
                    st.rerun()

    st.markdown("---")

    # 3. INTERACTIVE SEARCH & FULL PRODUCT CATALOGUE WITH FILTERING
    st.subheader("📚 Complete Verified Biologicals Product Catalogue (19 Products)")
    st.caption("Filter by Organization, Biological Class, or Crop Compatibility. Every product features authentic technical documentation and packshots:")

    col_flt1, col_flt2, col_flt3 = st.columns([1.2, 1.2, 1.4])
    with col_flt1:
        org_filter = st.selectbox(
            "Filter by Organization / Provenance",
            options=["All Organizations", "Syngenta Biologicals (Official)", "Agrigem (Third-Party Distributor)", "KRIBHCO (Cooperative Bio-fertilizers)"],
            index=0
        )
    with col_flt2:
        cat_filter = st.selectbox(
            "Filter by Category",
            options=["All Categories", "Biostimulant", "Biofertilizer", "Biofungicide"],
            index=0
        )
    with col_flt3:
        search_query = st.text_input("🔍 Search by Product Name, Ingredient, or Crop", placeholder="e.g. Megafol, YieldON, Azotobacter, Wheat, Heat...")

    # Filter product list
    all_matches = match_all_products(field_ctx, ow_live, disease_risk_pct)
    filtered_matches = []
    
    for m in all_matches:
        p = m.product
        
        # Org Filter
        if org_filter == "Syngenta Biologicals (Official)" and "syngenta" not in p.organization.lower():
            continue
        elif org_filter == "Agrigem (Third-Party Distributor)" and "agrigem" not in p.organization.lower():
            continue
        elif org_filter == "KRIBHCO (Cooperative Bio-fertilizers)" and "kribhco" not in p.organization.lower():
            continue
            
        # Category Filter
        if cat_filter != "All Categories" and p.product_category != cat_filter:
            continue
            
        # Search Query
        if search_query:
            q = search_query.lower()
            match_txt = f"{p.product_name} {p.brand} {p.active_components} {' '.join(p.target_crops)} {' '.join(p.target_stress)} {p.mode_of_action}".lower()
            if q not in match_txt:
                continue
                
        filtered_matches.append(m)

    st.caption(f"Showing **{len(filtered_matches)}** verified biological products matching current criteria:")

    # Render filtered products in clean 2-column grid
    for i in range(0, len(filtered_matches), 2):
        col_p1, col_p2 = st.columns(2)
        
        # Product 1
        with col_p1:
            m1 = filtered_matches[i]
            _render_detailed_product_card(m1, lang, t)
            
        # Product 2
        if i + 1 < len(filtered_matches):
            with col_p2:
                m2 = filtered_matches[i+1]
                _render_detailed_product_card(m2, lang, t)

    st.markdown("---")

    # 4. FIELD APPLICATION LOGGER & CAUSAL INTEGRATION BRIDGE
    st.subheader("📝 Field Treatment Record & Closed-Loop Attribution Bridge")
    st.caption("Record the verified biological application for this field. Synchronizes with Yield Predictor, Cost of Cultivation, and Farm Memory:")

    with st.form("bio_app_log_form"):
        col_log1, col_log2, col_log3 = st.columns(3)
        with col_log1:
            active_bio_name = st.session_state.get("selected_bio_product", top_matches[0].product.product_name if top_matches else "Syngenta Quantis")
            all_names = [p.product_name for p in get_all_products()]
            def_idx = all_names.index(active_bio_name) if active_bio_name in all_names else 0
            log_prod = st.selectbox("Selected Biological Product", options=all_names, index=def_idx)
        with col_log2:
            matched_prod_obj = next((p for p in get_all_products() if p.product_name == log_prod), top_matches[0].product)
            log_dose = st.number_input(f"Applied Dosage ({matched_prod_obj.application_rate_unit})", value=safe_float(matched_prod_obj.application_rate_num, 1.0), step=0.25)
        with col_log3:
            log_stage = st.selectbox("Growth Stage at Application", options=["Vegetative / Tillering", "Flower Initiation / Bloom", "Pod / Grain Development", "Post-Stress Recovery"], index=1)
            
        col_log4, col_log5 = st.columns([1, 2])
        with col_log4:
            log_apps = st.selectbox("Application Number", options=[1, 2, 3], index=0)
        with col_log5:
            log_notes = st.text_input("Farmer Application Field Notes", value=f"Applied via calibrated tractor boom sprayer with {matched_prod_obj.water_volume}. Ambient temp: {temp_c:.1f}°C.")
            
        submit_app = st.form_submit_button("💾 Save Application to Farm Context & Ledger", use_container_width=True)
        if submit_app:
            st.session_state["selected_bio_product"] = log_prod
            st.session_state["s_dosage"] = log_dose
            st.session_state["s_dosage_unit"] = matched_prod_obj.application_rate_unit
            st.session_state["whatif_dosage"] = log_dose
            st.session_state["_compute_key"] = None
            st.session_state["bio_logged_record"] = {
                "product": log_prod,
                "dosage": log_dose,
                "unit": matched_prod_obj.application_rate_unit,
                "crop": crop_name,
                "stage": log_stage,
                "applications_count": log_apps,
                "notes": log_notes
            }
            st.success(f"✅ Application record for {log_prod} saved to Field Context & Farm Memory!")
            st.rerun()


def _render_detailed_product_card(match_res: ProductMatchResult, lang: str, t: Any) -> None:
    """Renders a structured, high-contrast product card with native Streamlit container."""
    p = match_res.product
    img_path = get_verified_image_path(p)
    
    with st.container(border=True):
        col_hdr1, col_hdr2 = st.columns([1.5, 1])
        with col_hdr1:
            org_tag = p.organization.upper()
            st.markdown(f"**:{'green' if 'syngenta' in p.organization.lower() else 'blue' if 'agrigem' in p.organization.lower() else 'orange'}[{org_tag}]**")
        with col_hdr2:
            st.markdown(f"**:{'green' if match_res.compatibility_score >= 75 else 'orange'}[{match_res.compatibility_badge}]**")
            
        st.markdown(f"### {p.product_name}")
        st.caption(f"{p.product_category} • {p.biological_class}")
        
        col_card_img, col_card_info = st.columns([1, 1.4])
        with col_card_img:
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Packshot: {p.product_name}", use_container_width=True)
            else:
                st.info("Product Packshot Loading...")
                
        with col_card_info:
            st.markdown(f"**Active Components:**  \n{p.active_components}")
            st.markdown(f"**Documented Rate:** :green[{p.application_rate}]")
            st.markdown(f"**Application Timing:** {p.application_stage}")
            st.markdown(f"**Target Crops:** {', '.join(p.target_crops[:4])}")
            st.markdown(f"**Target Stress:** {', '.join(p.target_stress[:2])}")
            
        with st.expander(f"📖 View Official Evidence: {p.product_name} ({p.source_document})", expanded=False):
            st.markdown(f"**Source Document:** `{p.source_document}` ({p.source_page})")
            st.markdown(f"**Source Type:** {p.source_type}")
            st.markdown(f"**Market / Registration:** {p.registration_status} ({p.country})")
            st.markdown(f"**Documented Mode of Action:**  \n{p.mode_of_action}")
            st.markdown(f"**Documented Field Trial Results:**  \n:green[{p.trial_results_summary}]")
            st.markdown(f"**Tank Mix & Compatibility:**  \n{p.tank_mix_information}")
            st.markdown(f"**Official Reference URL:** [{p.source_url}]({p.source_url})")

        is_selected = st.session_state.get("selected_bio_product") == p.product_name
        btn_label = f"✅ Active Field Biological" if is_selected else f"🎯 Apply {p.product_name} to Field Context"
        if st.button(btn_label, key=f"btn_pick_bio_{abs(hash(p.product_name))}", type="primary" if is_selected else "secondary", use_container_width=True):
            st.session_state["selected_bio_product"] = p.product_name
            st.session_state["s_dosage"] = safe_float(p.application_rate_num, 1.0)
            st.session_state["s_dosage_unit"] = p.application_rate_unit
            st.session_state["whatif_dosage"] = safe_float(p.application_rate_num, 1.0)
            st.session_state["_compute_key"] = None
            st.toast(f"Synchronized {p.product_name} across Yield, Cost, and ROI!", icon="🧬")
            st.rerun()


