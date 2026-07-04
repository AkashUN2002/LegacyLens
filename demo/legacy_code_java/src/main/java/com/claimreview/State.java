package com.claimreview;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * State and schema definitions for the Multi-Modal Evidence Review pipeline.
 *
 * Contents:
 *   - Controlled-vocabulary enums (allowed output values from the problem spec)
 *   - object_part enums, one per object type, with a coercion helper
 *   - POJO models for the LLM-node outputs (extract_claim, analyze_images, reconcile)
 *   - The flat LangGraph ClaimState map
 */
public final class State {

    private State() {}

    // ---------------------------------------------------------------------------
    // Controlled vocabularies (allowed values, straight from the spec)
    // each member carries its string value, so it serializes to CSV directly
    // and drops cleanly into structured-output parsing
    // ---------------------------------------------------------------------------

    public enum ClaimObject {
        car("car"),
        laptop("laptop"),
        package_("package");   // 'package' is a Java keyword; value stays "package"

        private final String value;
        ClaimObject(String value) { this.value = value; }
        @JsonValue public String getValue() { return value; }
        @JsonCreator public static ClaimObject from(String v) {
            for (ClaimObject m : values()) if (m.value.equals(v)) return m;
            throw new IllegalArgumentException(v);
        }
        @Override public String toString() { return value; }
    }

    public enum ClaimStatus {
        supported("supported"),
        contradicted("contradicted"),
        not_enough_information("not_enough_information");

        private final String value;
        ClaimStatus(String value) { this.value = value; }
        @JsonValue public String getValue() { return value; }
        @JsonCreator public static ClaimStatus from(String v) {
            for (ClaimStatus m : values()) if (m.value.equals(v)) return m;
            throw new IllegalArgumentException(v);
        }
        @Override public String toString() { return value; }
    }

    public enum IssueType {
        dent("dent"),
        scratch("scratch"),
        crack("crack"),
        glass_shatter("glass_shatter"),
        broken_part("broken_part"),
        missing_part("missing_part"),
        torn_packaging("torn_packaging"),
        crushed_packaging("crushed_packaging"),
        water_damage("water_damage"),
        stain("stain"),
        none("none"),
        unknown("unknown");

        private final String value;
        IssueType(String value) { this.value = value; }
        @JsonValue public String getValue() { return value; }
        @JsonCreator public static IssueType from(String v) {
            for (IssueType m : values()) if (m.value.equals(v)) return m;
            throw new IllegalArgumentException(v);
        }
        @Override public String toString() { return value; }
    }

    public enum Severity {
        none("none"),
        low("low"),
        medium("medium"),
        high("high"),
        unknown("unknown");

        private final String value;
        Severity(String value) { this.value = value; }
        @JsonValue public String getValue() { return value; }
        @JsonCreator public static Severity from(String v) {
            for (Severity m : values()) if (m.value.equals(v)) return m;
            throw new IllegalArgumentException(v);
        }
        @Override public String toString() { return value; }
    }

    public enum RiskFlag {
        none("none"),
        blurry_image("blurry_image"),
        cropped_or_obstructed("cropped_or_obstructed"),
        low_light_or_glare("low_light_or_glare"),
        wrong_angle("wrong_angle"),
        wrong_object("wrong_object"),
        wrong_object_part("wrong_object_part"),
        damage_not_visible("damage_not_visible"),
        claim_mismatch("claim_mismatch"),
        possible_manipulation("possible_manipulation"),
        non_original_image("non_original_image"),
        text_instruction_present("text_instruction_present"),
        user_history_risk("user_history_risk"),
        manual_review_required("manual_review_required");

        private final String value;
        RiskFlag(String value) { this.value = value; }
        @JsonValue public String getValue() { return value; }
        @JsonCreator public static RiskFlag from(String v) {
            for (RiskFlag m : values()) if (m.value.equals(v)) return m;
            throw new IllegalArgumentException(v);
        }
        @Override public String toString() { return value; }
    }

    // ---------------------------------------------------------------------------
    // object_part: one enum per object type (vocabularies differ by object)
    // ---------------------------------------------------------------------------

    public enum CarPart {
        front_bumper("front_bumper"), rear_bumper("rear_bumper"), door("door"),
        hood("hood"), windshield("windshield"), side_mirror("side_mirror"),
        headlight("headlight"), taillight("taillight"), fender("fender"),
        quarter_panel("quarter_panel"), body("body"), unknown("unknown");
        private final String value;
        CarPart(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    public enum LaptopPart {
        screen("screen"), keyboard("keyboard"), trackpad("trackpad"),
        hinge("hinge"), lid("lid"), corner("corner"), port("port"),
        base("base"), body("body"), unknown("unknown");
        private final String value;
        LaptopPart(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    public enum PackagePart {
        box("box"), package_corner("package_corner"), package_side("package_side"),
        seal("seal"), label("label"), contents("contents"), item("item"),
        unknown("unknown");
        private final String value;
        PackagePart(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    // Map an object to its part enum, plus a coercion helper for validation
    public static final Map<ClaimObject, String[]> PART_ENUM = new LinkedHashMap<>();
    static {
        PART_ENUM.put(ClaimObject.car, values(CarPart.class));
        PART_ENUM.put(ClaimObject.laptop, values(LaptopPart.class));
        PART_ENUM.put(ClaimObject.package_, values(PackagePart.class));
    }

    private static String[] values(Class<? extends Enum<?>> cls) {
        Enum<?>[] members = cls.getEnumConstants();
        String[] out = new String[members.length];
        for (int i = 0; i < members.length; i++) out[i] = members[i].name();
        return out;
    }

    /**
     * Snap a raw part string to a valid value for this object.
     *
     * Tolerant matching, because the model returns natural phrasing ('front bumper',
     * 'package corner', 'laptop corner') that rarely equals the enum token exactly:
     *   1. exact match after normalizing separators ('front bumper' -> 'front_bumper')
     *   2. a member whose token set is contained in the value's tokens, preferring the
     *      most specific (longest) match ('package corner crushed' -> 'package_corner';
     *      'corner' for a laptop -> 'corner')
     * Falls back to 'unknown' only when nothing plausibly matches.
     */
    public static String coerce_part(ClaimObject claim_object, String value) {
        String[] enum_cls = PART_ENUM.get(claim_object);
        if (enum_cls == null) {
            return "unknown";
        }

        String v = norm(value);
        if (v.isEmpty()) {
            return "unknown";
        }

        // 1) exact normalized match
        for (String m : enum_cls) {
            if (m.equals(v)) return m;
        }

        // 2) token-subset match, most specific (longest enum value) wins
        Set<String> v_tokens = new HashSet<>(List.of(v.split("_")));
        String best = null;
        int best_len = -1;
        for (String m : enum_cls) {
            if (m.equals("unknown")) continue;
            Set<String> m_tokens = new HashSet<>(List.of(m.split("_")));
            if (v_tokens.containsAll(m_tokens) && m.length() > best_len) {
                best = m;
                best_len = m.length();
            }
        }
        return best != null ? best : "unknown";
    }

    private static String norm(String s) {
        return (s == null ? "" : s).trim().toLowerCase().replace("-", "_").replace(" ", "_");
    }

    // ---------------------------------------------------------------------------
    // LLM-node structured outputs (the contract handed to the model)
    // object_part is kept loose (str) here; validate via coerce_part at finalize
    // ---------------------------------------------------------------------------

    /** One per submitted image, produced by analyze_images. */
    public static final class ImageFinding {
        public String image_id;                 // filename without extension, e.g. "img_1"
        public String detected_object;          // car | laptop | package | other | unclear
        public String detected_part;
        public IssueType detected_issue;
        public Severity severity = Severity.unknown;
        public List<RiskFlag> flags = new ArrayList<>();
        public boolean text_instruction_present = false;
        public boolean usable;                  // is THIS single image usable for review?
    }

    /** Wrapper returned by analyze_images (model returns one object, not a bare list). */
    public static final class ImageAnalysis {
        public List<ImageFinding> findings = new ArrayList<>();
    }

    /** The reconciled decision, produced by reconcile. */
    public static final class Verdict {
        public ClaimStatus claim_status;
        public IssueType issue_type;
        public String object_part;              // coerced against PART_ENUM at finalize
        public Severity severity;
        public List<String> supporting_image_ids = new ArrayList<>();
        public List<RiskFlag> risk_flags = new ArrayList<>();
        public String justification;
    }

    public static final class ClaimedItem {
        public String object_part;
        public String issue_type;
        public String claimed_severity;         // severity ONLY if customer states it; else null
    }

    public static final class ExtractedClaim {
        public List<ClaimedItem> items = new ArrayList<>();
        public String summary;
        public String source_language;
        public boolean manipulation_attempt;
    }

    // ---------------------------------------------------------------------------
    // Graph state: flat, single-writer fields. Modeled as a string-keyed map so
    // nodes can return partial updates, mirroring the Python TypedDict (total=False).
    // ---------------------------------------------------------------------------
    public static final class ClaimState extends LinkedHashMap<String, Object> {
    }
}
