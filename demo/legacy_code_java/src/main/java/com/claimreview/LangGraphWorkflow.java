package com.claimreview;

import com.claimreview.State.ClaimObject;
import com.claimreview.State.ClaimState;
import com.claimreview.State.ClaimStatus;
import com.claimreview.State.ClaimedItem;
import com.claimreview.State.ExtractedClaim;
import com.claimreview.State.ImageAnalysis;
import com.claimreview.State.ImageFinding;
import com.claimreview.State.IssueType;
import com.claimreview.State.RiskFlag;
import com.claimreview.State.Severity;
import com.claimreview.State.Verdict;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.bedrock.BedrockChatModel;
import dev.langchain4j.model.bedrock.BedrockChatRequestParameters;
import dev.langchain4j.model.chat.ChatModel;
import io.github.cdimascio.dotenv.Dotenv;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static com.claimreview.Utility.coerce_enum;
import static com.claimreview.Utility.enum_value;
import static com.claimreview.Utility.load_image_block;
import static com.claimreview.Utility.normalize;
import static com.claimreview.Utility.object_value;
import static com.claimreview.Utility.part_matches;
import static com.claimreview.Utility.pick_requirement;
import static com.claimreview.Utility.rebuild_findings;
import static com.claimreview.Utility.summarize_unusable;
import static com.claimreview.Utility.to_risk_flags;

/**
 * LangGraph workflow for the Multi-Modal Evidence Review pipeline.
 *
 * All eight nodes are implemented:
 *   load_context, extract_claim, analyze_images (+ usable_gate),
 *   evidence_check, reconcile, force_nei, risk_merge, finalize.
 * finalize writes `output_row` (the 14-column CSV row) as the deliverable.
 */
public class LangGraphWorkflow {

    static final Dotenv ENV = Dotenv.configure().ignoreIfMissing().load();

    // --- prompts ---
    static final Map<String, String> config = loadPrompts();
    static final String claim_extraction_prompt = config.get("claim_extraction_prompt");
    static final String image_analysis_prompt = config.get("image_analysis_prompt");
    static final String reconcile_prompt = config.get("reconcile_prompt");

    // --- where image_paths resolve from (image_paths look like images/test/case_x/img_1.jpg) ---
    static final String IMAGE_ROOT = getenv("IMAGE_ROOT", "dataset");

    // --- model + structured-output bindings ---
    // Longer read timeout + retries: multi-image vision calls can exceed the default.
    static final ChatModel llm = BedrockChatModel.builder()
            .modelId(getenv("ANTHROPIC_MODEL", null))
            .region(software.amazon.awssdk.regions.Region.US_EAST_1)
            .timeout(Duration.ofSeconds(120))
            .maxRetries(4)
            .defaultRequestParameters(BedrockChatRequestParameters.builder().temperature(0.0).build())   // deterministic
            .build();
    static final ChatModel vision_llm = BedrockChatModel.builder()
            .modelId(getenv("VISION_MODEL", getenv("ANTHROPIC_MODEL", null)))
            .region(software.amazon.awssdk.regions.Region.US_EAST_1)
            .timeout(Duration.ofSeconds(120))
            .maxRetries(4)
            .build();
    static final StructuredLlm<ImageAnalysis> analyzer = new StructuredLlm<>(vision_llm, ImageAnalysis.class);
    static final StructuredLlm<ExtractedClaim> extractor = new StructuredLlm<>(llm, ExtractedClaim.class);
    static final StructuredLlm<Verdict> reconciler = new StructuredLlm<>(llm, Verdict.class);

    static final StateGraph graph = new StateGraph();

    // ===========================================================================
    // Nodes
    // ===========================================================================

    /** Deterministic. Parse image_paths into (image_id, rel_path) pairs and normalize
     * claim_object to the enum. history + evidence_reqs arrive in the invoke payload. */
    static Map<String, Object> load_context(ClaimState state) {
        List<String[]> images = new ArrayList<>();
        for (String t : ((String) state.get("image_paths")).split(";")) {
            if (!t.strip().isEmpty()) {
                String stem = Path.of(t.strip()).getFileName().toString().replaceFirst("[.][^.]+$", "");
                images.add(new String[]{stem, t.strip()});
            }
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("claim_object", ClaimObject.from(((String) state.get("claim_object")).strip().toLowerCase()));
        out.put("images", images);
        return out;
    }

    /** Text LLM. Normalize the transcript into a structured claim. Flags manipulation. */
    static Map<String, Object> extract_claim(ClaimState state) {
        String transcript = (String) state.get("user_claim");
        ClaimObject claim_object = (ClaimObject) state.get("claim_object");
        String user_msg = "claim_object: " + claim_object + "\n\nTranscript:\n" + transcript;
        ExtractedClaim parsed = extractor.invoke(List.of(
                SystemMessage.from(claim_extraction_prompt + EXTRACT_SCHEMA),
                UserMessage.from(user_msg)
        ));
        return Map.of("claim", parsed);
    }

    /** Vision LLM. Per-image structured findings + per-image `usable`. Claim-blind. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> analyze_images(ClaimState state) {
        List<String[]> images = (List<String[]>) state.get("images");   // [(image_id, rel_path), ...]
        List<String> ordered_ids = new ArrayList<>();
        for (String[] p : images) ordered_ids.add(p[0]);

        List<Content> content = new ArrayList<>();
        content.add(TextContent.from("Analyze these " + images.size() + " image(s). "
                + "Return one finding per image using these ids in order: "
                + String.join(", ", ordered_ids) + "."));
        Set<String> load_errors = new LinkedHashSet<>();
        for (String[] p : images) {
            String iid = p[0], rel = p[1];
            content.add(TextContent.from("--- image_id: " + iid + " ---"));
            try {
                content.add(load_image_block(rel, IMAGE_ROOT));
            } catch (Exception e) {
                content.add(TextContent.from("(image could not be loaded)"));
                load_errors.add(iid);
            }
        }

        Map<String, ImageFinding> by_id = new LinkedHashMap<>();
        try {
            ImageAnalysis result = analyzer.invoke(List.of(
                    SystemMessage.from(image_analysis_prompt + ANALYZE_SCHEMA),
                    UserMessage.from(content)
            ));
            for (ImageFinding f : result.findings) by_id.put(f.image_id, f);
        } catch (Exception ignored) {
            by_id = new LinkedHashMap<>();   // degrade whole set to unusable
        }

        List<ImageFinding> findings = rebuild_findings(ordered_ids, by_id, load_errors);
        boolean valid = findings.stream().anyMatch(f -> f.usable);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("findings", findings);
        out.put("valid_image", valid);
        return out;
    }

    /** Deterministic coverage check: do usable images show the claimed object/part. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> evidence_check(ClaimState state) {
        List<ImageFinding> findings = (List<ImageFinding>) state.getOrDefault("findings", new ArrayList<>());
        List<ImageFinding> usable = new ArrayList<>();
        for (ImageFinding f : findings) if (f.usable) usable.add(f);
        ExtractedClaim claim = (ExtractedClaim) state.get("claim");
        ClaimObject claim_object = (ClaimObject) state.get("claim_object");
        List<Map<String, String>> evidence_reqs = (List<Map<String, String>>) state.getOrDefault("evidence_reqs", new ArrayList<>());

        String obj = object_value(claim_object);

        boolean object_ok = usable.stream().anyMatch(f -> normalize(f.detected_object).equals(obj));

        List<String> claimed_parts = new ArrayList<>();
        if (claim != null && claim.items != null) for (ClaimedItem it : claim.items) claimed_parts.add(it.object_part);
        boolean part_ok;
        if (!claimed_parts.isEmpty()) {
            part_ok = false;
            for (ImageFinding f : usable) for (String cp : claimed_parts) if (part_matches(cp, f.detected_part)) part_ok = true;
        } else {
            part_ok = object_ok;
        }

        boolean met = !usable.isEmpty() && object_ok && part_ok;

        Map<String, String> req = pick_requirement(claim_object, claim, evidence_reqs);
        String cite = req != null ? " (per " + req.get("requirement_id") + ")" : "";

        String reason;
        if (met) {
            reason = "At least one usable image clearly shows the claimed " + obj + " and relevant part, sufficient to inspect the claim" + cite + ".";
        } else if (usable.isEmpty()) {
            reason = "No usable image is available to inspect the claim.";
        } else if (!object_ok) {
            reason = "Usable images do not clearly show a " + obj + "; the claimed object cannot be inspected" + cite + ".";
        } else {
            List<String> nonEmpty = new ArrayList<>();
            for (String p : claimed_parts) if (p != null && !p.isEmpty()) nonEmpty.add(p);
            String parts = nonEmpty.isEmpty() ? "claimed part" : String.join(", ", nonEmpty);
            reason = "Usable images show the " + obj + " but not the " + parts + "; the claimed part cannot be inspected" + cite + ".";
        }

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("evidence_standard_met", met);
        out.put("evidence_standard_met_reason", reason);
        return out;
    }

    /** LLM reasoning over claim vs findings (no images re-sent; findings are truth). */
    @SuppressWarnings("unchecked")
    static Map<String, Object> reconcile(ClaimState state) {
        ExtractedClaim claim = (ExtractedClaim) state.get("claim");
        List<ImageFinding> findings = (List<ImageFinding>) state.getOrDefault("findings", new ArrayList<>());
        ClaimObject claim_object = (ClaimObject) state.get("claim_object");

        List<String> finding_lines = new ArrayList<>();
        for (ImageFinding f : findings) {
            List<String> fl = new ArrayList<>();
            for (RiskFlag x : f.flags) fl.add(enum_value(x));
            finding_lines.add("  image_id=" + f.image_id + ": object=" + f.detected_object + ", part=" + f.detected_part
                    + ", issue=" + enum_value(f.detected_issue) + ", severity=" + enum_value(f.severity)
                    + ", usable=" + f.usable + ", flags=" + fl + ", text_instruction=" + f.text_instruction_present);
        }
        List<String> item_lines = new ArrayList<>();
        if (claim != null && claim.items != null) for (ClaimedItem it : claim.items)
            item_lines.add("  part=" + it.object_part + ", issue=" + it.issue_type + ", stated_severity=" + it.claimed_severity);
        Object evid = state.get("evidence_standard_met");

        String msg = "claim_object: " + object_value(claim_object) + "\n"
                + "claim_summary: " + (claim != null ? claim.summary : "") + "\n"
                + "claimed_items:\n" + (item_lines.isEmpty() ? "  (none)" : String.join("\n", item_lines)) + "\n"
                + "evidence_standard_met: " + (evid != null ? evid : "not assessed") + "\n"
                + "valid_image: " + state.get("valid_image") + "\n"
                + "findings:\n" + (finding_lines.isEmpty() ? "  (none)" : String.join("\n", finding_lines));

        try {
            Verdict v = reconciler.invoke(List.of(
                    SystemMessage.from(reconcile_prompt + RECONCILE_SCHEMA),
                    UserMessage.from(msg)
            ));
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("claim_status", v.claim_status);
            out.put("issue_type", v.issue_type);
            out.put("object_part", v.object_part);
            out.put("severity", v.severity);
            out.put("supporting_image_ids", v.supporting_image_ids != null ? v.supporting_image_ids : new ArrayList<>());
            out.put("risk_flags", v.risk_flags != null ? v.risk_flags : new ArrayList<>());
            out.put("claim_status_justification", v.justification);
            return out;
        } catch (Exception e) {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("claim_status", ClaimStatus.not_enough_information);
            out.put("issue_type", IssueType.unknown);
            out.put("object_part", "unknown");
            out.put("severity", Severity.unknown);
            out.put("supporting_image_ids", new ArrayList<>());
            out.put("risk_flags", new ArrayList<>());
            out.put("claim_status_justification", "Could not adjudicate; defaulting to not_enough_information.");
            return out;
        }
    }

    /** Deterministic short-circuit for truly unusable image sets (no model call). */
    @SuppressWarnings("unchecked")
    static Map<String, Object> force_nei(ClaimState state) {
        List<ImageFinding> findings = (List<ImageFinding>) state.getOrDefault("findings", new ArrayList<>());
        String cause = summarize_unusable(findings);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("evidence_standard_met", false);
        out.put("evidence_standard_met_reason", "No usable image to inspect the claim (" + cause + ").");
        out.put("claim_status", ClaimStatus.not_enough_information);
        out.put("issue_type", IssueType.unknown);
        out.put("object_part", "unknown");
        out.put("severity", Severity.unknown);
        out.put("supporting_image_ids", new ArrayList<>());
        out.put("claim_status_justification", "Cannot confirm or refute the claim: no usable image evidence (" + cause + ").");
        return out;
    }

    /** Deterministic. Union all flags + apply review-escalation rules. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> risk_merge(ClaimState state) {
        List<ImageFinding> findings = (List<ImageFinding>) state.getOrDefault("findings", new ArrayList<>());
        ExtractedClaim claim = (ExtractedClaim) state.get("claim");
        Map<String, String> history = (Map<String, String>) state.getOrDefault("history", new LinkedHashMap<>());

        Set<RiskFlag> flags = new LinkedHashSet<>(to_risk_flags((List<?>) state.getOrDefault("risk_flags", new ArrayList<>())));

        for (ImageFinding f : findings) {
            flags.addAll(f.flags);
            if (f.text_instruction_present) flags.add(RiskFlag.text_instruction_present);
        }

        if (claim != null && claim.manipulation_attempt) flags.add(RiskFlag.text_instruction_present);

        for (String tok : (history.getOrDefault("history_flags", "")).split(";")) {
            tok = tok.strip();
            if (!tok.isEmpty() && !tok.equalsIgnoreCase("none")) flags.addAll(to_risk_flags(List.of(tok)));
        }

        if (flags.contains(RiskFlag.user_history_risk)) flags.add(RiskFlag.manual_review_required);
        if (flags.contains(RiskFlag.text_instruction_present)) flags.add(RiskFlag.manual_review_required);
        if (flags.contains(RiskFlag.possible_manipulation) || flags.contains(RiskFlag.non_original_image))
            flags.add(RiskFlag.manual_review_required);

        flags.remove(RiskFlag.none);
        List<RiskFlag> ordered = new ArrayList<>();
        for (RiskFlag f : RiskFlag.values()) if (flags.contains(f) && f != RiskFlag.none) ordered.add(f);
        return Map.of("risk_flags", ordered.isEmpty() ? List.of(RiskFlag.none) : ordered);
    }

    /** Deterministic. Coerce every field, join lists to ';' strings, assemble output_row. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> finalize(ClaimState state) {
        ClaimObject claim_object = (ClaimObject) state.get("claim_object");

        ClaimStatus status = coerce_enum(state.get("claim_status"), ClaimStatus.not_enough_information);
        IssueType issue = coerce_enum(state.get("issue_type"), IssueType.unknown);
        Severity sev = coerce_enum(state.get("severity"), Severity.unknown);
        String part = State.coerce_part(claim_object, enum_value(state.getOrDefault("object_part", "unknown")));

        List<Object> risk = (List<Object>) state.getOrDefault("risk_flags", List.of(RiskFlag.none));
        if (risk.isEmpty()) risk = List.of(RiskFlag.none);
        List<String> riskVals = new ArrayList<>();
        for (Object r : risk) riskVals.add(enum_value(r));
        String risk_str = riskVals.isEmpty() ? "none" : String.join(";", riskVals);

        List<String> sup = new ArrayList<>();
        for (Object s : (List<Object>) state.getOrDefault("supporting_image_ids", new ArrayList<>()))
            if (!String.valueOf(s).equalsIgnoreCase("none")) sup.add(String.valueOf(s));
        String sup_str = sup.isEmpty() ? "none" : String.join(";", sup);

        Object evid = state.get("evidence_standard_met");
        boolean evidBool = evid != null ? (boolean) evid : status != ClaimStatus.not_enough_information;

        Map<String, Object> row = new LinkedHashMap<>();
        row.put("user_id", state.getOrDefault("user_id", ""));
        row.put("image_paths", state.getOrDefault("image_paths", ""));
        row.put("user_claim", state.getOrDefault("user_claim", ""));
        row.put("claim_object", object_value(claim_object));
        row.put("evidence_standard_met", evidBool ? "true" : "false");
        row.put("evidence_standard_met_reason", state.getOrDefault("evidence_standard_met_reason", ""));
        row.put("risk_flags", risk_str);
        row.put("issue_type", enum_value(issue));
        row.put("object_part", part);
        row.put("claim_status", enum_value(status));
        row.put("claim_status_justification", state.getOrDefault("claim_status_justification", ""));
        row.put("supporting_image_ids", sup_str);
        row.put("valid_image", Boolean.TRUE.equals(state.get("valid_image")) ? "true" : "false");
        row.put("severity", enum_value(sev));
        return Map.of("output_row", row);
    }

    // ===========================================================================
    // Routing
    // ===========================================================================

    /** After analyze_images: route on what the images can support. */
    @SuppressWarnings("unchecked")
    static String usable_gate(ClaimState state) {
        if (Boolean.TRUE.equals(state.get("valid_image"))) return "evidence_check";
        List<ImageFinding> findings = (List<ImageFinding>) state.get("findings");
        for (ImageFinding f : findings)
            if (f.detected_issue != IssueType.unknown && !f.detected_object.equals("unclear")) return "reconcile";
        return "force_nei";
    }

    // ===========================================================================
    // Wiring
    // ===========================================================================

    public static final StateGraph workflow;
    static {
        graph.add_node("load_context", LangGraphWorkflow::load_context);
        graph.add_node("extract_claim", LangGraphWorkflow::extract_claim);
        graph.add_node("analyze_images", LangGraphWorkflow::analyze_images);
        graph.add_node("evidence_check", LangGraphWorkflow::evidence_check);
        graph.add_node("reconcile", LangGraphWorkflow::reconcile);
        graph.add_node("force_nei", LangGraphWorkflow::force_nei);
        graph.add_node("risk_merge", LangGraphWorkflow::risk_merge);
        graph.add_node("finalize", LangGraphWorkflow::finalize);

        graph.add_edge(StateGraph.START, "load_context");
        graph.add_edge("load_context", "extract_claim");
        graph.add_edge("extract_claim", "analyze_images");
        graph.add_conditional_edges("analyze_images", LangGraphWorkflow::usable_gate, Map.of(
                "evidence_check", "evidence_check",
                "reconcile", "reconcile",
                "force_nei", "force_nei"
        ));
        graph.add_edge("evidence_check", "reconcile");
        graph.add_edge("reconcile", "risk_merge");
        graph.add_edge("force_nei", "risk_merge");
        graph.add_edge("risk_merge", "finalize");
        graph.add_edge("finalize", StateGraph.END);
        workflow = graph.compile();
    }

    // --- structured-output JSON contracts appended to each system prompt ---
    static final String EXTRACT_SCHEMA = "\n\nReturn ONLY JSON: {\"items\":[{\"object_part\":\"\",\"issue_type\":\"\",\"claimed_severity\":null}],\"summary\":\"\",\"source_language\":\"\",\"manipulation_attempt\":false}";
    static final String ANALYZE_SCHEMA = "\n\nReturn ONLY JSON: {\"findings\":[{\"image_id\":\"\",\"detected_object\":\"\",\"detected_part\":\"\",\"detected_issue\":\"\",\"severity\":\"\",\"flags\":[],\"text_instruction_present\":false,\"usable\":true}]}";
    static final String RECONCILE_SCHEMA = "\n\nReturn ONLY JSON: {\"claim_status\":\"\",\"issue_type\":\"\",\"object_part\":\"\",\"severity\":\"\",\"supporting_image_ids\":[],\"risk_flags\":[],\"justification\":\"\"}";

    static String getenv(String key, String fallback) {
        String v = System.getenv(key);
        if (v == null) v = System.getProperty(key);
        if (v == null) v = ENV.get(key);
        return v != null ? v : fallback;
    }

    private static Map<String, String> loadPrompts() {
        Map<String, String> out = new LinkedHashMap<>();
        try {
            String text;
            try (InputStream in = LangGraphWorkflow.class.getResourceAsStream("/prompts.ini")) {
                if (in != null) text = new String(in.readAllBytes());
                else text = Files.readString(Path.of("prompts.ini"));
            }
            Pattern key = Pattern.compile("^(\\w+)\\s*=\\s*(.*)$");
            String curKey = null; StringBuilder buf = new StringBuilder();
            for (String line : text.split("\n", -1)) {
                if (line.strip().equals("[prompts]") || line.strip().isEmpty()) {
                    if (curKey != null && line.strip().isEmpty()) buf.append("\n");
                    continue;
                }
                Matcher m = key.matcher(line);
                if (m.matches() && !line.startsWith(" ")) {
                    if (curKey != null) out.put(curKey, buf.toString().strip());
                    curKey = m.group(1); buf = new StringBuilder(m.group(2));
                } else if (curKey != null) {
                    buf.append("\n").append(line.strip());
                }
            }
            if (curKey != null) out.put(curKey, buf.toString().strip());
        } catch (Exception e) {
            throw new RuntimeException("could not load prompts.ini", e);
        }
        return out;
    }
}
