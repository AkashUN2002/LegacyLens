package com.claimreview;

import com.claimreview.State.ClaimObject;
import com.claimreview.State.ClaimStatus;
import com.claimreview.State.ClaimedItem;
import com.claimreview.State.ExtractedClaim;
import com.claimreview.State.ImageFinding;
import com.claimreview.State.IssueType;
import com.claimreview.State.RiskFlag;
import com.claimreview.State.Severity;
import dev.langchain4j.data.image.Image;
import dev.langchain4j.data.message.ImageContent;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Pure, model-free helpers for the claim-review pipeline.
 *
 *   - build_evidence_index / build_history_index : load reference CSVs once, at startup
 *   - load_image_block / default_finding / rebuild_findings : analyze_images support
 */
public final class Utility {

    private Utility() {}

    public static final int MAX_EDGE = 2048;
    public static final int MAX_BYTES = 4_500_000;
    public static final Set<String> BEDROCK_OK = Set.of("image/jpeg", "image/png", "image/webp", "image/gif");

    // ---------------------------------------------------------------------------
    // Reference-data indexes (built ONCE before the workflow loop, then passed
    // into each graph.invoke payload as `history` and `evidence_reqs`).
    // ---------------------------------------------------------------------------

    /** Group requirements by object; each bucket includes the universal 'all' rows. */
    public static Map<String, List<Map<String, String>>> build_evidence_index(String path) throws IOException {
        Map<String, List<Map<String, String>>> by_object = new LinkedHashMap<>();
        List<Map<String, String>> all_rows = new ArrayList<>();
        try (Reader r = Files.newBufferedReader(Path.of(path));
             CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build().parse(r)) {
            for (CSVRecord rec : parser) {
                Map<String, String> row = rec.toMap();
                String obj = row.get("claim_object").trim().toLowerCase();
                if (obj.equals("all")) {
                    all_rows.add(row);
                } else {
                    by_object.computeIfAbsent(obj, k -> new ArrayList<>()).add(row);
                }
            }
        }
        Map<String, List<Map<String, String>>> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<Map<String, String>>> e : by_object.entrySet()) {
            List<Map<String, String>> rows = new ArrayList<>(e.getValue());
            rows.addAll(all_rows);
            out.put(e.getKey(), rows);
        }
        return out;
    }

    /** Map user_id -> its single history row (dict). */
    public static Map<String, Map<String, String>> build_history_index(String path) throws IOException {
        Map<String, Map<String, String>> out = new LinkedHashMap<>();
        try (Reader r = Files.newBufferedReader(Path.of(path));
             CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build().parse(r)) {
            for (CSVRecord rec : parser) {
                Map<String, String> row = rec.toMap();
                out.put(row.get("user_id"), row);
            }
        }
        return out;
    }

    // ---------------------------------------------------------------------------
    // analyze_images helpers
    // ---------------------------------------------------------------------------

    /** Detect image type from magic bytes. File extensions lie here: many .jpg
     * files in the dataset are actually WebP or PNG, which Bedrock rejects if the
     * declared media type does not match the bytes. */
    static String _sniff_mime(byte[] data, String fallback) {
        if (data.length >= 3 && (data[0] & 0xFF) == 0xFF && (data[1] & 0xFF) == 0xD8 && (data[2] & 0xFF) == 0xFF)
            return "image/jpeg";
        if (data.length >= 8 && (data[0] & 0xFF) == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G'
                && data[4] == '\r' && data[5] == '\n' && (data[6] & 0xFF) == 0x1a && data[7] == '\n')
            return "image/png";
        if (data.length >= 12 && data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F'
                && data[8] == 'W' && data[9] == 'E' && data[10] == 'B' && data[11] == 'P')
            return "image/webp";
        if (data.length >= 6 && data[0] == 'G' && data[1] == 'I' && data[2] == 'F' && data[3] == '8'
                && (data[4] == '7' || data[4] == '9') && data[5] == 'a')
            return "image/gif";
        return fallback;
    }

    /** Return a Bedrock-safe multimodal image block. Pass through recognized,
     * in-spec, under-size images untouched; otherwise normalize via ImageIO to JPEG
     * (handles unknown formats, oversize files, and corrupt headers). */
    public static ImageContent load_image_block(String rel_path, String image_root) throws IOException {
        Path full = Path.of(image_root, rel_path);
        byte[] raw = Files.readAllBytes(full);

        String mime = _sniff_mime(raw, "");

        // fast path: recognized format AND within size -> send original bytes
        if (BEDROCK_OK.contains(mime) && raw.length <= MAX_BYTES) {
            String data = Base64.getEncoder().encodeToString(raw);
            return ImageContent.from(Image.builder().base64Data(data).mimeType(mime).build());
        }

        // everything else: decode whatever it is and re-encode as JPEG
        BufferedImage img = ImageIO.read(new ByteArrayInputStream(raw));
        if (img == null) throw new IOException("could not decode image: " + rel_path);
        int w = img.getWidth(), h = img.getHeight();
        if (Math.max(w, h) > MAX_EDGE) {
            double s = MAX_EDGE / (double) Math.max(w, h);
            img = resize(img, (int) (w * s), (int) (h * s));
        }
        byte[] data = raw;
        for (int quality : new int[]{85, 70, 55, 40}) {
            data = writeJpeg(img, quality / 100f);
            if (data.length <= MAX_BYTES) break;
        }
        String b64 = Base64.getEncoder().encodeToString(data);
        return ImageContent.from(Image.builder().base64Data(b64).mimeType("image/jpeg").build());
    }

    private static BufferedImage resize(BufferedImage src, int w, int h) {
        BufferedImage dst = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = dst.createGraphics();
        g.drawImage(src, 0, 0, w, h, null);
        g.dispose();
        return dst;
    }

    private static byte[] writeJpeg(BufferedImage img, float quality) throws IOException {
        BufferedImage rgb = img;
        if (img.getType() != BufferedImage.TYPE_INT_RGB) {
            rgb = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
            Graphics2D g = rgb.createGraphics();
            g.drawImage(img, 0, 0, null);
            g.dispose();
        }
        ImageWriter writer = ImageIO.getImageWritersByFormatName("jpeg").next();
        ImageWriteParam param = writer.getDefaultWriteParam();
        param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
        param.setCompressionQuality(quality);
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        try (ImageOutputStream ios = ImageIO.createImageOutputStream(buf)) {
            writer.setOutput(ios);
            writer.write(null, new IIOImage(rgb, null, null), param);
        } finally {
            writer.dispose();
        }
        return buf.toByteArray();
    }

    /** Safe 'unusable' finding used when an image fails to load or the model omits it. */
    public static ImageFinding default_finding(String image_id) {
        ImageFinding f = new ImageFinding();
        f.image_id = image_id;
        f.detected_object = "unclear";
        f.detected_part = "unknown";
        f.detected_issue = IssueType.unknown;
        f.severity = Severity.unknown;
        f.flags = new ArrayList<>();
        f.text_instruction_present = false;
        f.usable = false;
        return f;
    }

    /** Force exactly one finding per input image, correct id, defaults for any gap.
     * Guards against the model dropping, reordering, renaming, or inventing findings. */
    public static List<ImageFinding> rebuild_findings(List<String> ordered_ids, Map<String, ImageFinding> by_id, Set<String> load_errors) {
        List<ImageFinding> out = new ArrayList<>();
        for (String iid : ordered_ids) {
            if (load_errors.contains(iid) || !by_id.containsKey(iid)) {
                out.add(default_finding(iid));
            } else {
                ImageFinding f = by_id.get(iid);
                f.image_id = iid;          // correct any id drift from the model
                out.add(f);
            }
        }
        return out;
    }

    // ---------------------------------------------------------------------------
    // evidence_check helpers (pure, no model)
    // ---------------------------------------------------------------------------

    /** Lowercase and unify separators: 'Front Bumper' / 'front-bumper' -> 'front_bumper'. */
    public static String normalize(String s) {
        return (s == null ? "" : s).trim().toLowerCase().replace("-", "_").replace(" ", "_");
    }

    /** Get the plain string for a ClaimObject enum or raw string ('car'). */
    public static String object_value(Object claim_object) {
        if (claim_object instanceof ClaimObject co) return co.getValue().trim().toLowerCase();
        return String.valueOf(claim_object).trim().toLowerCase();
    }

    /** Tolerant part match. True when a usable image plausibly shows the claimed part. */
    public static boolean part_matches(String claimed, String detected) {
        String c = normalize(claimed), d = normalize(detected);
        if (c.isEmpty() || d.isEmpty() || c.equals("unknown") || d.equals("unknown")) return false;
        if (c.equals(d)) return true;
        Set<String> GENERIC = Set.of("body", "car", "laptop", "package", "box", "exterior", "object");
        if (GENERIC.contains(d)) return true;
        if (c.contains(d) || d.contains(c)) return true;
        Set<String> POS = Set.of("left", "right", "front", "rear", "back", "upper", "lower", "top", "bottom");
        Set<String> STOP = new java.util.HashSet<>(POS);
        STOP.addAll(Set.of("side", "the", "a", "of", "outer", "inner"));
        String[] cp = c.split("_"), dp = d.split("_");
        Set<String> c_pos = new java.util.HashSet<>();
        for (String t : cp) if (POS.contains(t)) c_pos.add(t);
        Set<String> d_pos = new java.util.HashSet<>();
        for (String t : dp) if (POS.contains(t)) d_pos.add(t);
        if (!c_pos.isEmpty() && !d_pos.isEmpty()) {
            Set<String> inter = new java.util.HashSet<>(c_pos); inter.retainAll(d_pos);
            if (inter.isEmpty()) return false;
        }
        Set<String> ct = new java.util.HashSet<>();
        for (String t : cp) if (!t.isEmpty() && !STOP.contains(t)) ct.add(t);
        Set<String> dt = new java.util.HashSet<>();
        for (String t : dp) if (!t.isEmpty() && !STOP.contains(t)) dt.add(t);
        ct.retainAll(dt);
        return !ct.isEmpty();
    }

    /** Best-effort: choose the requirement row most relevant to this claim, by token
     * overlap. Used ONLY to cite a requirement_id; never affects the boolean. */
    public static Map<String, String> pick_requirement(ClaimObject claim_object, ExtractedClaim claim, List<Map<String, String>> evidence_reqs) {
        String obj = object_value(claim_object);
        Set<String> toks = new java.util.HashSet<>();
        if (claim != null && claim.items != null) {
            for (ClaimedItem it : claim.items) {
                toks.addAll(List.of(normalize(it.issue_type).split("_")));
                toks.addAll(List.of(normalize(it.object_part).split("_")));
            }
        }
        Map<String, String> best = null;
        double best_score = -1.0;
        for (Map<String, String> req : (evidence_reqs == null ? List.<Map<String, String>>of() : evidence_reqs)) {
            String row_obj = (req.getOrDefault("claim_object", "")).trim().toLowerCase();
            if (!row_obj.equals(obj) && !row_obj.equals("all")) continue;
            Set<String> applies = new java.util.HashSet<>(List.of(normalize(req.getOrDefault("applies_to", "")).split("_")));
            Set<String> inter = new java.util.HashSet<>(toks); inter.retainAll(applies);
            double score = inter.size() + (row_obj.equals(obj) ? 0.5 : 0.0);
            if (score > best_score) { best = req; best_score = score; }
        }
        return best;
    }

    // ---------------------------------------------------------------------------
    // reconcile / force_nei / risk_merge / finalize helpers (pure, no model)
    // ---------------------------------------------------------------------------

    // Output CSV column order (matches sample_claims.csv exactly)
    public static final List<String> OUTPUT_COLUMNS = List.of(
            "user_id", "image_paths", "user_claim", "claim_object",
            "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
            "issue_type", "object_part", "claim_status", "claim_status_justification",
            "supporting_image_ids", "valid_image", "severity"
    );

    /** Return the underlying string for an enum member, or x unchanged. */
    public static String enum_value(Object x) {
        if (x instanceof ClaimObject m) return m.getValue();
        if (x instanceof ClaimStatus m) return m.getValue();
        if (x instanceof IssueType m) return m.getValue();
        if (x instanceof Severity m) return m.getValue();
        if (x instanceof RiskFlag m) return m.getValue();
        return x == null ? null : String.valueOf(x);
    }

    /** Snap a value (enum or raw string) to ClaimStatus, else default. */
    public static ClaimStatus coerce_enum(Object value, ClaimStatus default_) {
        if (value instanceof ClaimStatus cs) return cs;
        try { return ClaimStatus.from(enum_value(value)); } catch (Exception e) { return default_; }
    }

    /** Snap a value (enum or raw string) to IssueType, else default. */
    public static IssueType coerce_enum(Object value, IssueType default_) {
        if (value instanceof IssueType it) return it;
        try { return IssueType.from(enum_value(value)); } catch (Exception e) { return default_; }
    }

    /** Snap a value (enum or raw string) to Severity, else default. */
    public static Severity coerce_enum(Object value, Severity default_) {
        if (value instanceof Severity s) return s;
        try { return Severity.from(enum_value(value)); } catch (Exception e) { return default_; }
    }

    /** Convert a mix of RiskFlag members / strings into valid RiskFlag members,
     * dropping anything that is not a recognized flag. */
    public static List<RiskFlag> to_risk_flags(List<?> tokens) {
        List<RiskFlag> out = new ArrayList<>();
        for (Object t : (tokens == null ? List.of() : tokens)) {
            if (t instanceof RiskFlag rf) { out.add(rf); continue; }
            try { out.add(RiskFlag.from(normalize(String.valueOf(t)))); } catch (Exception ignored) {}
        }
        return out;
    }

    /** Human-readable cause string for force_nei reasons. */
    public static String summarize_unusable(List<ImageFinding> findings) {
        Set<String> causes = new TreeSet<>();
        for (ImageFinding f : findings) {
            for (RiskFlag fl : f.flags) causes.add(enum_value(fl));
            String dobj = normalize(f.detected_object);
            if (dobj.equals("other") || dobj.equals("unclear")) causes.add("no clear inspectable object");
        }
        return causes.isEmpty() ? "no inspectable image" : String.join(", ", causes);
    }
}
