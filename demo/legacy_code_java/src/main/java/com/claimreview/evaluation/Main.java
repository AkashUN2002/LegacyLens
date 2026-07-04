package com.claimreview.evaluation;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Evaluation workflow: score an existing predictions CSV against the labeled sample set.
 *
 * Reads a predictions CSV and compares it to the expected-output columns in
 * sample_claims.csv, reporting per-field accuracy, a claim_status confusion matrix,
 * and a severity off-by-one breakdown.
 */
public class Main {

    static final Path BASE = Path.of(System.getProperty("user.dir"));
    static final Path CODE = BASE.getParent();
    static final Path REPO = CODE.getParent();
    static final Path DATASET = REPO.resolve("dataset");

    static final String DEFAULT_PRED = DATASET.resolve("output.csv").toString();
    static final String DEFAULT_LABELS = DATASET.resolve("sample_claims.csv").toString();

    static final List<String> CATEGORICAL = List.of(
            "evidence_standard_met", "valid_image", "claim_status",
            "issue_type", "object_part", "severity");
    static final List<String> SET_FIELDS = List.of("risk_flags", "supporting_image_ids");
    static final List<String> TEXT_FIELDS = List.of("evidence_standard_met_reason", "claim_status_justification");

    static final Map<String, Integer> SEV_ORDER = Map.of("none", 0, "low", 1, "medium", 2, "high", 3);

    static List<Map<String, String>> load_csv(String path) throws IOException {
        for (String enc : new String[]{"utf-8", "cp1252", "latin-1"}) {
            try (Reader r = Files.newBufferedReader(Path.of(path), Charset.forName(enc));
                 CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build().parse(r)) {
                List<Map<String, String>> out = new ArrayList<>();
                for (CSVRecord rec : parser) out.add(rec.toMap());
                return out;
            } catch (Exception e) {
                // try next encoding
            }
        }
        throw new IOException("could not decode " + path);
    }

    static String lc(String x) {
        return (x == null ? "" : x).strip().toLowerCase();
    }

    static java.util.Set<String> as_set(String x) {
        java.util.Set<String> items = new java.util.HashSet<>();
        for (String t : (x == null ? "" : x).split(";")) if (!t.strip().isEmpty()) items.add(t.strip().toLowerCase());
        items.remove("none"); items.remove("");
        return items;
    }

    static String key(Map<String, String> row) {
        return row.getOrDefault("user_id", "") + "\u0000" + row.getOrDefault("image_paths", "");
    }

    static Map<String, Object> score(List<Map<String, String>> pred_rows, List<Map<String, String>> gold_rows) {
        Map<String, Map<String, String>> gmap = new LinkedHashMap<>();
        for (Map<String, String> g : gold_rows) gmap.put(key(g), g);

        Map<String, Integer> cat_hits = new LinkedHashMap<>(), cat_total = new LinkedHashMap<>();
        for (String c : CATEGORICAL) { cat_hits.put(c, 0); cat_total.put(c, 0); }
        Map<String, Integer> set_hits = new LinkedHashMap<>(), set_total = new LinkedHashMap<>();
        for (String c : SET_FIELDS) { set_hits.put(c, 0); set_total.put(c, 0); }
        Map<String, Integer> text_present = new LinkedHashMap<>();
        for (String c : TEXT_FIELDS) text_present.put(c, 0);

        Map<String, Map<String, Integer>> confusion = new TreeMap<>();
        int sev_adjacent = 0, sev_far = 0, fully_correct = 0, aligned = 0, missing = 0;

        for (Map<String, String> p : pred_rows) {
            Map<String, String> g = gmap.get(key(p));
            if (g == null) { missing++; continue; }
            aligned++;
            boolean row_all_correct = true;
            for (String c : CATEGORICAL) {
                String gv = lc(g.get(c));
                if (gv.isEmpty()) continue;
                cat_total.merge(c, 1, Integer::sum);
                if (lc(p.get(c)).equals(gv)) cat_hits.merge(c, 1, Integer::sum);
                else row_all_correct = false;
            }
            for (String c : SET_FIELDS) {
                if (lc(g.get(c)).isEmpty() && lc(p.get(c)).isEmpty()) continue;
                set_total.merge(c, 1, Integer::sum);
                if (as_set(p.get(c)).equals(as_set(g.get(c)))) set_hits.merge(c, 1, Integer::sum);
            }
            for (String c : TEXT_FIELDS) if (!(p.get(c) == null ? "" : p.get(c)).strip().isEmpty()) text_present.merge(c, 1, Integer::sum);

            confusion.computeIfAbsent(lc(g.get("claim_status")), k -> new TreeMap<>()).merge(lc(p.get("claim_status")), 1, Integer::sum);
            String gv_s = lc(g.get("severity")), pv_s = lc(p.get("severity"));
            if (!gv_s.equals(pv_s) && SEV_ORDER.containsKey(gv_s) && SEV_ORDER.containsKey(pv_s)) {
                if (Math.abs(SEV_ORDER.get(gv_s) - SEV_ORDER.get(pv_s)) == 1) sev_adjacent++; else sev_far++;
            }
            if (row_all_correct) fully_correct++;
        }

        Map<String, Object> m = new LinkedHashMap<>();
        m.put("aligned", aligned); m.put("missing", missing); m.put("fully_correct", fully_correct);
        m.put("cat_hits", cat_hits); m.put("cat_total", cat_total);
        m.put("set_hits", set_hits); m.put("set_total", set_total);
        m.put("text_present", text_present); m.put("confusion", confusion);
        m.put("sev_adjacent", sev_adjacent); m.put("sev_far", sev_far);
        return m;
    }

    static int pct(int h, int t) {
        return t != 0 ? Math.round(100f * h / t) : 0;
    }

    @SuppressWarnings("unchecked")
    static void report(Map<String, Object> m, String pred_path, String labels_path) {
        var cat_hits = (Map<String, Integer>) m.get("cat_hits");
        var cat_total = (Map<String, Integer>) m.get("cat_total");
        var set_hits = (Map<String, Integer>) m.get("set_hits");
        var set_total = (Map<String, Integer>) m.get("set_total");
        var text_present = (Map<String, Integer>) m.get("text_present");
        var confusion = (Map<String, Map<String, Integer>>) m.get("confusion");

        System.out.println("\nPredictions : " + pred_path);
        System.out.println("Labels      : " + labels_path);
        System.out.println("Aligned rows: " + m.get("aligned") + "   (unmatched predictions: " + m.get("missing") + ")");

        System.out.println("\n=== Categorical field accuracy ===");
        List<Integer> cat_pcts = new ArrayList<>();
        for (String c : CATEGORICAL) {
            int h = cat_hits.get(c), t = cat_total.get(c);
            cat_pcts.add(pct(h, t));
            System.out.printf("  %-24s %3d/%-3d  %3d%%%n", c, h, t, pct(h, t));
        }
        int mean = cat_pcts.isEmpty() ? 0 : Math.round((float) cat_pcts.stream().mapToInt(Integer::intValue).sum() / cat_pcts.size());
        System.out.printf("  %-24s %7s %3d%%%n", "MEAN", "", mean);
        System.out.println("  fully-correct rows (all categorical): " + m.get("fully_correct") + "/" + m.get("aligned"));

        System.out.println("\n=== Set-based field accuracy (order-independent) ===");
        for (String c : SET_FIELDS) {
            int h = set_hits.get(c), t = set_total.get(c);
            System.out.printf("  %-24s %3d/%-3d  %3d%%%n", c, h, t, pct(h, t));
        }

        System.out.println("\n=== claim_status confusion (rows: gold -> pred) ===");
        List<String> statuses = List.of("supported", "contradicted", "not_enough_information");
        StringBuilder header = new StringBuilder("  gold \\ pred         ");
        for (String s : statuses) header.append(String.format("%14s", s.substring(0, Math.min(12, s.length()))));
        System.out.println(header);
        for (String gs : statuses) {
            StringBuilder line = new StringBuilder(String.format("  %-20s", gs));
            for (String ps : statuses) line.append(String.format("%14d", confusion.getOrDefault(gs, Map.of()).getOrDefault(ps, 0)));
            System.out.println(line);
        }

        System.out.println("\n=== severity error profile ===");
        System.out.println("  off-by-one (adjacent level, e.g. medium<->high): " + m.get("sev_adjacent"));
        System.out.println("  off-by-more or to/from unknown                : " + m.get("sev_far"));
        System.out.println("  (adjacent-level severity misses are largely label subjectivity)");

        System.out.println("\n=== free-text completeness (not accuracy) ===");
        for (String c : TEXT_FIELDS) System.out.printf("  %-30s non-empty in %d/%s rows%n", c, text_present.get(c), m.get("aligned"));
    }

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws IOException {
        String pred_path = args.length > 0 ? args[0] : DEFAULT_PRED;
        String labels_path = args.length > 1 ? args[1] : DEFAULT_LABELS;

        var pred_rows = load_csv(pred_path);
        var gold_rows = load_csv(labels_path);
        var m = score(pred_rows, gold_rows);
        report(m, pred_path, labels_path);

        try {
            Path out = BASE.resolve("metrics.json");
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("aligned", m.get("aligned"));
            summary.put("fully_correct", m.get("fully_correct"));
            var cat_hits = (Map<String, Integer>) m.get("cat_hits");
            var cat_total = (Map<String, Integer>) m.get("cat_total");
            Map<String, Object> categorical = new LinkedHashMap<>();
            for (String c : CATEGORICAL) categorical.put(c, Map.of("hits", cat_hits.get(c), "total", cat_total.get(c), "pct", pct(cat_hits.get(c), cat_total.get(c))));
            summary.put("categorical", categorical);
            var set_hits = (Map<String, Integer>) m.get("set_hits");
            var set_total = (Map<String, Integer>) m.get("set_total");
            Map<String, Object> set_fields = new LinkedHashMap<>();
            for (String c : SET_FIELDS) set_fields.put(c, Map.of("hits", set_hits.get(c), "total", set_total.get(c), "pct", pct(set_hits.get(c), set_total.get(c))));
            summary.put("set_fields", set_fields);
            summary.put("claim_status_confusion", m.get("confusion"));
            summary.put("severity_adjacent", m.get("sev_adjacent"));
            summary.put("severity_far", m.get("sev_far"));
            new ObjectMapper().writerWithDefaultPrettyPrinter().writeValue(out.toFile(), summary);
            System.out.println("\nWrote metrics to " + out);
        } catch (IOException ignored) {
        }
    }
}
