package com.claimreview;

import com.claimreview.State.ClaimState;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Driver: run every claim in an input CSV through the LangGraph workflow and write output.csv.
 *
 * Each input row becomes one workflow.invoke; the finalize node's `output_row` is collected
 * and written out in the exact 14-column order. A failure on any single row is caught and
 * emitted as a safe not_enough_information row so one bad claim never kills the batch.
 */
public class Main {

    // --- anchor paths to this file so relative cwd never matters ---
    static final Path BASE = Path.of(System.getProperty("user.dir"));
    static final Path DATASET = BASE.getParent().resolve("dataset");

    static {
        // The workflow reads IMAGE_ROOT at class-load -> set BEFORE referencing it.
        if (System.getProperty("IMAGE_ROOT") == null) System.setProperty("IMAGE_ROOT", DATASET.toString());
    }

    /** Safe NEI row (echoes inputs) used when a single claim raises. */
    static Map<String, String> fallback_row(Map<String, String> row, String err) {
        Map<String, String> out = new LinkedHashMap<>();
        out.put("user_id", row.getOrDefault("user_id", ""));
        out.put("image_paths", row.getOrDefault("image_paths", ""));
        out.put("user_claim", row.getOrDefault("user_claim", ""));
        out.put("claim_object", row.getOrDefault("claim_object", "").strip().toLowerCase());
        out.put("evidence_standard_met", "false");
        out.put("evidence_standard_met_reason", "pipeline error: " + err);
        out.put("risk_flags", "manual_review_required");
        out.put("issue_type", "unknown");
        out.put("object_part", "unknown");
        out.put("claim_status", "not_enough_information");
        out.put("claim_status_justification", "Processing failed; defaulting to not_enough_information.");
        out.put("supporting_image_ids", "none");
        out.put("valid_image", "false");
        out.put("severity", "unknown");
        return out;
    }

    @SuppressWarnings("unchecked")
    static void run(String input_csv, String output_csv) throws IOException {
        // reference data: built ONCE, passed into each invoke (resolved outside the graph)
        var evidence_index = Utility.build_evidence_index(DATASET.resolve("evidence_requirements.csv").toString());
        var history_index = Utility.build_history_index(DATASET.resolve("user_history.csv").toString());

        List<Map<String, String>> rows = new ArrayList<>();
        try (Reader r = Files.newBufferedReader(Path.of(input_csv));
             CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build().parse(r)) {
            for (CSVRecord rec : parser) rows.add(rec.toMap());
        }

        List<Map<String, Object>> results = new ArrayList<>();
        int i = 0;
        for (Map<String, String> row : rows) {
            i++;
            String claim_object = (row.get("claim_object") == null ? "" : row.get("claim_object")).strip().toLowerCase();
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("user_id", row.get("user_id"));
            payload.put("image_paths", row.get("image_paths"));
            payload.put("user_claim", row.get("user_claim"));
            payload.put("claim_object", row.get("claim_object"));
            payload.put("history", history_index.getOrDefault(row.get("user_id"), new LinkedHashMap<>()));
            payload.put("evidence_reqs", evidence_index.getOrDefault(claim_object, new ArrayList<>()));

            Map<String, Object> output_row;
            try {
                ClaimState final_state = LangGraphWorkflow.workflow.invoke(payload);
                output_row = (Map<String, Object>) final_state.get("output_row");
            } catch (Exception e) {
                output_row = new LinkedHashMap<>(fallback_row(row, e.toString()));
            }

            results.add(output_row);
            System.out.printf("[%d/%d] %s %-8s -> %s (valid_image=%s)%n",
                    i, rows.size(), row.get("user_id"), claim_object,
                    output_row.get("claim_status"), output_row.get("valid_image"));
        }

        try (Writer w = Files.newBufferedWriter(Path.of(output_csv));
             CSVPrinter printer = new CSVPrinter(w, CSVFormat.DEFAULT.builder()
                     .setHeader(Utility.OUTPUT_COLUMNS.toArray(new String[0])).build())) {
            for (Map<String, Object> rr : results) {
                List<Object> vals = new ArrayList<>();
                for (String c : Utility.OUTPUT_COLUMNS) vals.add(rr.getOrDefault(c, ""));
                printer.printRecord(vals);
            }
        }

        System.out.println("\nWrote " + results.size() + " rows to " + output_csv);
    }

    public static void main(String[] args) throws IOException {
        String INPUT_CSV = args.length > 0 ? args[0] : DATASET.resolve("claims.csv").toString();
        String OUTPUT_CSV = args.length > 1 ? args[1] : DATASET.resolve("output.csv").toString();
        run(INPUT_CSV, OUTPUT_CSV);
    }
}
